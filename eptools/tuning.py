"""eptools.tuning -- Optuna hyperparameter search across Syntetos-Boylan classes.

Engine only. A caller (a notebook that consumes eptools) supplies a
contract-compliant ``forecast_fn`` and a ``search_space`` callable; this module
runs one Optuna study per SB class, scoring each trial by WMAPE via the shared
backtest harness, and persists every study to whatever storage you pass in.

Contract recap:
  forecast_fn(train_df, horizon, **params) -> DataFrame[sku_id, month, y_pred]
  search_space(trial)                       -> dict of params
"""

import optuna
import pandas as pd

from eptools.modelling import (
    get_collision_sales_df, get_sku_active_windows, build_full_panel, classify_sb,
    get_skus_by_segment, subset_panel, run_backtest, pooled_wmape,
    rolling_average_wmape, validate_forecast_fn,
)

SB_CLASSES = ("Smooth", "Erratic", "Intermittent", "Lumpy")

_DATA_CACHE = {}


def load_data_cached():
    """Return (full_panel, windows, sb_lookup), memoised for the session.

    Reuses eptools.modelling's own DATA-path auto-discovery (Colab / Mac /
    Windows, or the EPTOOLS_DATA_PATH env var) -- this does NOT reinvent path
    handling. The raw CSV is already cached inside load_dataframes; this adds a
    cache for the derived panel so repeated tuning calls don't rebuild it.
    """
    if "bundle" not in _DATA_CACHE:
        collision = get_collision_sales_df()
        windows = get_sku_active_windows(collision)
        full_panel = build_full_panel(collision, windows)
        sb_lookup = classify_sb(full_panel)
        _DATA_CACHE["bundle"] = (full_panel, windows, sb_lookup)
    return _DATA_CACHE["bundle"]


def build_study_name(model_name, sb_class, min_train_months, dataset_tag="v1"):
    """Deterministic, collision-proof study name. Includes min_train_months and
    a dataset tag so trials scored under incompatible configs never pool into
    one study (e.g. naive_ma__Lumpy__mtm12__v1)."""
    return f"{model_name}__{sb_class}__mtm{min_train_months}__{dataset_tag}"


def make_objective(forecast_fn, search_space, panel, panel_windows,
                   min_train_months, objective_metric):
    """Build the Optuna objective for one class's panel.

    Records BOTH pooled and 3-month-rolling WMAPE on every trial regardless of
    which one is optimised, so reporting against the scope's rolling metric
    never needs a re-run.
    """
    def objective(trial):
        params = search_space(trial)
        results = run_backtest(
            forecast_fn, panel, panel_windows,
            params=params, min_train_months=min_train_months,
        )
        pooled = float(pooled_wmape(results))
        rolling = rolling_average_wmape(results, window=3, method="pooled")
        rolling3 = float(rolling.dropna().iloc[-1]) if rolling.notna().any() else float("nan")
        trial.set_user_attr("pooled_wmape", pooled)
        trial.set_user_attr("rolling3_wmape", rolling3)
        trial.set_user_attr("n_folds", int(len(results)))
        return pooled if objective_metric == "pooled" else rolling3
    return objective


class TuningResult:
    """Per-class Optuna studies plus convenience views."""

    def __init__(self, model_name, studies, skipped):
        self.model_name = model_name
        self.studies = studies      # {sb_class: optuna.Study}
        self.skipped = skipped      # {sb_class: reason}

    @property
    def best_params(self):
        return {c: s.best_params for c, s in self.studies.items()}

    @property
    def results_df(self):
        rows = []
        for c, s in self.studies.items():
            t = s.best_trial
            rows.append({
                "model": self.model_name, "sb_class": c,
                "best_value": s.best_value,
                "pooled_wmape": t.user_attrs.get("pooled_wmape"),
                "rolling3_wmape": t.user_attrs.get("rolling3_wmape"),
                "best_params": s.best_params,
                "n_trials": len(s.trials),
                "skipped_reason": None,
            })
        for c, reason in self.skipped.items():
            rows.append({
                "model": self.model_name, "sb_class": c,
                "best_value": None, "pooled_wmape": None, "rolling3_wmape": None,
                "best_params": None, "n_trials": 0, "skipped_reason": reason,
            })
        return pd.DataFrame(rows)

    def export(self, path):
        """Dump every trial across all classes to parquet. The free Supabase
        tier has no backups, so export anything worth keeping."""
        frames = []
        for c, s in self.studies.items():
            df = s.trials_dataframe()
            df.insert(0, "sb_class", c)
            df.insert(0, "model", self.model_name)
            frames.append(df)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        out.to_parquet(path)
        return path


def tune_over_sb_classes(forecast_fn, search_space, *, model_name,
                         storage=None, n_trials=30, sb_classes=SB_CLASSES,
                         min_train_months=12, objective_metric="pooled",
                         direction="minimize", sampler=None,
                         load_if_exists=True, dataset_tag="v1", seed=42,
                         verbose=True):
    """Tune one contract-compliant model across the SB classes, minimising WMAPE.

    Parameters
    ----------
    forecast_fn : callable
        forecast_fn(train_df, horizon, **params) -> DataFrame[sku_id, month, y_pred]
    search_space : callable
        trial -> params dict (e.g. lambda t: {"window": t.suggest_int("window", 1, 18)})
    model_name : str
        Identity used in study names.
    storage : str or None
        Optuna storage. Pass your Supabase Postgres connection string here from
        the CONSUMING notebook. Use the DIRECT connection (port 5432), NOT the
        transaction pooler -- Optuna's prepared statements break under PgBouncer
        transaction-mode pooling. If None, falls back to a local sqlite file so
        the code runs offline.
    objective_metric : {"pooled", "rolling3"}
        Which WMAPE the search minimises. Both are always recorded per trial.

    Returns
    -------
    TuningResult
    """
    if objective_metric not in ("pooled", "rolling3"):
        raise ValueError("objective_metric must be 'pooled' or 'rolling3'")

    # Fail fast on the y_pred contract before spinning up any studies. A genuine
    # violation (missing/ non-numeric y_pred) raises ValueError and propagates;
    # a model that simply requires params raises TypeError -- skip the pre-check
    # in that case and let the first trial surface any real problem.
    try:
        validate_forecast_fn(forecast_fn)
    except TypeError:
        if verbose:
            print("[tuning] contract pre-check skipped (model needs params); "
                  "first trial will surface any issue.")

    if storage is None:
        storage = "sqlite:///optuna_tuning.db"
        if verbose:
            print(f"[tuning] no storage given -> local fallback: {storage}")

    full_panel, windows, sb_lookup = load_data_cached()

    studies, skipped = {}, {}
    for sb_class in sb_classes:
        skus = get_skus_by_segment(sb_lookup, "sb_class", [sb_class])
        panel, panel_windows = subset_panel(full_panel, windows, skus)

        study = optuna.create_study(
            study_name=build_study_name(model_name, sb_class, min_train_months, dataset_tag),
            storage=storage, direction=direction, load_if_exists=load_if_exists,
            sampler=sampler or optuna.samplers.TPESampler(seed=seed),
        )
        study.set_user_attr("model_name", model_name)
        study.set_user_attr("sb_class", sb_class)
        study.set_user_attr("min_train_months", min_train_months)
        study.set_user_attr("objective_metric", objective_metric)
        study.set_user_attr("n_skus", int(len(skus)))

        objective = make_objective(forecast_fn, search_space, panel, panel_windows,
                                   min_train_months, objective_metric)

        # catch=(Exception,) => a failing trial is marked FAILED rather than
        # aborting the whole sweep (e.g. a class too sparse to make folds).
        study.optimize(objective, n_trials=n_trials, catch=(Exception,))

        complete = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
        if complete:
            studies[sb_class] = study
            if verbose:
                print(f"  {sb_class:>13}: {len(skus):>4} SKUs | "
                      f"best {objective_metric} WMAPE={study.best_value:.4f} | "
                      f"{study.best_params}")
        else:
            skipped[sb_class] = "no successful trials (class likely too sparse for folds)"
            if verbose:
                print(f"  {sb_class:>13}: SKIPPED -- {skipped[sb_class]}")

    return TuningResult(model_name, studies, skipped)
