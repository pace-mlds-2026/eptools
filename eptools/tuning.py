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
from optuna.storages import RDBStorage

from eptools.modelling import (
    get_collision_sales_df, get_sku_active_windows, build_full_panel, classify_sb,
    segment_scope, run_backtest, pooled_wmape,
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
                   min_train_months, objective_metric, cold_start_strategy="zero"):
    """Build the Optuna objective for one class's panel.

    Records BOTH pooled and 3-month-rolling WMAPE on every trial regardless of
    which one is optimised, so reporting against the scope's rolling metric
    never needs a re-run.

    cold_start_strategy="zero": run_backtest raises by default
    (cold_start_strategy="error") on any SKU that launches partway through
    the backtest with no prediction available for it yet. The real panel has
    thousands of such fold-SKU cases across a full backtest, so the strict
    default would abort almost every trial immediately. "zero" scores those
    as an explicit zero forecast instead, matching this module's older,
    pre-strict-validation behaviour.
    """
    def objective(trial):
        params = search_space(trial)
        results = run_backtest(
            forecast_fn, panel, panel_windows,
            params=params, min_train_months=min_train_months,
            cold_start_strategy=cold_start_strategy,
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



# Postgres engine settings that keep us well under Supabase's session-pooler
# client cap (free tier: 15 total). A SINGLE shared engine with a small pool,
# reused across every study, is the whole trick -- passing a URL string to
# create_study instead builds a fresh engine (and pool) per call and leaks them.
_PG_ENGINE_KWARGS = {"pool_size": 2, "max_overflow": 3,
                     "pool_pre_ping": True, "pool_recycle": 300}


def _build_storage(storage):
    """Return (storage_obj, owns). A URL string is wrapped in ONE RDBStorage
    (capped pool for Postgres); an existing storage object is passed through
    untouched. `owns` says whether we created the engine and must dispose it."""
    if storage is None:
        storage = "sqlite:///optuna_tuning.db"
    if isinstance(storage, str):
        kw = _PG_ENGINE_KWARGS if storage.startswith(("postgresql", "postgres")) else {}
        return RDBStorage(url=storage, engine_kwargs=kw), True
    return storage, False


def tune_over_sb_classes(forecast_fn, search_space, *, model_name,
                         storage=None, n_trials=30, sb_classes=SB_CLASSES,
                         min_train_months=12, objective_metric="pooled",
                         direction="minimize", sampler=None,
                         cold_start_strategy="zero",
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
        the CONSUMING notebook. Use the SESSION POOLER (host
        aws-0-<region>.pooler.supabase.com, user postgres.<ref>, port 5432): it
        is IPv4 (the direct host db.<ref>.supabase.co is IPv6-only and often
        won't resolve) and session-mode, so Optuna's prepared statements are
        safe. Do NOT use the Transaction pooler (port 6543). If None, falls back
        to a local sqlite file so the code runs offline.
    objective_metric : {"pooled", "rolling3"}
        Which WMAPE the search minimises. Both are always recorded per trial.
    cold_start_strategy : {"zero", "error"}
        Passed through to run_backtest via make_objective. run_backtest's own
        default is "error" -- raise on any SKU that launches partway through
        a backtest with no prediction yet available for it. The real panel
        has thousands of these fold-SKU cases across a full backtest, so
        that default would abort almost every trial immediately here. "zero"
        (this function's default) scores them as an explicit zero forecast
        instead.

    Routing notes
    -------------
    Each class's study is scored via segment_scope(forecast_fn, [sb_class]),
    which reclassifies every fold from that fold's own train_df -- so a SKU
    that drifts class over its life (e.g. Lumpy -> Smooth) is scored under
    whichever class it actually was at that fold's origin, not whatever it
    ends up classified as over its full history. sb_lookup (retrospective,
    whole-panel classify_sb) is used ONLY to report a current SKU count per
    class -- never to decide which SKUs a fold forecasts. See classify_sb's
    own docstring: the retrospective label is for slicing/reporting, not
    routing.

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

    if storage is None and verbose:
        print("[tuning] no storage given -> local fallback: sqlite:///optuna_tuning.db")

    # ONE shared, pool-capped storage reused by every study (see _build_storage).
    storage_obj, owns_storage = _build_storage(storage)
    full_panel, windows, sb_lookup = load_data_cached()

    studies, skipped = {}, {}
    try:
      for sb_class in sb_classes:
          # Reporting only -- see "Routing notes" above. Actual per-fold
          # routing happens inside segment_scope via point-in-time classify_sb.
          n_skus_now = int((sb_lookup["sb_class"] == sb_class).sum())
          scoped_forecast_fn = segment_scope(forecast_fn, allowed_classes=[sb_class])

          study = optuna.create_study(
              study_name=build_study_name(model_name, sb_class, min_train_months, dataset_tag),
              storage=storage_obj, direction=direction, load_if_exists=load_if_exists,
              sampler=sampler or optuna.samplers.TPESampler(seed=seed),
          )
          study.set_user_attr("model_name", model_name)
          study.set_user_attr("sb_class", sb_class)
          study.set_user_attr("min_train_months", min_train_months)
          study.set_user_attr("objective_metric", objective_metric)
          study.set_user_attr("n_skus", n_skus_now)

          objective = make_objective(scoped_forecast_fn, search_space, full_panel, windows,
                                     min_train_months, objective_metric,
                                     cold_start_strategy=cold_start_strategy)

          # catch=(Exception,) => a failing trial is marked FAILED rather than
          # aborting the whole sweep (e.g. a class too sparse to make folds).
          study.optimize(objective, n_trials=n_trials, catch=(Exception,))

          complete = [t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE]
          if complete:
              studies[sb_class] = study
              if verbose:
                  print(f"  {sb_class:>13}: {n_skus_now:>4} SKUs | "
                        f"best {objective_metric} WMAPE={study.best_value:.4f} | "
                        f"{study.best_params}")
          else:
              skipped[sb_class] = "no successful trials (class likely too sparse for folds)"
              if verbose:
                  print(f"  {sb_class:>13}: SKIPPED -- {skipped[sb_class]}")

    finally:
        if owns_storage:
            # release pooled connections so we don't hold session-mode clients
            try:
                storage_obj.engine.dispose()
            except Exception:
                pass

    return TuningResult(model_name, studies, skipped)


def summarise_all_studies(storage, winners_only=False, sort_by="best_value"):
    """Pull EVERY study from ``storage`` and return a best-per-(model, class)
    leaderboard as a DataFrame.

    Complements the Optuna dashboard: the dashboard is strong on per-study
    drill-down but weak at comparing many studies (model x class) side by side.
    Point this at the same connection string the sweeps wrote to.

    Reads the study-level user attrs that tune_over_sb_classes stamps
    (model_name, sb_class, objective_metric, n_skus); for any study missing them
    it falls back to parsing the build_study_name convention
    (``model__class__mtmXX__tag``). Studies with no completed trial appear with
    null metrics rather than being dropped.

    winners_only : if True, collapse to the single best study per sb_class.
    """
    storage_obj, owns_storage = _build_storage(storage)
    try:
        summaries = optuna.get_all_study_summaries(storage_obj)
    finally:
        if owns_storage:
            try:
                storage_obj.engine.dispose()
            except Exception:
                pass
    rows = []
    for s in summaries:
        ua = s.user_attrs or {}
        model, sb_class = ua.get("model_name"), ua.get("sb_class")
        if model is None or sb_class is None:
            parts = s.study_name.split("__")
            model = model or (parts[0] if len(parts) > 0 else None)
            sb_class = sb_class or (parts[1] if len(parts) > 1 else None)
        bt = s.best_trial
        rows.append({
            "model": model,
            "sb_class": sb_class,
            "objective_metric": ua.get("objective_metric"),
            "best_value": (bt.value if bt is not None else None),
            "pooled_wmape": (bt.user_attrs.get("pooled_wmape") if bt is not None else None),
            "rolling3_wmape": (bt.user_attrs.get("rolling3_wmape") if bt is not None else None),
            "best_params": (bt.params if bt is not None else None),
            "n_trials": s.n_trials,
            "n_skus": ua.get("n_skus"),
            "study_name": s.study_name,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["sb_class", sort_by], na_position="last").reset_index(drop=True)

    if winners_only:
        scored = df.dropna(subset=["best_value"])
        idx = scored.groupby("sb_class")["best_value"].idxmin()
        df = df.loc[idx].sort_values("sb_class").reset_index(drop=True)
    return df
