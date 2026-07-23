"""eptools.tuning -- Optuna hyperparameter search across Syntetos-Boylan classes.

Engine only. A caller (a notebook that consumes eptools) supplies a
contract-compliant ``forecast_fn`` and a ``search_space`` callable; this module
runs one Optuna study per SB class, scoring each trial by WMAPE via the shared
backtest harness, and persists every study to whatever storage you pass in.

Contract recap:
  forecast_fn(train_df, horizon, **params) -> DataFrame[sku_id, month, y_pred]
  search_space(trial)                       -> dict of params
"""

import os

import optuna
import pandas as pd
from optuna.storages import RDBStorage

from eptools.modelling import (
    get_collision_sales_df, get_sku_active_windows, build_full_panel, classify_sb,
    scope_to, run_backtest, pooled_wmape,
    rolling_average_wmape, validate_forecast_fn, _resolve_data_path,
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


def get_optuna_export_db_path(data_path=None) -> str:
    """Return the local filesystem path to supabase_export.db.

    Reuses eptools.modelling's _resolve_data_path for the same Colab / Mac /
    Windows auto-discovery (or the EPTOOLS_DATA_PATH env var) load_dataframes
    uses to find the DATA directory, then descends into OPTUNA_EXPORT -- no
    path-detection logic duplicated here.

    Args:
        data_path: Optional override for the DATA directory (same semantics
            as load_dataframes' data_path). Not the OPTUNA_EXPORT path itself.
    """
    resolved = _resolve_data_path(data_path)
    db_path = os.path.join(resolved, "OPTUNA_EXPORT", "supabase_export.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"supabase_export.db not found at {db_path}")
    return db_path


def build_study_name(model_name, sb_class, min_train_months, dataset_tag="v1"):
    """Deterministic, collision-proof study name. Includes min_train_months and
    a dataset tag so trials scored under incompatible configs never pool into
    one study (e.g. naive_ma__Lumpy__mtm12__v1)."""
    return f"{model_name}__{sb_class}__mtm{min_train_months}__{dataset_tag}"


def make_objective(forecast_fn, search_space, panel, panel_windows,
                   min_train_months, objective_metric, cold_start_strategy="zero",
                   scope_fn=None):
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

    scope_fn: passed straight through to run_backtest. Narrows the fold's
    known-at-origin/test universe to match whatever SKUs forecast_fn is
    actually being asked to predict (e.g. scope_to(sb_class)) -- without
    this, a forecast_fn that only predicts a subset of known SKUs fails
    run_backtest's "every known SKU must get a prediction" check.
    """
    def objective(trial):
        params = search_space(trial)
        results = run_backtest(
            forecast_fn, panel, panel_windows,
            params=params, min_train_months=min_train_months,
            cold_start_strategy=cold_start_strategy,
            scope_fn=scope_fn,
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
    Each class's study is scored via scope_fn=scope_to(sb_class), passed
    through to run_backtest. scope_to() reclassifies every fold from that
    fold's own train_df (via classify_sb), so a SKU that drifts class over
    its life (e.g. Lumpy -> Smooth) is scored under whichever class it
    actually was at that fold's origin, not whatever it ends up classified
    as over its full history. Critically, scope_fn also narrows the fold's
    known-at-origin/test universe to the same class -- forecast_fn is only
    ever asked to predict its own class's SKUs, so run_backtest's "every
    known SKU must get a prediction" check stays satisfied (the older
    segment_scope() wrapper only filtered forecast_fn's inputs/outputs, not
    the fold's expected universe, which made it incompatible with that
    check -- see MODELLING.ipynb's _apply_scope for the fix).
    sb_lookup (retrospective, whole-panel classify_sb) is used ONLY to
    report a current SKU count per class -- never to decide which SKUs a
    fold forecasts. See classify_sb's own docstring: the retrospective
    label is for slicing/reporting, not routing.

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
          # routing happens inside scope_to(sb_class) via point-in-time classify_sb.
          n_skus_now = int((sb_lookup["sb_class"] == sb_class).sum())

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

          objective = make_objective(forecast_fn, search_space, full_panel, windows,
                                     min_train_months, objective_metric,
                                     cold_start_strategy=cold_start_strategy,
                                     scope_fn=scope_to(sb_class))

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


def leave_one_in_arms(feature_fields, param_name, as_list=False):
    """Build {'baseline': {...}, <short_field_name>: {...}, ...} for
    run_feature_ablation: one arm per feature plus a no-feature baseline,
    every arm fixing ONLY the feature parameter. The hyperparameter search
    space is identical and gets the same trial budget for every arm, so an
    arm's best WMAPE is comparable to another arm's rather than confounded
    by how many trials Optuna happened to spend exploring it.

    as_list=True wraps each feature in a single-element list (for params
    that accept a list, e.g. macro_fields=[...]); as_list=False passes the
    field on its own (for params that accept a single value or None, e.g.
    macro_field=...).
    """
    baseline_value = [] if as_list else None
    arms = {"baseline": {param_name: baseline_value}}
    for ff in feature_fields:
        short = ff.split("#")[-1]
        arms[short] = {param_name: [ff] if as_list else ff}
    return arms


def make_pruning_objective(forecast_fn, hp_search_space, panel, panel_windows,
                          min_train_months, objective_metric,
                          fixed_params=None, prune_warmup_folds=15,
                          cold_start_strategy="zero", scope_fn=None):
    """Same contract as make_objective above, plus Optuna pruning wired
    through run_backtest's existing on_fold_complete hook.

    Every trial in a study walks the SAME fold sequence in the SAME order
    (only forecast_fn's params differ), so comparing a trial's running WMAPE
    against other trials in the same study at the same fold index is valid.
    Trials clearly losing by prune_warmup_folds are killed early instead of
    always running all ~50 folds to completion regardless of how bad the
    hyperparameters obviously are.

    cold_start_strategy="zero": run_backtest raises by default
    (cold_start_strategy="error") on any SKU that launches partway through
    the backtest with no prediction available for it yet. The real panel has
    thousands of such fold-SKU cases across a full backtest, so the strict
    default would abort almost every trial immediately. "zero" restores
    scoring those as an explicit zero forecast instead, matching this
    module's older, pre-strict-validation behaviour.

    scope_fn: passed straight through to run_backtest (e.g. scope_to(sb_class)).
    Narrows the fold's known-at-origin/test universe to match whatever SKUs
    forecast_fn actually predicts -- without this, a forecast_fn that only
    covers a subset of known SKUs fails run_backtest's "every known SKU must
    get a prediction" check.

    Every trial's per-fold (target_date, abs_error_sum, abs_actual_sum) is
    stashed in trial.user_attrs as "fold_trajectory" -- NOT just the two
    aggregate scalars. run_backtest's results_df is otherwise discarded once
    pooled/rolling3 are computed, which would silently throw away the one
    thing a "WMAPE over time" comparison chart needs later: without this,
    reconstructing that chart after the sweep finishes means re-running the
    winning configs from scratch. Storing atomic sums (not the precomputed
    per-fold wmape column) matters too -- summing abs_error_sum/abs_actual_sum
    BEFORE dividing reconstructs pooled/rolling WMAPE exactly; averaging
    per-fold wmape values directly does not. Pruned trials keep whatever
    prefix of folds they completed -- a partial trajectory is still useful,
    not nothing.
    """
    fixed_params = fixed_params or {}

    def objective(trial):
        params = {**fixed_params, **hp_search_space(trial)}
        pruned = False

        def on_fold_complete(fold_idx, running_pooled):
            nonlocal pruned
            if fold_idx < prune_warmup_folds:
                return False
            trial.report(running_pooled, step=fold_idx)
            if trial.should_prune():
                pruned = True
                return True  # tells run_backtest to break out of the fold loop
            return False

        results = run_backtest(
            forecast_fn, panel, panel_windows,
            params=params, min_train_months=min_train_months,
            on_fold_complete=on_fold_complete,
            cold_start_strategy=cold_start_strategy,
            scope_fn=scope_fn,
        )
        pooled = float(pooled_wmape(results))
        rolling = rolling_average_wmape(results, window=3, method="pooled")
        rolling3 = float(rolling.dropna().iloc[-1]) if rolling.notna().any() else float("nan")
        trial.set_user_attr("pooled_wmape", pooled)
        trial.set_user_attr("rolling3_wmape", rolling3)
        trial.set_user_attr("n_folds", int(len(results)))
        trial.set_user_attr("pruned", pruned)
        trial.set_user_attr("fold_trajectory", [
            {
                "target_date": str(pd.Timestamp(row.target_date).date()),
                "abs_error_sum": float(row.abs_error_sum),
                "abs_actual_sum": float(row.abs_actual_sum),
            }
            for row in results.itertuples()
        ])

        if pruned:
            raise optuna.TrialPruned()
        return pooled if objective_metric == "pooled" else rolling3

    return objective


class AblationResult:
    """Per-(arm, sb_class) Optuna studies from run_feature_ablation, plus a
    tidy comparison table. Mirrors TuningResult's shape, keyed on
    (arm, sb_class) instead of just sb_class."""

    def __init__(self, model_name, studies, skipped):
        self.model_name = model_name
        self.studies = studies   # {(arm_name, sb_class): optuna.Study}
        self.skipped = skipped   # {(arm_name, sb_class): reason}

    @property
    def results_df(self):
        rows = []
        for (arm_name, sb_class), study in self.studies.items():
            complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            pruned_n = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
            best = study.best_trial
            rows.append({
                "arm": arm_name,
                "sb_class": sb_class,
                "pooled_wmape": best.user_attrs.get("pooled_wmape"),
                "rolling3_wmape": best.user_attrs.get("rolling3_wmape"),
                "n_complete": len(complete),
                "n_pruned": pruned_n,
                "best_params": best.params,
            })
        return pd.DataFrame(rows)

    def deltas_vs_baseline(self, baseline_arm="baseline", metric="pooled_wmape"):
        """Per (arm, sb_class): metric minus the baseline arm's metric for
        that same class. Negative = the feature improved WMAPE. This is the
        table an arm-by-class heatmap would plot directly."""
        df = self.results_df
        base = df.loc[df["arm"] == baseline_arm].set_index("sb_class")[metric]
        out = df.loc[df["arm"] != baseline_arm].copy()
        out["baseline_" + metric] = out["sb_class"].map(base)
        out["delta"] = out[metric] - out["baseline_" + metric]
        return out.sort_values(["sb_class", "delta"]).reset_index(drop=True)

    def trajectory(self, arm_name, sb_class, trial="best"):
        """Fold-level (target_date, abs_error_sum, abs_actual_sum, wmape) for
        one (arm, sb_class)'s trial, reconstructed from the fold_trajectory
        stashed by make_pruning_objective -- NOT a re-run. wmape here is
        exact per-fold (abs_error_sum / abs_actual_sum), and a caller can
        roll it up further (e.g. via rolling_average_wmape on a frame shaped
        like this) for a smoothed trend line.

        trial="best" (default) uses the study's best trial; pass an
        optuna.trial.FrozenTrial to inspect any other one (e.g. to compare
        the winning config against a specific runner-up).
        """
        study = self.studies[(arm_name, sb_class)]
        t = study.best_trial if trial == "best" else trial
        traj = t.user_attrs.get("fold_trajectory", [])
        df = pd.DataFrame(traj)
        if df.empty:
            return df
        df["target_date"] = pd.to_datetime(df["target_date"])
        df["wmape"] = df["abs_error_sum"] / df["abs_actual_sum"]
        return df.sort_values("target_date").reset_index(drop=True)

    def all_trajectories(self, baseline_arm="baseline"):
        """Long-format (arm, sb_class, target_date, wmape, is_baseline) across
        every (arm, sb_class)'s BEST trial -- exactly the shape a WMAPE-over-
        time comparison chart needs, for every arm at once, no re-run."""
        frames = []
        for (arm_name, sb_class) in self.studies:
            df = self.trajectory(arm_name, sb_class)
            if df.empty:
                continue
            df = df.assign(arm=arm_name, sb_class=sb_class, is_baseline=(arm_name == baseline_arm))
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def export(self, path):
        """Dump every trial (including fold_trajectory) across all (arm,
        sb_class) studies to parquet -- a durability backup independent of
        the Optuna storage backend, same rationale as TuningResult.export
        (no backups on the free Supabase tier)."""
        frames = []
        for (arm_name, sb_class), study in self.studies.items():
            df = study.trials_dataframe()
            df.insert(0, "sb_class", sb_class)
            df.insert(0, "arm", arm_name)
            df.insert(0, "model", self.model_name)
            frames.append(df)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        out.to_parquet(path)
        return path


def _banked_trial_count(study):
    """COMPLETE + PRUNED trials already on this study. RUNNING trials (e.g.
    left behind by a worker that was killed mid-trial) don't count, so a
    re-attempt naturally re-runs that slot instead of treating it as done."""
    return sum(
        1 for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    )


def run_feature_ablation(forecast_fn, hp_search_space, arms, *, model_name,
                         storage=None, n_trials_per_arm=15,
                         sb_classes=SB_CLASSES,
                         min_train_months=12, objective_metric="pooled",
                         direction="minimize", sampler=None, pruner=None,
                         prune_warmup_folds=15, cold_start_strategy="zero",
                         load_if_exists=True,
                         dataset_tag="v1", seed=42, verbose=True):
    """Controlled leave-one-in feature ablation across SB classes.

    Every arm in `arms` gets the SAME hp_search_space and the SAME
    n_trials_per_arm budget, so an arm's best WMAPE is directly comparable to
    another arm's -- unlike a single joint search over features +
    hyperparameters together, where some feature combinations get sampled far
    more than others purely by luck.

    Resume-aware: for every (arm, sb_class) study, tops up to
    n_trials_per_arm banked (COMPLETE/PRUNED) trials (see
    _banked_trial_count) rather than unconditionally running
    n_trials_per_arm MORE trials on every call. This is what makes it safe
    to shard -- multiple processes (or repeated calls against a study that's
    already partway done, e.g. left behind by an earlier run) can all call
    this against the same shared storage without duplicating work or
    overshooting the trial budget.

    Routing is point-in-time: every arm is scored via
    scope_fn=scope_to(sb_class), passed straight to run_backtest, which
    reclassifies from each fold's own train_df -- same as
    tune_over_sb_classes. scope_fn also narrows the fold's known-at-origin/
    test universe to that same class, so forecast_fn is only ever asked to
    predict its own class's SKUs and run_backtest's "every known SKU must
    get a prediction" check stays satisfied.

    Total trials = len(arms) * n_trials_per_arm * len(sb_classes). Pruning
    (MedianPruner by default) kills trials clearly losing to the rest of
    their own study after prune_warmup_folds folds, so you can raise
    n_trials_per_arm without paying the full linear cost of running every
    trial to completion.

    cold_start_strategy="zero": run_backtest's own default is "error" --
    raise on any SKU that launches partway through the backtest with no
    prediction yet. "zero" (this function's default) scores those as an
    explicit zero forecast instead, so a real run doesn't abort on the first
    late-launching SKU.

    storage : same contract as tune_over_sb_classes -- a URL string is
    wrapped in one pool-capped RDBStorage via _build_storage (keeps well
    under Supabase free-tier's 15-connection cap), an existing storage
    object is passed through untouched. If None, falls back to a local
    sqlite file so the code runs offline.

    arms : dict[str, dict]
        {arm_name: fixed_params}, e.g. from leave_one_in_arms(). Include a
        "baseline" arm (no feature params) as the reference every other arm
        is compared against via AblationResult.deltas_vs_baseline().

    Returns
    -------
    AblationResult
    """
    if objective_metric not in ("pooled", "rolling3"):
        raise ValueError("objective_metric must be 'pooled' or 'rolling3'")

    try:
        validate_forecast_fn(forecast_fn)
    except TypeError:
        if verbose:
            print("[ablation] contract pre-check skipped (model needs params); "
                  "first trial will surface any issue.")

    if storage is None and verbose:
        print("[ablation] no storage given -> local fallback: sqlite:///optuna_tuning.db")

    pruner = pruner or optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=prune_warmup_folds, interval_steps=1,
    )

    # ONE shared, pool-capped storage reused by every study (see _build_storage).
    storage_obj, owns_storage = _build_storage(storage)
    full_panel, windows, sb_lookup = load_data_cached()

    studies, skipped = {}, {}
    total_arms_classes = len(arms) * len(sb_classes)
    print(f"[ablation] {len(arms)} arms x {len(sb_classes)} classes x "
          f"{n_trials_per_arm} trials/arm = up to {total_arms_classes * n_trials_per_arm} trials total "
          f"(pruning + resume-skip cut the effective cost below this)")

    try:
        for sb_class in sb_classes:
            for arm_name, fixed_params in arms.items():
                study_name = f"{model_name}.{arm_name}__{sb_class}__mtm{min_train_months}__{dataset_tag}"
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_obj, direction=direction, load_if_exists=load_if_exists,
                    sampler=sampler or optuna.samplers.TPESampler(seed=seed),
                    pruner=pruner,
                )
                study.set_user_attr("model_name", model_name)
                study.set_user_attr("arm", arm_name)
                study.set_user_attr("sb_class", sb_class)
                study.set_user_attr("min_train_months", min_train_months)
                study.set_user_attr("objective_metric", objective_metric)

                n_banked = _banked_trial_count(study)
                n_remaining = max(0, n_trials_per_arm - n_banked)

                if n_remaining:
                    objective = make_pruning_objective(
                        forecast_fn, hp_search_space, full_panel, windows,
                        min_train_months, objective_metric,
                        fixed_params=fixed_params, prune_warmup_folds=prune_warmup_folds,
                        cold_start_strategy=cold_start_strategy,
                        scope_fn=scope_to(sb_class),
                    )
                    study.optimize(objective, n_trials=n_remaining, catch=(Exception,))
                elif verbose:
                    print(f"  {arm_name:<22} / {sb_class:>13}: already at budget "
                          f"({n_banked}/{n_trials_per_arm}), skipping")

                complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                pruned_n = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
                if complete:
                    studies[(arm_name, sb_class)] = study
                    if verbose:
                        print(f"  {arm_name:<22} / {sb_class:>13}: "
                              f"{len(complete):>3} complete, {pruned_n:>3} pruned | "
                              f"best {objective_metric} WMAPE={study.best_value:.4f}")
                else:
                    skipped[(arm_name, sb_class)] = "no successful trials"
                    if verbose:
                        print(f"  {arm_name:<22} / {sb_class:>13}: SKIPPED -- no successful trials")
    finally:
        if owns_storage:
            # release pooled connections so we don't hold session-mode clients
            try:
                storage_obj.engine.dispose()
            except Exception:
                pass

    return AblationResult(model_name, studies, skipped)
