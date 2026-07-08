import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Optional

from eptools.modelling import expanding_window_backtest_folds, LAG_MONTHS, HORIZON



def wmape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted mean absolute percentage error: sum|error| / sum|actual|."""
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)))


def as_forecast_fn(model_fn: Callable[..., pd.DataFrame], **model_kwargs) -> "ForecastFn":
    """Wrap an Edwin-style ``model_fn`` into the canonical nixtla ``forecast_fn``.

    Edwin's contract (collision_demand_forecasting_edwin_v3):
        model_fn(train_df[sku_id, month, demand], horizon, **kwargs)
            -> DataFrame[sku_id, month, h, yhat, used_fallback, fallback_reason]

    The wrapper renames the incoming nixtla frame to Edwin's schema, calls the model with
    ``horizon = len(horizon_dates)``, maps the returned per-step ``h`` back onto this fold's
    actual ``horizon_dates`` (so a truncated tail near the end of data still lines up), and
    renames the output back to ``unique_id / ds / y_pred``. ``params`` and ``trial`` are
    accepted and ignored — Edwin's models take their settings via ``model_kwargs``.

    Transitional only: see the contract note above.
    """
    def forecast_fn(train_df, horizon_dates, params, trial=None):
        tf = train_df.rename(columns={"unique_id": "sku_id", "ds": "month", "y": "demand"})
        out = model_fn(tf, horizon=len(horizon_dates), **model_kwargs)
        h_to_ds = {h: d for h, d in enumerate(horizon_dates, start=1)}
        out = out[out["h"].isin(h_to_ds)].copy()
        out["ds"] = out["h"].map(h_to_ds)
        return (out.rename(columns={"sku_id": "unique_id", "yhat": "y_pred"})
                   [["unique_id", "ds", "y_pred"]])
    return forecast_fn



ForecastFn = Callable[..., pd.DataFrame]


def score_fold(
    fold: Dict[str, Any],
    preds: pd.DataFrame,
    clip_negative: bool = True,
) -> Dict[str, Any]:
    """Turn one fold's forecast into a row of atomic sums (see the note above).

    SKUs active at the target month but missing from ``preds`` are excluded from the WMAPE
    and reported in ``n_skus_excluded_new`` rather than scored as a zero forecast.
    """
    target_date = fold["target_date"]

    # SKUs active at the target month = the held-out actuals in a within-window-dense panel
    actuals = fold["test_df"][["unique_id", "y"]]
    n_active = int(len(actuals))

    # keep only the scored month from whatever curve the model returned
    target_preds = preds.loc[preds["ds"] == target_date, ["unique_id", "y_pred"]].copy()

    if clip_negative:
        # LSTM/ARIMA can emit negatives; Croston cannot. Clip identically for all so the
        # post-processing rule is a harness property, not a per-model quirk.
        target_preds["y_pred"] = target_preds["y_pred"].clip(lower=0)

    joined = actuals.merge(target_preds, on="unique_id", how="left")

    # SKUs the model could not have known about (launched inside the lag window): exclude
    # from the metric, but count them so a model quietly failing on new SKUs is visible.
    missing = joined["y_pred"].isna()
    n_excluded_new = int(missing.sum())
    joined = joined[~missing]

    err = joined["y"] - joined["y_pred"]
    abs_actual_sum = float(joined["y"].abs().sum())

    return {
        "origin": fold["origin"],
        "target_month": target_date,
        "n_skus_scored": int(len(joined)),
        "n_skus_active_at_target": n_active,
        "n_skus_excluded_new": n_excluded_new,
        "abs_error_sum": float(err.abs().sum()),   # WMAPE numerator
        "abs_actual_sum": abs_actual_sum,          # WMAPE denominator
        "error_sum": float(err.sum()),             # signed -> bias (over/under-forecast)
        # per-fold ratio kept only for the 'mean' rolling variant and for eyeballing;
        # never aggregate these directly across folds — aggregate the sums instead.
        "wmape": float(np.abs(err).sum() / abs_actual_sum) if abs_actual_sum > 0 else np.nan,
    }



def run_rolling_backtest(
    forecast_fn: ForecastFn,
    params: Optional[Dict[str, Any]] = None,
    min_train_months: int = 12,
    window_months: Optional[int] = None,
    trial: Any = None,
    clip_negative: bool = True,
    report_intermediate: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the fixed rolling-origin backtest for one model configuration.

    Parameters
    ----------
    forecast_fn : callable
        Satisfies the model contract: (train_df, horizon_dates, params, trial)
        -> DataFrame[unique_id, ds, y_pred]. Edwin-style models: wrap with as_forecast_fn.
    params : dict, optional
        Hyperparameters forwarded verbatim to forecast_fn.
    min_train_months : int
        Warm-up history before the first origin. Keep identical across models.
    window_months : int, optional
        None (default) = expanding window; an integer = trailing sliding window, forwarded
        to expanding_window_backtest_folds.
    trial : optuna.Trial, optional
        Passed through to forecast_fn. Also used here for pruning if report_intermediate=True.
    clip_negative : bool
        Clip forecasts at zero before scoring. Applied to every model identically.
    report_intermediate : bool
        If True and a trial is supplied, report the running pooled WMAPE after each fold via
        trial.report(step=fold_index) and honour trial.should_prune().

    Returns
    -------
    pd.DataFrame
        One row per fold: origin, target_month, n_skus_scored, n_skus_active_at_target,
        n_skus_excluded_new, abs_error_sum, abs_actual_sum, error_sum, wmape. Sorted by origin.
    """
    import optuna  # local import: harness stays importable without optuna installed

    params = params or {}
    rows = []
    run_abs_err = 0.0
    run_abs_act = 0.0

    fold_iter = enumerate(expanding_window_backtest_folds(
        min_train_months=min_train_months, window_months=window_months))
    for fold_idx, fold in fold_iter:
        preds = forecast_fn(
            fold["train_df"],
            fold["horizon_dates"],
            params,
            trial,
        )
        row = score_fold(fold, preds, clip_negative=clip_negative)
        rows.append(row)

        run_abs_err += row["abs_error_sum"]
        run_abs_act += row["abs_actual_sum"]

        if verbose:
            print(
                f"fold {fold_idx + 1:>2}  origin={row['origin']:%Y-%m}  "
                f"target={row['target_month']:%Y-%m}  wmape={row['wmape']:.4f}  "
                f"excluded_new={row['n_skus_excluded_new']}"
            )

        if report_intermediate and trial is not None:
            running_pooled = run_abs_err / run_abs_act if run_abs_act > 0 else float("nan")
            trial.report(running_pooled, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)



def pooled_wmape(results_df: pd.DataFrame, by: Optional[list] = None):
    """Volume-weighted pooled WMAPE from atomic sums.

    Σ|error| / Σ|actual| over the rows. With `by`, returns one pooled WMAPE per
    group (e.g. by='target_date' for a per-month curve, or by a joined segment
    column). Never averages per-fold ratios — always divides summed sums.
    """
    if by is None:
        denom = results_df["abs_actual_sum"].sum()
        return float(results_df["abs_error_sum"].sum() / denom) if denom > 0 else float("nan")

    g = results_df.groupby(by)
    out = (g["abs_error_sum"].sum() / g["abs_actual_sum"].sum()).rename("pooled_wmape")
    return out.reset_index()


def rolling_average_wmape(results_df, window=3, method="pooled"):
    r = results_df.sort_values("origin").reset_index(drop=True)
    if method == "pooled":
        return r["abs_error_sum"].rolling(window).sum() / r["abs_actual_sum"].rolling(window).sum()
    if method == "mean":
        return r["wmape"].rolling(window).mean()
    raise ValueError("method must be 'pooled' or 'mean'")


def classify_sb(panel: pd.DataFrame) -> pd.DataFrame:
    """Syntetos-Boylan class per series from full history (reporting only).

    ADI = average demand interval (periods per demand occurrence); CV2 = squared coefficient
    of variation of nonzero demand. Smooth: ADI<1.32 & CV2<0.49; Erratic: ADI<1.32 & CV2>=0.49;
    Intermittent: ADI>=1.32 & CV2<0.49; Lumpy: ADI>=1.32 & CV2>=0.49; NoDemand: never sold.
    """
    is_demand = panel["y"] > 0
    stats = panel.assign(is_demand=is_demand).groupby("unique_id").agg(
        n_periods=("ds", "count"),
        n_demand_periods=("is_demand", "sum"),
        mean_nonzero=("y", lambda s: s[s > 0].mean()),
        std_nonzero=("y", lambda s: s[s > 0].std()),
    ).reset_index()

    stats["adi"] = stats["n_periods"] / stats["n_demand_periods"].replace(0, np.nan)
    stats["cv2"] = (stats["std_nonzero"] / stats["mean_nonzero"]) ** 2

    def label(row):
        if row["n_demand_periods"] == 0:
            return "NoDemand"
        smooth_adi = row["adi"] < 1.32
        smooth_cv2 = row["cv2"] < 0.49
        if smooth_adi and smooth_cv2:      return "Smooth"
        if smooth_adi and not smooth_cv2:  return "Erratic"
        if not smooth_adi and smooth_cv2:  return "Intermittent"
        return "Lumpy"

    stats["sb_class"] = stats.apply(label, axis=1)
    return stats[["unique_id", "adi", "cv2", "sb_class"]]



def score_fold_segments(fold, preds, sb_lookup, clip_negative=True):
    """Like score_fold but grouped by Syntetos-Boylan class. Returns per-segment sums so the
    top-level pooled-by-segment number can be computed after concatenating folds. Same
    exclude-missing behaviour: SKUs with no forecast are dropped from the metric."""
    target_date = fold["target_date"]
    actuals = fold["test_df"][["unique_id", "y"]]

    target_preds = preds.loc[preds["ds"] == target_date, ["unique_id", "y_pred"]].copy()
    if clip_negative:
        target_preds["y_pred"] = target_preds["y_pred"].clip(lower=0)

    joined = (actuals
              .merge(target_preds, on="unique_id", how="left")
              .merge(sb_lookup[["unique_id", "sb_class"]], on="unique_id", how="left")
              .dropna(subset=["y_pred"]))
    joined["abs_error"] = (joined["y"] - joined["y_pred"]).abs()

    seg = joined.groupby("sb_class").agg(
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("y", lambda s: s.abs().sum()),
        n_skus=("unique_id", "count"),
    ).reset_index()
    seg["origin"] = fold["origin"]
    return seg


def run_rolling_backtest_segmented(forecast_fn, sb_lookup, params=None, min_train_months=12,
                                   window_months=None, clip_negative=True):
    """Segmented sibling of run_rolling_backtest: one concatenated frame of per-class per-fold
    sums, ready for pooled_wmape_by_segment."""
    params = params or {}
    all_segs = []
    for fold in expanding_window_backtest_folds(min_train_months=min_train_months,
                                                window_months=window_months):
        preds = forecast_fn(fold["train_df"], fold["horizon_dates"], params, None)
        all_segs.append(score_fold_segments(fold, preds, sb_lookup, clip_negative=clip_negative))
    return pd.concat(all_segs, ignore_index=True)


def pooled_wmape_by_segment(segment_results: pd.DataFrame) -> pd.DataFrame:
    """Pooled WMAPE per Syntetos-Boylan class from the stored per-segment sums."""
    g = segment_results.groupby("sb_class")
    return pd.DataFrame({
        "wmape_pooled":        g["abs_error_sum"].sum() / g["abs_actual_sum"].sum(),
        "total_actual_volume": g["abs_actual_sum"].sum(),
        "avg_n_skus_per_fold": g["n_skus"].mean(),
    })

