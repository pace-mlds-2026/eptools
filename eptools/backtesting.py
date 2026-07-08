import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Optional

from eptools.modelling import expanding_window_backtest_folds, LAG_MONTHS, HORIZON


def wmape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted mean absolute percentage error: sum|error| / sum|actual|."""
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)))


ForecastFn = Callable[..., pd.DataFrame]


def _score_fold(
    fold: Dict[str, Any],
    preds: pd.DataFrame,
    clip_negative: bool = True,
) -> Dict[str, Any]:
    """Turn one fold's forecast into a row of atomic sums.

    Joins the model's predictions at the fold's target_date onto the held-out
    actuals, applies the shared post-processing (clip negatives to zero), and
    reduces to the sums that pooled WMAPE is built from. SKUs the model failed
    to predict get y_pred = 0, matching the ARIMA notebook's original behaviour.
    """
    target_date = fold["target_date"]

    # keep only the scored month from whatever curve the model returned
    target_preds = (
        preds.loc[preds["ds"] == target_date, ["unique_id", "y_pred"]]
        .copy()
    )

    if clip_negative:
        # LSTM/ARIMA can emit negatives; Croston cannot. Clip identically for all
        # so the post-processing rule is a harness property, not a per-model quirk.
        target_preds["y_pred"] = target_preds["y_pred"].clip(lower=0)

    merged = (
        fold["test_df"][["unique_id", "y"]]
        .merge(target_preds, on="unique_id", how="left")
        .fillna({"y_pred": 0.0})
    )

    err = merged["y"] - merged["y_pred"]
    abs_actual_sum = float(merged["y"].abs().sum())

    return {
        "origin": fold["origin"],
        "target_date": target_date,
        "n_skus": int(len(merged)),
        "abs_error_sum": float(err.abs().sum()),   # WMAPE numerator
        "abs_actual_sum": abs_actual_sum,          # WMAPE denominator
        "error_sum": float(err.sum()),             # signed -> bias (over/under-forecast)
        # per-fold ratio kept only for the 'mean' rolling variant and for eyeballing;
        # never aggregate these directly across folds — aggregate the sums instead.
        "wmape": float(np.abs(err).sum() / abs_actual_sum) if abs_actual_sum > 0 else np.nan,
    }


def run_backtest(
    forecast_fn: ForecastFn,
    params: Optional[Dict[str, Any]] = None,
    min_train_months: int = 12,
    trial: Any = None,
    clip_negative: bool = True,
    report_intermediate: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the fixed rolling-origin backtest for one model configuration.

    Parameters
    ----------
    forecast_fn : callable
        Satisfies the forecast_fn contract: (train_df, horizon_dates, params, trial)
        -> DataFrame[unique_id, ds, y_pred].
    params : dict, optional
        Hyperparameters forwarded verbatim to forecast_fn.
    min_train_months : int
        Warm-up history before the first origin. Keep identical across models.
    trial : optuna.Trial, optional
        Passed through to forecast_fn. Also used here for pruning if
        report_intermediate=True.
    clip_negative : bool
        Clip forecasts at zero before scoring. Applied to every model identically.
    report_intermediate : bool
        If True and a trial is supplied, report the running pooled WMAPE after each
        fold via trial.report(step=fold_index) and honour trial.should_prune().
        Only enable when earlier-origin performance is informative for pruning.

    Returns
    -------
    pd.DataFrame
        One row per fold: origin, target_date, n_skus, abs_error_sum,
        abs_actual_sum, error_sum, wmape. Sorted by origin.
    """
    import optuna  # local import: harness stays importable without optuna installed

    params = params or {}
    rows = []
    run_abs_err = 0.0
    run_abs_act = 0.0

    fold_iter = enumerate(expanding_window_backtest_folds(min_train_months=min_train_months))
    for fold_idx, fold in fold_iter:
        preds = forecast_fn(
            fold["train_df"],
            fold["horizon_dates"],
            params,
            trial,
        )
        row = _score_fold(fold, preds, clip_negative=clip_negative)
        rows.append(row)

        run_abs_err += row["abs_error_sum"]
        run_abs_act += row["abs_actual_sum"]

        if verbose:
            print(
                f"fold {fold_idx + 1:>2}  origin={row['origin']:%Y-%m}  "
                f"target={row['target_date']:%Y-%m}  wmape={row['wmape']:.4f}"
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
