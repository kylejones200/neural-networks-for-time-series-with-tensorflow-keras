"""Structural Time Series forecasting with statsmodels (replaces tensorflow_probability.sts)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.evaluator import Evaluator
from statsmodels.tsa.statespace.structural import UnobservedComponents

from src import (
    create_forecast_plot,
    ensure_output_dir,
    get_output_dir,
    load_config,
    load_time_series,
    save_plot,
)


def build_structural_model(
    observed_time_series: np.ndarray,
    num_seasons: int = 12,
    include_trend: bool = True,
    include_seasonal: bool = True,
    include_autoregressive: bool = False,
    ar_order: int = 1,
) -> dict:
    """
    Build a structural time series model specification.

    Replaces tfp.sts.LocalLinearTrend / Seasonal / Autoregressive / Sum.
    Returns a kwargs dict for statsmodels UnobservedComponents.
    """
    spec: dict = {}
    if include_trend:
        spec["level"] = "local linear trend"
    elif not include_seasonal and (not include_autoregressive):
        spec["level"] = "local level"
    if include_seasonal:
        spec["seasonal"] = num_seasons
    if include_autoregressive:
        spec["autoregressive"] = ar_order
    if not spec:
        raise ValueError(
            "At least one component (trend, seasonal, or autoregressive) must be included"
        )
    return spec


def fit_model(
    model_spec: dict,
    observed_time_series: "np.ndarray | pd.Series",
    num_variational_steps: int = 200,
    learning_rate: float = 0.1,
    num_samples: int = 50,
) -> tuple:
    """
    Fit the structural time series model using statsmodels MLE.

    Replaces tfp.vi.fit_surrogate_posterior + tfp.sts.build_factored_surrogate_posterior.

    Returns:
        fitted_result  – statsmodels UnobservedComponentsResults
        state_samples  – simulation-smoother draws (shape: num_samples × T × state_dim)
        log_likelihood – scalar np.ndarray (proxy for ELBO loss curve)
    """
    if not isinstance(observed_time_series, pd.Series):
        observed_time_series = pd.Series(observed_time_series)
    uc = UnobservedComponents(observed_time_series, **model_spec)
    result = uc.fit(method="powell", maxiter=num_variational_steps, disp=False)
    sim = result.simulation_smoother()
    state_samples = []
    for _ in range(num_samples):
        sim.simulate()
        state_samples.append(sim.simulated_state.copy())
    state_samples = np.stack(state_samples)
    loss_curve = np.array([-result.llf])
    return (result, state_samples, loss_curve)


def forecast(
    fitted_result,
    observed_time_series: np.ndarray,
    state_samples: np.ndarray,
    forecast_horizon: int,
    num_samples: int = 20,
) -> tuple:
    """
    Generate a probabilistic forecast.

    Replaces tfp.sts.forecast → forecast_dist.mean() / stddev() / sample().

    Returns:
        forecast_mean    – shape (forecast_horizon,)
        forecast_std     – shape (forecast_horizon,)
        forecast_samples – shape (num_samples, forecast_horizon)
    """
    fc = fitted_result.get_forecast(forecast_horizon)
    forecast_mean = fc.predicted_mean.values
    forecast_std = fc.se_mean.values
    sims = fitted_result.simulate(
        anchor="end", nsimulations=forecast_horizon, repetitions=num_samples
    )
    forecast_samples = np.array(sims).T
    return (forecast_mean, forecast_std, forecast_samples)


def load_data() -> None:
    "Main execution function."

    script_dir = Path(__file__).parent

    config = load_config(script_dir / "config.yaml")

    output_dir = ensure_output_dir(get_output_dir(config, script_dir))

    data_config = config["data"]

    data_path = script_dir.parent / data_config["input_file"]

    series = load_time_series(
        str(data_path),
        date_column=data_config.get("date_column", "date"),
        value_column=data_config.get("value_column", "value"),
    )

    evaluator = Evaluator(test_size=config["evaluation"].get("test_size", 0.2))

    train, test = evaluator.split(series)

    train_values = train.values.astype(np.float64)

    model_config = config.get("model", {})

    logger.info("=== Building structural time series model ===")

    model_spec = build_structural_model(
        observed_time_series=train_values,
        num_seasons=model_config.get("num_seasons", 12),
        include_trend=model_config.get("include_trend", True),
        include_seasonal=model_config.get("include_seasonal", True),
        include_autoregressive=model_config.get("include_autoregressive", False),
        ar_order=model_config.get("ar_order", 1),
    )

    logger.info(f"Components: {list(model_spec.keys())}")

    logger.info("=== Fitting model ===")

    fitted_result, state_samples, loss_curve = fit_model(
        model_spec=model_spec,
        observed_time_series=train_values,
        num_variational_steps=model_config.get("num_variational_steps", 200),
        learning_rate=model_config.get("learning_rate", 0.1),
        num_samples=model_config.get("num_samples", 50),
    )

    logger.info(f"Log-likelihood: {-loss_curve[0]:.2f}")

    forecast_horizon = config["evaluation"].get("forecast_horizon", len(test))

    logger.info(f"=== Generating {forecast_horizon}-step forecast ===")

    forecast_mean, forecast_std, forecast_samples = forecast(
        fitted_result=fitted_result,
        observed_time_series=train_values,
        state_samples=state_samples,
        forecast_horizon=forecast_horizon,
        num_samples=model_config.get("forecast_samples", 20),
    )

    freq = pd.infer_freq(train.index) or "D"

    forecast_dates = pd.date_range(
        start=train.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=forecast_horizon,
        freq=freq,
    )

    forecast_series = pd.Series(forecast_mean, index=forecast_dates)

    conf_int = pd.DataFrame(
        {
            "lower": forecast_mean - 1.96 * forecast_std,
            "upper": forecast_mean + 1.96 * forecast_std,
        },
        index=forecast_dates,
    )

    if len(forecast_series) == len(test):
        rmse = np.sqrt(mean_squared_error(test.values, forecast_mean))
        mae = mean_absolute_error(test.values, forecast_mean)
        r2 = r2_score(test.values, forecast_mean)
        logger.info("=== Test Set Performance ===")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  R²:   {r2:.4f}")
        logger.info(f"  Mean σ: {forecast_std.mean():.4f}")

    fig, ax = create_forecast_plot(
        train=train,
        test=test if len(test) <= len(forecast_series) else None,
        forecast=forecast_series,
        conf_int=conf_int,
        title="Structural Time Series Forecast (statsmodels UnobservedComponents)",
        xlabel="Date",
        ylabel="Value",
        train_label="Historical (Train)",
        test_label="Actual (Test)",
        forecast_label="STS Forecast",
        show_ci=True,
    )

    plot_path = output_dir / config["output"].get("plot_file", "sts_forecast.png")

    save_plot(fig, plot_path, dpi=config["output"].get("dpi", 300))

    logger.info(f"Plot saved: {plot_path}")

    forecast_df = pd.DataFrame(
        {
            "date": forecast_series.index,
            "forecast": forecast_mean,
            "std": forecast_std,
            "lower_95": conf_int["lower"].values,
            "upper_95": conf_int["upper"].values,
        }
    )

    csv_path = output_dir / config["output"].get("forecast_file", "sts_forecast.csv")

    forecast_df.to_csv(csv_path, index=False, encoding="utf-8")

    logger.info(f"Forecast saved: {csv_path}")


def main() -> None:
    load_data()


if __name__ == "__main__":
    main()
