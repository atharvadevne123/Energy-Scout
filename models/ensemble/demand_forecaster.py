"""
Energy demand forecasting ensemble.

XGBoost + LightGBM soft-voting ensemble with:
- Multi-step time series forecasting via lag features
- Confidence intervals through quantile regression
- SHAP-based explanations
- MLflow experiment tracking
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


class DemandForecaster:
    """
    Two-model soft-voting ensemble (XGBoost + LightGBM) for multi-step
    energy demand forecasting with quantile-based confidence intervals.
    """

    def __init__(
        self,
        xgb_params: dict | None = None,
        lgb_params: dict | None = None,
        quantile_alpha: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.random_state = random_state
        self.quantile_alpha = quantile_alpha
        self.feature_names: list[str] = []
        self._fitted = False

        # Central-tendency models
        xgb_defaults = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="mae",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        lgb_defaults = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

        # Quantile models for lower / upper confidence bounds
        xgb_lower_params = {
            **xgb_defaults,
            "objective": "reg:quantileerror",
            "quantile_alpha": quantile_alpha,
        }
        xgb_upper_params = {
            **xgb_defaults,
            "objective": "reg:quantileerror",
            "quantile_alpha": 1 - quantile_alpha,
        }
        lgb_lower_params = {**lgb_defaults, "objective": "quantile", "alpha": quantile_alpha}
        lgb_upper_params = {**lgb_defaults, "objective": "quantile", "alpha": 1 - quantile_alpha}

        self.xgb_model = xgb.XGBRegressor(**(xgb_params or xgb_defaults))
        self.lgb_model = lgb.LGBMRegressor(**(lgb_params or lgb_defaults))

        self.xgb_lower = xgb.XGBRegressor(**xgb_lower_params)
        self.xgb_upper = xgb.XGBRegressor(**xgb_upper_params)
        self.lgb_lower = lgb.LGBMRegressor(**lgb_lower_params)
        self.lgb_upper = lgb.LGBMRegressor(**lgb_upper_params)

        self.scaler = StandardScaler()
        self._shap_explainer: shap.TreeExplainer | None = None

        # Ensemble weights: XGB 60%, LGB 40%
        self._weights = np.array([0.6, 0.4])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_X: pd.DataFrame | None = None,
        eval_y: pd.Series | None = None,
        mlflow_run: bool = True,
    ) -> DemandForecaster:
        """Fit the ensemble on training data.

        Args:
            X: Feature DataFrame (lag features, temporal, weather, etc.)
            y: Target — energy consumption in kWh
            eval_X: Optional validation features for metric logging
            eval_y: Optional validation targets
            mlflow_run: Whether to log the run to MLflow
        """
        self.feature_names = list(X.columns)
        X_arr = self.scaler.fit_transform(X.values)
        y_arr = y.values

        logger.info("Training DemandForecaster on {:,} samples.", len(y_arr))

        run_ctx = (
            mlflow.start_run(run_name="energy_demand_forecast") if mlflow_run else _NullContext()
        )
        with run_ctx:
            # Fit central models
            self.xgb_model.fit(X_arr, y_arr)
            self.lgb_model.fit(X_arr, y_arr)

            # Fit quantile bounds
            self.xgb_lower.fit(X_arr, y_arr)
            self.xgb_upper.fit(X_arr, y_arr)
            self.lgb_lower.fit(X_arr, y_arr)
            self.lgb_upper.fit(X_arr, y_arr)

            if mlflow_run:
                train_pred = self._predict_raw(X_arr)
                mlflow.log_metric("train_mae", mean_absolute_error(y_arr, train_pred))
                mlflow.log_metric(
                    "train_rmse", float(np.sqrt(mean_squared_error(y_arr, train_pred)))
                )
                mlflow.log_params(
                    {
                        "n_estimators_xgb": self.xgb_model.n_estimators,
                        "n_estimators_lgb": self.lgb_model.n_estimators,
                        "quantile_alpha": self.quantile_alpha,
                    }
                )

                if eval_X is not None and eval_y is not None:
                    eval_arr = self.scaler.transform(eval_X.values)
                    val_pred = self._predict_raw(eval_arr)
                    mlflow.log_metric("val_mae", mean_absolute_error(eval_y.values, val_pred))
                    mlflow.log_metric(
                        "val_rmse", float(np.sqrt(mean_squared_error(eval_y.values, val_pred)))
                    )
                    logger.info("Val MAE: {:.4f}", mean_absolute_error(eval_y.values, val_pred))

        # Build SHAP explainer on the XGBoost central model
        try:
            self._shap_explainer = shap.TreeExplainer(self.xgb_model)
            logger.success("SHAP TreeExplainer initialised.")
        except Exception as exc:
            logger.warning("Could not build SHAP explainer: {}", exc)

        self._fitted = True
        logger.success("DemandForecaster fitted on {:,} samples.", len(y_arr))
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forecast(
        self,
        X: pd.DataFrame,
        horizon: int = 24,
    ) -> dict:
        """Produce multi-step forecasts.

        For single-row input the same feature vector is reused across the
        horizon (steady-state scenario). For multi-row input each row
        corresponds to one step.

        Returns:
            dict with keys:
              - forecast_kwh: list[float]       — point forecast per step
              - lower_kwh:    list[float]       — lower confidence bound
              - upper_kwh:    list[float]       — upper confidence bound
              - peak_demand_kwh: float
        """
        self._check_fitted()

        if len(X) == 1 and horizon > 1:
            X = pd.concat([X] * horizon, ignore_index=True)
        elif len(X) < horizon:
            # Pad by repeating the last row
            padding = pd.concat([X.iloc[[-1]]] * (horizon - len(X)), ignore_index=True)
            X = pd.concat([X, padding], ignore_index=True)
        else:
            X = X.iloc[:horizon]

        X_arr = self.scaler.transform(X.values)

        point = self._predict_raw(X_arr)
        lower = self._predict_lower(X_arr)
        upper = self._predict_upper(X_arr)

        # Enforce physical constraints: no negative energy
        point = np.maximum(point, 0.0)
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)

        return {
            "forecast_kwh": [round(float(v), 3) for v in point],
            "lower_kwh": [round(float(v), 3) for v in lower],
            "upper_kwh": [round(float(v), 3) for v in upper],
            "peak_demand_kwh": round(float(np.max(point)), 3),
        }

    def explain(self, X: pd.DataFrame, max_display: int = 10) -> list[dict]:
        """Return SHAP-based feature attributions for each row in X."""
        if self._shap_explainer is None:
            return [{"error": "SHAP explainer not available."}]

        X_arr = self.scaler.transform(X.values)
        shap_values = self._shap_explainer.shap_values(X_arr)
        results = []
        for i in range(len(X)):
            top_idx = np.argsort(np.abs(shap_values[i]))[::-1][:max_display]
            results.append(
                {
                    "shap_values": {
                        self.feature_names[j]: round(float(shap_values[i][j]), 6) for j in top_idx
                    },
                    "base_value": round(float(self._shap_explainer.expected_value), 6),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else ARTIFACT_DIR / "demand_forecaster.joblib"
        joblib.dump(self, path)
        logger.info("DemandForecaster saved → {}", path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> DemandForecaster:
        path = Path(path) if path else ARTIFACT_DIR / "demand_forecaster.joblib"
        obj = joblib.load(path)
        logger.info("DemandForecaster loaded ← {}", path)
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_raw(self, X_arr: np.ndarray) -> np.ndarray:
        xgb_pred = self.xgb_model.predict(X_arr)
        lgb_pred = self.lgb_model.predict(X_arr)
        return self._weights[0] * xgb_pred + self._weights[1] * lgb_pred

    def _predict_lower(self, X_arr: np.ndarray) -> np.ndarray:
        xgb_pred = self.xgb_lower.predict(X_arr)
        lgb_pred = self.lgb_lower.predict(X_arr)
        return self._weights[0] * xgb_pred + self._weights[1] * lgb_pred

    def _predict_upper(self, X_arr: np.ndarray) -> np.ndarray:
        xgb_pred = self.xgb_upper.predict(X_arr)
        lgb_pred = self.lgb_upper.predict(X_arr)
        return self._weights[0] * xgb_pred + self._weights[1] * lgb_pred

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call train() before forecast().")


class _NullContext:
    """Context manager that does nothing — substitutes for mlflow.start_run."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
