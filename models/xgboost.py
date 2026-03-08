from __future__ import annotations

from xgboost import XGBRegressor


def build_model(**kwargs) -> XGBRegressor:
    """
    Create an XGBRegressor with basic defaults.
    Extra keyword arguments can override the defaults.
    """
    params: dict = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist",
    }
    params.update(kwargs)
    return XGBRegressor(**params)

