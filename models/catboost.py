from __future__ import annotations

from catboost import CatBoostRegressor


def build_model(**kwargs) -> CatBoostRegressor:
    """
    Create a CatBoostRegressor with basic defaults.
    Extra keyword arguments can override the defaults.
    """
    params: dict = {
        "depth": 6,
        "learning_rate": 0.1,
        "loss_function": "RMSE",
        "n_estimators": 500,
        "random_seed": 42,
        "verbose": False,
    }
    params.update(kwargs)
    return CatBoostRegressor(**params)

