from __future__ import annotations

from lightgbm import LGBMRegressor


def build_model(**kwargs) -> LGBMRegressor:
    """
    Create an LGBMRegressor with basic defaults.
    Extra keyword arguments can override the defaults.
    """
    params: dict = {
        "n_estimators": 500,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 1,
    }
    params.update(kwargs)
    return LGBMRegressor(**params)

