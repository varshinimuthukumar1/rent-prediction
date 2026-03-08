from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor


def build_model(**kwargs) -> RandomForestRegressor:
    """
    Create a RandomForestRegressor with sensible defaults for this project.
    Extra keyword arguments can override the defaults.
    """
    params: dict = {
        "n_estimators": 50,
        "max_depth": 20,
        "n_jobs": 1,
        "random_state": 42,
    }
    params.update(kwargs)
    return RandomForestRegressor(**params)

