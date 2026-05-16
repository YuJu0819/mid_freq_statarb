"""
Residual-Based Mixture of Experts ensemble container.

Lives in its own module (not in train_ebm_signal.py) so loky/joblib workers
can resolve the class under a stable import path. When the class was defined
in train_ebm_signal.py, running that script as `python -m src.scripts.
train_ebm_signal` put it under `__main__`, and worker processes — which
re-import the module under its real dotted path — failed to find
`__main__.ResidualMoE` and raised a PicklingError.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ResidualMoE:
    """
    Residual-Based Mixture of Experts ensemble.

    Holds one global EBM ensemble (trained on the full window target) and a
    per-regime expert EBM ensemble (trained on global residuals).

    Prediction
    ----------
    Score_total = Global_Score + (λ × Expert_Score)

    When the active regime has no trained expert the expert contribution
    falls back to zero, so the total score equals the global score.
    """

    def __init__(
        self,
        global_models: list,
        # str(regime) -> list[ExplainableBoostingRegressor]
        expert_dict: dict,
    ):
        self.global_models = global_models
        self.expert_dict = expert_dict

    # ------------------------------------------------------------------
    def predict_global(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.global_models], axis=0)

    def predict_expert(
        self,
        regime: str | None,
        X: np.ndarray,
        X_expert: "np.ndarray | None" = None,
    ) -> np.ndarray:
        if regime is None or regime not in self.expert_dict:
            return np.zeros(len(X))
        experts = self.expert_dict[regime]
        if not experts:
            return np.zeros(len(X))
        # If ADX-neutralized features are provided for experts, use them;
        # otherwise fall back to the same X as the global model.
        X_in = X_expert if X_expert is not None else X
        return np.mean([m.predict(X_in) for m in experts], axis=0)

    def predict_total(
        self,
        X: np.ndarray,
        regime: str | None,
        moe_boost_lambda: float,
        X_expert: "np.ndarray | None" = None,
    ) -> np.ndarray:
        g = self.predict_global(X)
        e = self.predict_expert(regime, X, X_expert=X_expert)
        return g + moe_boost_lambda * e

    # ------------------------------------------------------------------
    def global_importances(self) -> pd.Series:
        imp = [
            pd.Series(m.term_importances(), index=list(m.term_names_))
            for m in self.global_models
        ]
        return pd.concat(imp, axis=1).mean(axis=1)

    def expert_importances(self, regime: str) -> "pd.Series | None":
        if regime not in self.expert_dict or not self.expert_dict[regime]:
            return None
        imp = [
            pd.Series(m.term_importances(), index=list(m.term_names_))
            for m in self.expert_dict[regime]
        ]
        return pd.concat(imp, axis=1).mean(axis=1)
