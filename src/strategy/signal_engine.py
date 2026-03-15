"""Hybrid signal scoring engine for technical and fundamental inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalEngineConfig:
    """Configuration for hybrid signal scoring."""

    technical_weight: float = 0.60
    fundamental_weight: float = 0.40
    minimum_score_threshold: float = 70.0

    def __post_init__(self) -> None:
        total_weight = self.technical_weight + self.fundamental_weight
        if not np.isclose(total_weight, 1.0):
            raise ValueError("technical_weight + fundamental_weight must equal 1.0.")
        if self.minimum_score_threshold < 0 or self.minimum_score_threshold > 100:
            raise ValueError("minimum_score_threshold must be between 0 and 100.")


class SignalScoringEngine:
    """Score trade candidates using a weighted technical/fundamental model.

    Notes:
    - This engine is vector-safe: all computations are pandas vector operations.
    - Inputs are expected on a 0-100 scale; values are clipped for numerical safety.
    """

    def __init__(self, config: SignalEngineConfig | None = None) -> None:
        self.config = config or SignalEngineConfig()

    def compute_hybrid_scores(
        self,
        technical_scores: pd.Series,
        fundamental_scores: pd.Series,
    ) -> pd.Series:
        """Compute weighted hybrid scores from technical and fundamental series.

        Args:
            technical_scores: Technical score series indexed by symbol/id.
            fundamental_scores: Fundamental score series indexed by symbol/id.

        Returns:
            Weighted hybrid score series clipped to [0, 100].
        """
        tech = pd.to_numeric(technical_scores, errors="coerce")
        fund = pd.to_numeric(fundamental_scores, errors="coerce")

        hybrid = self.config.technical_weight * tech + self.config.fundamental_weight * fund
        return hybrid.clip(lower=0.0, upper=100.0)

    def score_candidates(
        self,
        candidates: pd.DataFrame,
        technical_col: str = "technical_score",
        fundamental_col: str = "fundamental_score",
        sort_descending: bool = True,
    ) -> pd.DataFrame:
        """Score candidate universe and return DataFrame with pass/fail flags.

        Args:
            candidates: Input frame containing at least technical/fundamental columns.
            technical_col: Column name containing technical scores.
            fundamental_col: Column name containing fundamental scores.
            sort_descending: Whether to sort by score descending.

        Returns:
            DataFrame copy with:
            - ``hybrid_score`` (0-100)
            - ``is_selected`` (bool threshold pass)
        """
        if technical_col not in candidates.columns:
            raise ValueError(f"Missing required column: {technical_col}")
        if fundamental_col not in candidates.columns:
            raise ValueError(f"Missing required column: {fundamental_col}")

        scored = candidates.copy()
        scored["hybrid_score"] = self.compute_hybrid_scores(
            technical_scores=scored[technical_col],
            fundamental_scores=scored[fundamental_col],
        )
        scored["is_selected"] = scored["hybrid_score"] >= self.config.minimum_score_threshold

        if sort_descending:
            scored = scored.sort_values(by="hybrid_score", ascending=False, kind="mergesort")
        return scored
