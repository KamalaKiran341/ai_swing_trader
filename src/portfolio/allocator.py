"""Portfolio allocation engine for filtered top-N equal-weight deployment."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AllocationConfig:
    """Configuration for portfolio allocation behavior."""

    top_n: int = 10
    min_score_threshold: float = 70.0
    default_deploy_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be greater than 0.")
        if not (0.0 <= self.min_score_threshold <= 100.0):
            raise ValueError("min_score_threshold must be within [0, 100].")
        if not (0.0 < self.default_deploy_ratio <= 1.0):
            raise ValueError("default_deploy_ratio must be within (0, 1].")


@dataclass(frozen=True)
class AllocationResult:
    """Output payload for an allocation cycle."""

    selected_symbols: list[str]
    weights: dict[str, float]
    capital_allocations: dict[str, float]
    deployable_capital: float
    residual_cash: float
    filtered_candidates: pd.DataFrame


class PortfolioAllocator:
    """Allocate capital into top-N candidates after production filters.

    Expected input columns:
    - ``symbol`` (str)
    - ``score`` (numeric)
    - ``is_liquid`` (bool-like)
    - ``has_upcoming_earnings`` (bool-like; True means avoid)
    - ``sector_allowed`` (bool-like)
    """

    REQUIRED_COLUMNS = (
        "symbol",
        "score",
        "is_liquid",
        "has_upcoming_earnings",
        "sector_allowed",
    )

    def __init__(self, config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig()

    def allocate(
        self,
        candidates: pd.DataFrame,
        total_capital: float,
        currently_deployed_capital: float = 0.0,
        deploy_ratio: float | None = None,
        top_n: int | None = None,
    ) -> AllocationResult:
        """Allocate capital using equal weights across top-ranked filtered symbols.

        Args:
            candidates: Candidate universe with required columns.
            total_capital: Total strategy capital (cash + deployed).
            currently_deployed_capital: Current active deployed capital.
            deploy_ratio: Optional target deployment ratio for this cycle.
            top_n: Optional override for number of positions.

        Returns:
            ``AllocationResult`` with selected symbols, weights, and allocations.
        """
        if total_capital < 0:
            raise ValueError("total_capital cannot be negative.")
        if currently_deployed_capital < 0:
            raise ValueError("currently_deployed_capital cannot be negative.")

        ratio = self.config.default_deploy_ratio if deploy_ratio is None else float(deploy_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("deploy_ratio must be within (0, 1].")

        target_top_n = self.config.top_n if top_n is None else int(top_n)
        if target_top_n <= 0:
            raise ValueError("top_n must be greater than 0.")

        filtered = self.filter_candidates(candidates)
        selected = self.select_top_n(filtered, top_n=target_top_n)
        deployable_capital = self.compute_deployable_capital(
            total_capital=total_capital,
            currently_deployed_capital=currently_deployed_capital,
            deploy_ratio=ratio,
        )

        if selected.empty or deployable_capital <= 0:
            return AllocationResult(
                selected_symbols=[],
                weights={},
                capital_allocations={},
                deployable_capital=float(max(deployable_capital, 0.0)),
                residual_cash=float(max(deployable_capital, 0.0)),
                filtered_candidates=filtered,
            )

        symbols = selected["symbol"].tolist()
        equal_weight = 1.0 / len(symbols)
        weights = {symbol: equal_weight for symbol in symbols}

        per_position = deployable_capital * equal_weight
        allocations = {symbol: float(per_position) for symbol in symbols}
        deployed = float(sum(allocations.values()))
        residual_cash = float(max(deployable_capital - deployed, 0.0))

        return AllocationResult(
            selected_symbols=symbols,
            weights=weights,
            capital_allocations=allocations,
            deployable_capital=float(deployable_capital),
            residual_cash=residual_cash,
            filtered_candidates=filtered,
        )

    def filter_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Apply score/liquidity/earnings/sector filters in vectorized form."""
        self._validate_candidate_columns(candidates)

        frame = candidates.copy()
        frame["symbol"] = frame["symbol"].astype(str).str.strip()
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        frame["is_liquid"] = frame["is_liquid"].astype(bool)
        frame["has_upcoming_earnings"] = frame["has_upcoming_earnings"].astype(bool)
        frame["sector_allowed"] = frame["sector_allowed"].astype(bool)

        mask = (
            frame["symbol"].ne("")
            & frame["score"].notna()
            & (frame["score"] >= self.config.min_score_threshold)
            & frame["is_liquid"]
            & (~frame["has_upcoming_earnings"])
            & frame["sector_allowed"]
        )
        return frame.loc[mask].copy()

    def select_top_n(self, candidates: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
        """Select top-N rows by score with deterministic tie-break on symbol."""
        if candidates.empty:
            return candidates.copy()
        target_top_n = self.config.top_n if top_n is None else int(top_n)
        if target_top_n <= 0:
            raise ValueError("top_n must be greater than 0.")

        frame = candidates.copy()
        frame["symbol_key"] = frame["symbol"].astype(str)
        ranked = frame.sort_values(
            by=["score", "symbol_key"],
            ascending=[False, True],
            kind="mergesort",
        ).head(target_top_n)
        return ranked.drop(columns=["symbol_key"])

    def compute_deployable_capital(
        self,
        total_capital: float,
        currently_deployed_capital: float = 0.0,
        deploy_ratio: float | None = None,
    ) -> float:
        """Compute incremental capital to deploy this cycle.

        Gradual deployment targets ``total_capital * deploy_ratio`` and only deploys
        additional capital above current deployed amount.
        """
        if total_capital < 0:
            raise ValueError("total_capital cannot be negative.")
        if currently_deployed_capital < 0:
            raise ValueError("currently_deployed_capital cannot be negative.")

        ratio = self.config.default_deploy_ratio if deploy_ratio is None else float(deploy_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("deploy_ratio must be within (0, 1].")

        target_deployed = float(total_capital) * ratio
        incremental = target_deployed - float(currently_deployed_capital)
        return float(max(incremental, 0.0))

    def _validate_candidate_columns(self, candidates: pd.DataFrame) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in candidates.columns]
        if missing:
            raise ValueError(f"Missing required candidate columns: {missing}")
