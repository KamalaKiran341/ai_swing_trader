"""Portfolio rebalance planning with monthly cadence and eligibility controls."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.portfolio.allocator import PortfolioAllocator


@dataclass(frozen=True)
class RebalancePlan:
    """Output plan for a rebalance cycle."""

    rebalance_required: bool
    rebalance_date: str
    keep_symbols: list[str]
    exit_symbols: list[str]
    entry_symbols: list[str]
    entry_allocations: dict[str, float]
    deployable_capital: float
    residual_cash: float


class PortfolioRebalanceEngine:
    """Generate deterministic monthly rebalance plans.

    Responsibilities:
    - Trigger monthly rebalance once per month.
    - Validate existing positions against current eligibility.
    - Exit positions that lose eligibility.
    - Fill only available slots up to allocator top-N.
    - Respect gradual deployment for new entries.
    """

    def __init__(self, allocator: PortfolioAllocator) -> None:
        self.allocator = allocator

    def is_rebalance_day(
        self,
        current_date: pd.Timestamp,
        last_rebalance_date: pd.Timestamp | None,
    ) -> bool:
        """Return True if current date is first rebalance opportunity in month."""
        current = pd.Timestamp(current_date).normalize()
        if last_rebalance_date is None:
            return True
        last = pd.Timestamp(last_rebalance_date).normalize()
        return (current.year, current.month) != (last.year, last.month)

    def build_rebalance_plan(
        self,
        current_date: pd.Timestamp,
        candidates: pd.DataFrame,
        existing_positions: pd.DataFrame,
        total_capital: float,
        currently_deployed_capital: float,
        last_rebalance_date: pd.Timestamp | None = None,
        deploy_ratio: float | None = None,
    ) -> RebalancePlan:
        """Build a complete rebalance plan for the given date.

        Args:
            current_date: Date for planning.
            candidates: Universe with allocator-required filter columns.
            existing_positions: Current holdings containing at least ``symbol``.
            total_capital: Total strategy capital.
            currently_deployed_capital: Capital already deployed before rebalance.
            last_rebalance_date: Last rebalance date, if any.
            deploy_ratio: Optional gradual deployment override.
        """
        current = pd.Timestamp(current_date).normalize()
        if not self.is_rebalance_day(current, last_rebalance_date):
            return RebalancePlan(
                rebalance_required=False,
                rebalance_date=current.date().isoformat(),
                keep_symbols=[],
                exit_symbols=[],
                entry_symbols=[],
                entry_allocations={},
                deployable_capital=0.0,
                residual_cash=0.0,
            )

        normalized_existing = self._normalize_existing_positions(existing_positions)
        filtered_candidates = self.allocator.filter_candidates(candidates)
        ranked_candidates = self.allocator.select_top_n(
            filtered_candidates,
            top_n=self.allocator.config.top_n,
        )

        existing_symbols = normalized_existing["symbol"].tolist()
        eligible_symbols = set(filtered_candidates["symbol"].astype(str).tolist())
        keep_symbols = [s for s in existing_symbols if s in eligible_symbols]
        exit_symbols = [s for s in existing_symbols if s not in eligible_symbols]

        available_slots = max(self.allocator.config.top_n - len(keep_symbols), 0)
        ranked_symbols = ranked_candidates["symbol"].astype(str).tolist()
        entry_candidates = [s for s in ranked_symbols if s not in keep_symbols][:available_slots]

        deployable = self.allocator.compute_deployable_capital(
            total_capital=total_capital,
            currently_deployed_capital=currently_deployed_capital,
            deploy_ratio=deploy_ratio,
        )

        if deployable <= 0 or not entry_candidates:
            return RebalancePlan(
                rebalance_required=True,
                rebalance_date=current.date().isoformat(),
                keep_symbols=keep_symbols,
                exit_symbols=exit_symbols,
                entry_symbols=[],
                entry_allocations={},
                deployable_capital=float(max(deployable, 0.0)),
                residual_cash=float(max(deployable, 0.0)),
            )

        per_position = deployable / len(entry_candidates)
        allocations = {symbol: float(per_position) for symbol in entry_candidates}
        deployed = float(sum(allocations.values()))
        residual_cash = float(max(deployable - deployed, 0.0))

        return RebalancePlan(
            rebalance_required=True,
            rebalance_date=current.date().isoformat(),
            keep_symbols=keep_symbols,
            exit_symbols=exit_symbols,
            entry_symbols=entry_candidates,
            entry_allocations=allocations,
            deployable_capital=float(deployable),
            residual_cash=residual_cash,
        )

    def _normalize_existing_positions(self, existing_positions: pd.DataFrame) -> pd.DataFrame:
        if "symbol" not in existing_positions.columns:
            raise ValueError("existing_positions must contain 'symbol' column.")
        out = existing_positions.copy()
        out["symbol"] = out["symbol"].astype(str).str.strip()
        out = out[out["symbol"] != ""]
        # Keep first occurrence for deterministic symbol-level planning.
        out = out.drop_duplicates(subset=["symbol"], keep="first")
        return out
