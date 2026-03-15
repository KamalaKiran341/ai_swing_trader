"""Zerodha broker adapter with paper-mode safety and testable injection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)

VALID_SIDES = {"BUY", "SELL"}


@dataclass(frozen=True)
class ZerodhaConfig:
    """Configuration required for Zerodha API sessions."""

    api_key: str
    api_secret: str
    access_token: str
    paper_mode: bool = True


class ZerodhaAdapter:
    """Paper-first Zerodha-compatible broker adapter."""

    def __init__(self, config: ZerodhaConfig, kite_client: Any | None = None) -> None:
        self.config = config
        self._kite = kite_client

        if self._kite is None:
            self._kite = self._create_kite_client()
        self._authenticate()

    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Fetch latest prices for symbols with simple retry handling."""
        cleaned = [s.strip() for s in symbols if isinstance(s, str) and s.strip()]
        if not cleaned:
            return {}

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._kite.ltp(cleaned)
                output: dict[str, float] = {}
                for symbol in cleaned:
                    raw = response.get(symbol, {}) if isinstance(response, dict) else {}
                    price = raw.get("last_price")
                    if price is None:
                        continue
                    output[symbol] = float(price)
                return output
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "event=broker_ltp_retry attempt=%s error=%s",
                    attempt + 1,
                    str(exc),
                )
                if attempt < 2:
                    time.sleep(0.1 * (2**attempt))

        LOGGER.error("event=broker_ltp_failure error=%s", str(last_error))
        return {}

    def get_holdings(self) -> list[dict[str, Any]]:
        """Return normalized holdings structure from broker response."""
        try:
            raw = self._kite.holdings()
        except Exception as exc:
            LOGGER.error("event=broker_holdings_failure error=%s", str(exc))
            return []

        normalized: list[dict[str, Any]] = []
        for item in raw or []:
            normalized.append(
                {
                    "symbol": str(item.get("tradingsymbol", "")),
                    "quantity": int(item.get("quantity", 0)),
                    "average_price": float(item.get("average_price", 0.0)),
                    "last_price": float(item.get("last_price", 0.0)),
                    "pnl": float(item.get("pnl", 0.0)),
                }
            )
        return normalized

    def get_positions(self) -> list[dict[str, Any]]:
        """Return normalized net positions."""
        try:
            raw = self._kite.positions()
        except Exception as exc:
            LOGGER.error("event=broker_positions_failure error=%s", str(exc))
            return []

        net_positions = raw.get("net", []) if isinstance(raw, dict) else []
        normalized: list[dict[str, Any]] = []
        for item in net_positions:
            normalized.append(
                {
                    "symbol": str(item.get("tradingsymbol", "")),
                    "quantity": int(item.get("quantity", 0)),
                    "average_price": float(item.get("average_price", 0.0)),
                    "last_price": float(item.get("last_price", 0.0)),
                    "pnl": float(item.get("pnl", 0.0)),
                }
            )
        return normalized

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        """Place order in paper mode (simulated) or live mode (Kite API)."""
        clean_symbol = symbol.strip() if isinstance(symbol, str) else ""
        normalized_side = side.upper() if isinstance(side, str) else ""
        if not clean_symbol:
            raise ValueError("symbol must be a non-empty string.")
        if normalized_side not in VALID_SIDES:
            raise ValueError("side must be one of {'BUY', 'SELL'}.")
        if quantity <= 0:
            raise ValueError("quantity must be greater than 0.")

        LOGGER.info(
            "event=order_attempt mode=%s symbol=%s side=%s qty=%s type=%s",
            "PAPER" if self.config.paper_mode else "LIVE",
            clean_symbol,
            normalized_side,
            quantity,
            order_type,
        )

        if self.config.paper_mode:
            LOGGER.info(
                "event=paper_order symbol=%s side=%s qty=%s type=%s",
                clean_symbol,
                normalized_side,
                quantity,
                order_type,
            )
            return {
                "status": "success",
                "mode": "paper",
                "order_id": f"PAPER-{clean_symbol}-{normalized_side}-{quantity}",
                "symbol": clean_symbol,
                "side": normalized_side,
                "quantity": int(quantity),
                "order_type": str(order_type),
            }

        try:
            order_id = self._kite.place_order(
                tradingsymbol=clean_symbol,
                exchange="NSE",
                transaction_type=normalized_side,
                quantity=int(quantity),
                order_type=order_type,
                product="CNC",
                variety="regular",
            )
            return {
                "status": "success",
                "mode": "live",
                "order_id": str(order_id),
                "symbol": clean_symbol,
                "side": normalized_side,
                "quantity": int(quantity),
                "order_type": str(order_type),
            }
        except Exception as exc:
            LOGGER.error("event=live_order_failure symbol=%s error=%s", clean_symbol, str(exc))
            raise

    def _authenticate(self) -> None:
        try:
            self._kite.set_access_token(self.config.access_token)
            LOGGER.info("event=broker_auth_success")
        except Exception as exc:
            LOGGER.error("event=broker_auth_failure error=%s", str(exc))
            raise

    def _create_kite_client(self) -> Any:
        try:
            from kiteconnect import KiteConnect  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "kiteconnect package is required when kite_client is not injected."
            ) from exc
        return KiteConnect(api_key=self.config.api_key)
