# ============================================================
# src/risk_manager.py
#
# PURPOSE:
#   Acts as a gatekeeper between the strategy and the paper
#   trader. Before any trade is placed, the risk manager checks
#   whether it is safe to do so.
#
# HOW IT WORKS:
#   For each signal the strategy generates, risk_manager:
#   1. Checks if we already have too many open positions.
#   2. Checks if we already hold this market.
#   3. Calculates a safe position size in USD.
#   4. Returns an approved or rejected trade order.
#
# KEY CONCEPTS FOR BEGINNERS:
#   - "Position sizing" means deciding HOW MUCH to bet, not just
#     whether to bet. Even a good signal should only risk a small
#     fraction of your portfolio.
#   - Fixed Fractional sizing (used here): always risk the same
#     percentage of your current balance, e.g. 5% per trade.
#     This keeps losses small and lets your account grow slowly.
# ============================================================

import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Validates signals and calculates safe position sizes.

    Usage:
        rm = RiskManager(config)
        order = rm.evaluate_signal(signal, portfolio_balance, open_positions)
    """

    def __init__(self, config: dict):
        risk_cfg = config["risk"]
        self.max_position_pct = risk_cfg["max_position_pct"]
        self.max_position_usd = risk_cfg["max_position_usd"]
        self.stop_loss_pct = risk_cfg["stop_loss_pct"]
        self.take_profit_pct = risk_cfg["take_profit_pct"]
        self.max_open_positions = risk_cfg["max_open_positions"]

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def evaluate_signal(
        self,
        signal: dict,
        portfolio_balance: float,
        open_positions: dict,
    ) -> dict | None:
        """
        Decide whether to act on a strategy signal.

        Args:
            signal:            Signal dict from Strategy.evaluate()
            portfolio_balance: Current virtual cash balance in USD
            open_positions:    Dict of currently open positions
                               {market_id: position_dict}

        Returns:
            An approved order dict, or None if the signal is rejected.
        """
        market_id = signal["market_id"]

        # --- Check 1: Too many open positions? ---
        if len(open_positions) >= self.max_open_positions:
            logger.debug(
                f"Rejected {market_id}: max open positions "
                f"({self.max_open_positions}) reached."
            )
            return None

        # --- Check 2: Already in this market? ---
        if market_id in open_positions:
            logger.debug(f"Rejected {market_id}: position already open.")
            return None

        # --- Check 3: Calculate position size ---
        size_usd = self._calculate_position_size(portfolio_balance)

        if size_usd <= 0:
            logger.warning(f"Rejected {market_id}: calculated size is $0.")
            return None

        # --- Build the approved order ---
        # Determine the entry price based on whether we're buying YES or NO
        from src.strategy import SIGNAL_BUY_YES
        if signal["signal"] == SIGNAL_BUY_YES:
            entry_price = signal["yes_price"]
            outcome = "YES"
        else:
            entry_price = signal["no_price"]
            outcome = "NO"

        # Pre-calculate stop and take-profit levels so the paper trader
        # can monitor them automatically
        stop_price = entry_price * (1 - self.stop_loss_pct)
        take_profit_price = entry_price * (1 + self.take_profit_pct)

        order = {
            "market_id": market_id,
            "question": signal["question"],
            "signal": signal["signal"],
            "outcome": outcome,           # "YES" or "NO"
            "entry_price": entry_price,
            "size_usd": size_usd,
            "shares": size_usd / entry_price,  # how many "shares" we get
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "edge": signal["edge"],
        }

        logger.info(
            f"Approved order: {outcome} on '{signal['question'][:50]}' | "
            f"Size: ${size_usd:.2f} @ {entry_price:.3f}"
        )
        return order

    def check_exit_conditions(
        self, position: dict, current_price: float
    ) -> str | None:
        """
        Check whether an open position should be closed.

        Args:
            position:      An open position dict (created by paper_trader)
            current_price: The latest market price for this position's outcome

        Returns:
            "STOP_LOSS", "TAKE_PROFIT", or None (keep holding)
        """
        entry = position["entry_price"]

        if current_price <= entry * (1 - self.stop_loss_pct):
            return "STOP_LOSS"

        if current_price >= entry * (1 + self.take_profit_pct):
            return "TAKE_PROFIT"

        return None  # no exit condition triggered

    # ------------------------------------------------------------------
    # Private / internal methods
    # ------------------------------------------------------------------

    def _calculate_position_size(self, balance: float) -> float:
        """
        Return the USD amount to risk on this trade.

        Uses fixed-fractional sizing: a set percentage of the current
        balance, capped at an absolute maximum.
        """
        # Fractional size based on current portfolio value
        fractional_size = balance * self.max_position_pct

        # Never exceed the hard cap
        size = min(fractional_size, self.max_position_usd)

        return round(size, 2)
