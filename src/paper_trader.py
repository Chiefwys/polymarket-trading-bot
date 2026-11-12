# ============================================================
# src/paper_trader.py
#
# PURPOSE:
#   Simulates trade execution without using real money.
#   Maintains a virtual portfolio and saves all data to files
#   so you can analyze results and resume across restarts.
#
# HOW IT WORKS:
#   1. open_position()       — "buys" shares and deducts from balance
#   2. close_position()      — "sells" shares and adds payout to balance
#   3. monitor_positions()   — checks open positions every cycle,
#                              closes any that hit stop/take-profit
#   4. save_state()          — saves open positions + performance summary
#                              to JSON files (called automatically)
#   5. _load_open_positions()— restores positions from the last session
#                              on startup
#
# OUTPUT FILES (all in logs/):
#   trades.csv             — every OPEN and CLOSE event with full details
#   open_positions.json    — current open positions (for resume on restart)
#   performance_summary.json — compact session stats snapshot
#
# KEY CONCEPTS FOR BEGINNERS:
#   - "Paper trading" means simulating trades with fake money.
#   - On Polymarket, YES shares pay $1.00 if the event resolves YES,
#     $0.00 if NO. Since we exit early via stop/take-profit, we
#     simulate the payout at the current market price instead.
#   - "Unrealized P&L" = profit/loss on positions still open
#   - "Realized P&L"   = profit/loss on positions already closed
# ============================================================

import csv
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Simulates a paper trading portfolio for Polymarket.

    Usage:
        trader = PaperTrader(config)
        trader.open_position(order)
        trader.monitor_positions(current_markets, risk_manager)
        trader.save_state()      # called automatically; also on shutdown
    """

    def __init__(self, config: dict):
        # -------------------------------------------------------
        # Portfolio state
        # -------------------------------------------------------
        self.starting_balance = config["paper_trading"]["starting_balance"]
        self.balance          = self.starting_balance  # may be overwritten by load
        self.currency         = config["paper_trading"]["currency"]

        # -------------------------------------------------------
        # File paths — all files live in the logs/ folder
        # -------------------------------------------------------
        self.log_file = config["logging"]["log_file"]

        # Derive the logs directory from the log file path.
        # e.g.  "logs/trades.log"  →  logs_dir = "logs"
        # The  `or "logs"` handles the edge case where log_file has no directory.
        logs_dir = os.path.dirname(self.log_file) or "logs"

        self.trades_csv       = os.path.join(logs_dir, "trades.csv")
        self.positions_json   = os.path.join(logs_dir, "open_positions.json")
        self.summary_json     = os.path.join(logs_dir, "performance_summary.json")

        # -------------------------------------------------------
        # In-memory collections
        # -------------------------------------------------------

        # Open positions: { market_id: { ...position details... } }
        self.open_positions: dict[str, dict] = {}

        # Full history of completed trades this session
        self.trade_history: list[dict] = []

        # Most recent known price per market — used to estimate unrealized P&L
        # at shutdown. Updated each call to monitor_positions().
        self._last_market_prices: dict[str, float] = {}

        # -------------------------------------------------------
        # Initialise files
        # -------------------------------------------------------
        os.makedirs(logs_dir, exist_ok=True)

        # Write the CSV header row only if the file does not exist yet
        # (so existing trade history is preserved across restarts)
        self._init_trades_csv()

        # -------------------------------------------------------
        # Resume from previous session if a save file exists
        # -------------------------------------------------------
        self._load_open_positions()

        logger.info(
            f"Paper trader started. "
            f"Balance: ${self.balance:.2f} {self.currency} | "
            f"Resumed {len(self.open_positions)} open position(s) from last session."
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def open_position(self, order: dict) -> bool:
        """
        Simulate buying shares based on an approved order.

        Args:
            order: An approved order dict from RiskManager.evaluate_signal().
                   app.py also attaches extra analytics fields to this dict:
                     category, momentum_score, price_change
                   (these come from the original strategy signal)

        Returns:
            True if position was opened successfully, False otherwise.
        """
        cost = order["size_usd"]

        # Guard: make sure we have enough virtual cash
        if cost > self.balance:
            logger.warning(
                f"Insufficient balance (${self.balance:.2f}) "
                f"to open position costing ${cost:.2f}."
            )
            return False

        # Deduct the cost from our virtual balance
        self.balance -= cost

        # Build the position record — include analytics fields from the signal.
        # We use .get() with safe defaults so nothing breaks if a field is missing.
        position = {
            "market_id":         order["market_id"],
            "question":          order["question"],
            "signal":            order["signal"],
            "outcome":           order["outcome"],        # "YES" or "NO"
            "entry_price":       order["entry_price"],
            "shares":            order["shares"],
            "cost_usd":          cost,
            "stop_price":        order["stop_price"],
            "take_profit_price": order["take_profit_price"],
            "opened_at":         datetime.now(timezone.utc).isoformat(),
            # --- Analytics fields (attached by app.py from the signal dict) ---
            "category":          order.get("category", "other"),
            "momentum_score":    order.get("momentum_score", 0.0),
            "edge":              order.get("edge", 0.0),
            "price_change":      order.get("price_change", 0.0),
            # signal_reason mirrors the signal type for easy filtering in CSV
            "signal_reason":     order["signal"],
        }

        self.open_positions[order["market_id"]] = position

        # Write a row to trades.csv for this OPEN event
        self._append_trade_csv(
            action="OPEN",
            position=position,
            exit_price=None,
            pnl=None,
            exit_reason="",
        )

        # Persist open positions and performance summary to disk immediately.
        # This means you never lose data even if the bot crashes between scans.
        self.save_state()

        logger.info(
            f"Opened position: {order['outcome']} on "
            f"'{order['question'][:50]}' | "
            f"Shares: {order['shares']:.2f} @ ${order['entry_price']:.3f} | "
            f"Cost: ${cost:.2f} | Balance: ${self.balance:.2f}"
        )
        return True

    def close_position(
        self, market_id: str, exit_price: float, reason: str
    ) -> bool:
        """
        Simulate selling shares and recording the P&L.

        Args:
            market_id:  The market to close
            exit_price: The current price at which we "sell" our shares
            reason:     Why we're closing ("STOP_LOSS", "TAKE_PROFIT", "MANUAL")

        Returns:
            True if position was closed, False if the position was not found.
        """
        if market_id not in self.open_positions:
            logger.warning(f"Tried to close non-existent position: {market_id}")
            return False

        position = self.open_positions.pop(market_id)

        # Calculate payout: shares × current price.
        # On Polymarket, price = implied probability = value per share.
        proceeds = position["shares"] * exit_price
        pnl      = proceeds - position["cost_usd"]

        # Add proceeds back to our virtual balance
        self.balance += proceeds

        # Save to in-memory trade history for the session summary
        closed_trade = {
            **position,
            "exit_price": exit_price,
            "pnl":        pnl,
            "exit_reason": reason,
            "closed_at":  datetime.now(timezone.utc).isoformat(),
        }
        self.trade_history.append(closed_trade)

        # Write a row to trades.csv for this CLOSE event
        self._append_trade_csv(
            action="CLOSE",
            position=position,
            exit_price=exit_price,
            pnl=pnl,
            exit_reason=reason,
        )

        # Persist state after every close
        self.save_state()

        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        logger.info(
            f"Closed position [{reason}]: {position['outcome']} on "
            f"'{position['question'][:50]}' | "
            f"Exit: ${exit_price:.3f} | P&L: {pnl_str} | "
            f"Balance: ${self.balance:.2f}"
        )
        return True

    def monitor_positions(self, markets: list[dict], risk_manager):
        """
        Check all open positions against current market prices and
        close any that have hit their stop-loss or take-profit levels.

        Also updates _last_market_prices so we can estimate unrealized P&L.

        Args:
            markets:      Latest market data from MarketDataFetcher
            risk_manager: RiskManager instance to check exit conditions
        """
        if not self.open_positions:
            return

        # Build a quick lookup: market_id → market dict
        market_lookup = {m["id"]: m for m in markets}

        # Remember the latest price for every held position.
        # We use this to estimate unrealized P&L when saving state.
        for market_id, position in self.open_positions.items():
            market = market_lookup.get(market_id)
            if market:
                price = (
                    market["yes_price"] if position["outcome"] == "YES"
                    else market["no_price"]
                )
                self._last_market_prices[market_id] = price

        # Iterate over a copy of keys because close_position() mutates the dict
        for market_id in list(self.open_positions.keys()):
            position = self.open_positions[market_id]
            market   = market_lookup.get(market_id)

            if market is None:
                # Market has dropped out of the API feed — skip for now
                logger.debug(f"Market {market_id} no longer in feed, skipping.")
                continue

            # Get the current price for the outcome we hold
            current_price = (
                market["yes_price"] if position["outcome"] == "YES"
                else market["no_price"]
            )

            # Ask the risk manager whether we should exit
            exit_reason = risk_manager.check_exit_conditions(position, current_price)

            if exit_reason:
                self.close_position(market_id, current_price, exit_reason)

    def save_state(self):
        """
        Save open positions + performance summary to JSON files.

        Called automatically after every open/close event and on shutdown
        (Ctrl+C). You can also call this manually at any time.
        """
        self._save_open_positions()
        self._save_performance_summary()

    def print_summary(self):
        """
        Print a clean, human-readable portfolio summary to the console.
        Called automatically on Ctrl+C shutdown.
        """
        realized_pnl   = sum(t["pnl"] for t in self.trade_history)
        wins           = [t for t in self.trade_history if t["pnl"] > 0]
        losses         = [t for t in self.trade_history if t["pnl"] <= 0]
        win_rate       = len(wins) / len(self.trade_history) if self.trade_history else 0
        unrealized_pnl = self._compute_unrealized_pnl()

        print("\n" + "=" * 60)
        print("  PAPER TRADING SUMMARY")
        print("=" * 60)
        print(f"  Starting Balance : ${self.starting_balance:.2f}")
        print(f"  Current Balance  : ${self.balance:.2f}")
        print(f"  Open Positions   : {len(self.open_positions)}")
        print(f"  Closed Trades    : {len(self.trade_history)}")
        print(f"  Wins / Losses    : {len(wins)} / {len(losses)}")
        print(f"  Win Rate         : {win_rate:.1%}")
        print(f"  Realized P&L     : {'+'if realized_pnl>=0 else ''}${realized_pnl:.2f}")
        if unrealized_pnl is not None:
            sign = "+" if unrealized_pnl >= 0 else ""
            print(f"  Unrealized P&L   : {sign}${unrealized_pnl:.2f}  (open positions, last known price)")
        print(f"\n  Saved files:")
        print(f"    {self.trades_csv}")
        print(f"    {self.positions_json}")
        print(f"    {self.summary_json}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Private / internal methods
    # ------------------------------------------------------------------

    def _init_trades_csv(self):
        """
        Create trades.csv and write the header row — but only if the file
        does not already exist. This preserves trade history across restarts.
        """
        if not os.path.exists(self.trades_csv):
            with open(self.trades_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",       # UTC time of this event
                    "market_id",       # Polymarket market ID
                    "question",        # Market question text
                    "category",        # crypto / politics / sports / other
                    "side",            # YES or NO
                    "action",          # OPEN or CLOSE
                    "entry_price",     # price when position was opened
                    "exit_price",      # price when position was closed (blank on OPEN)
                    "size_usd",        # USD spent to open the position
                    "shares",          # number of shares purchased
                    "pnl",             # profit or loss in USD (blank on OPEN)
                    "signal_reason",   # BUY_YES or BUY_NO
                    "momentum_score",  # priority score from strategy
                    "edge",            # SMA edge at time of signal
                    "price_change",    # price move that triggered signal
                    "exit_reason",     # STOP_LOSS / TAKE_PROFIT / MANUAL (blank on OPEN)
                ])

    def _append_trade_csv(
        self,
        action: str,
        position: dict,
        exit_price: float | None,
        pnl: float | None,
        exit_reason: str,
    ):
        """
        Append a single row to trades.csv for one OPEN or CLOSE event.

        For OPEN events:  exit_price, pnl, exit_reason are blank.
        For CLOSE events: all columns are filled in.
        """
        with open(self.trades_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),          # timestamp
                position["market_id"],                            # market_id
                position["question"],                             # question
                position.get("category", "other"),               # category
                position["outcome"],                              # side
                action,                                           # OPEN / CLOSE
                f"{position['entry_price']:.4f}",                 # entry_price
                f"{exit_price:.4f}" if exit_price is not None else "",  # exit_price
                f"{position['cost_usd']:.2f}",                    # size_usd
                f"{position['shares']:.4f}",                      # shares
                f"{pnl:.4f}" if pnl is not None else "",          # pnl
                position.get("signal_reason", ""),                # signal_reason
                f"{position.get('momentum_score', 0.0):.2f}",    # momentum_score
                f"{position.get('edge', 0.0):.4f}",               # edge
                f"{position.get('price_change', 0.0):.4f}",       # price_change
                exit_reason,                                      # exit_reason
            ])

    def _save_open_positions(self):
        """
        Write all currently open positions to open_positions.json.

        Format:
            {
              "saved_at": "<ISO timestamp>",
              "balance":  950.00,
              "positions": { "<market_id>": { ...position dict... }, ... }
            }

        This file is read back on the next startup by _load_open_positions()
        so the bot can resume tracking positions without losing them.
        """
        data = {
            "saved_at":  datetime.now(timezone.utc).isoformat(),
            "balance":   round(self.balance, 2),
            "positions": self.open_positions,  # already JSON-serializable
        }
        try:
            with open(self.positions_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error(f"Could not save open_positions.json: {e}")

    def _load_open_positions(self):
        """
        Load open positions from a previous session on startup.

        - If open_positions.json exists and has positions, restore them
          into self.open_positions and restore the saved balance.
        - If the file doesn't exist or has no positions, start fresh.
        - If the file is corrupted, log a warning and start fresh.
        """
        if not os.path.exists(self.positions_json):
            # No previous session file — this is a fresh start
            return

        try:
            with open(self.positions_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            saved_positions = data.get("positions", {})
            saved_balance   = data.get("balance", self.starting_balance)
            saved_at        = data.get("saved_at", "unknown time")

            if not saved_positions:
                logger.info(
                    "open_positions.json found but contains no open positions — "
                    "starting with a fresh portfolio."
                )
                return

            # Restore saved state into the trader
            self.open_positions = saved_positions
            self.balance        = saved_balance

            logger.info(
                f"Loaded previous session (saved {saved_at}): "
                f"{len(self.open_positions)} open position(s) | "
                f"balance=${self.balance:.2f}"
            )

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(
                f"Could not load open_positions.json ({e}) — "
                f"starting with a fresh portfolio."
            )

    def _save_performance_summary(self):
        """
        Write a compact performance snapshot to performance_summary.json.

        This file is overwritten on every save so it always reflects
        the latest state. Upload it for quick analysis.

        Fields:
            starting_balance      — balance at bot start
            current_balance       — current virtual cash
            open_positions_count  — number of positions still open
            closed_trades_count   — number of completed trades this session
            wins                  — trades with positive P&L
            losses                — trades with zero or negative P&L
            win_rate              — wins / total closed trades (0–1)
            realized_pnl          — total P&L on closed trades
            unrealized_pnl        — estimated P&L on open trades (or null)
        """
        realized_pnl = sum(t["pnl"] for t in self.trade_history)
        wins         = [t for t in self.trade_history if t["pnl"] > 0]
        losses       = [t for t in self.trade_history if t["pnl"] <= 0]
        win_rate     = len(wins) / len(self.trade_history) if self.trade_history else 0.0

        unrealized_pnl = self._compute_unrealized_pnl()

        summary = {
            "saved_at":             datetime.now(timezone.utc).isoformat(),
            "starting_balance":     round(self.starting_balance, 2),
            "current_balance":      round(self.balance, 2),
            "open_positions_count": len(self.open_positions),
            "closed_trades_count":  len(self.trade_history),
            "wins":                 len(wins),
            "losses":               len(losses),
            "win_rate":             round(win_rate, 4),
            "realized_pnl":         round(realized_pnl, 2),
            # None means we haven't seen any prices yet for open positions
            "unrealized_pnl":       (
                round(unrealized_pnl, 2) if unrealized_pnl is not None else None
            ),
        }

        try:
            with open(self.summary_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except OSError as e:
            logger.error(f"Could not save performance_summary.json: {e}")

    def _compute_unrealized_pnl(self) -> float | None:
        """
        Estimate unrealized P&L for all open positions using the most
        recently seen market prices (stored in _last_market_prices by
        monitor_positions()).

        Returns None if we have no price data for any open position yet
        (e.g., the bot just started and hasn't run monitor_positions yet).
        """
        if not self.open_positions or not self._last_market_prices:
            return None

        total    = 0.0
        has_data = False

        for market_id, position in self.open_positions.items():
            price = self._last_market_prices.get(market_id)
            if price is None:
                # No price seen for this position yet — skip it
                continue
            proceeds   = position["shares"] * price
            unrealized = proceeds - position["cost_usd"]
            total     += unrealized
            has_data   = True

        return round(total, 2) if has_data else None
