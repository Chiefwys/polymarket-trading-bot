# ============================================================
# app.py — Main entry point
#
# PURPOSE:
#   This is the file you run to start the bot.
#   It wires all the modules together and runs the main loop.
#
# HOW TO RUN:
#   1. Activate your virtual environment:
#        Windows:  botenv\Scripts\activate
#        Mac/Linux: source botenv/bin/activate
#
#   2. Install dependencies:
#        pip install -r requirements.txt
#
#   3. Start the bot:
#        python app.py
#
# MAIN LOOP (runs every `poll_interval_seconds` from config.yaml):
#   1. Fetch latest market data           (market_data.py)
#   2. Evaluate each market for signals   (strategy.py)
#   3. Check signals through risk manager (risk_manager.py)
#   4. Execute approved trades (simulated)(paper_trader.py)
#   5. Monitor open positions for exits   (paper_trader.py)
#   6. Wait, then repeat
# ============================================================

import logging
import os
import time
from collections import Counter

import yaml
from dotenv import load_dotenv

from src.market_data import MarketDataFetcher
from src.paper_trader import PaperTrader
from src.risk_manager import RiskManager
from src.strategy import Strategy, SIGNAL_HOLD


# ---------------------------------------------------------------
# Step 1: Load secrets from .env
# ---------------------------------------------------------------
# python-dotenv reads your .env file and makes the values
# available via os.getenv(). This keeps API keys out of code.
load_dotenv()


# ---------------------------------------------------------------
# Step 2: Load configuration from config.yaml
# ---------------------------------------------------------------
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------
# Step 3: Set up logging
# ---------------------------------------------------------------
def setup_logging(config: dict):
    """
    Configure Python's built-in logging module.

    - Logs go to both the console AND the log file defined in config.yaml
    - Log level controls verbosity: DEBUG shows everything, INFO is normal
    """
    log_cfg = config["logging"]
    level = getattr(logging, log_cfg["level"].upper(), logging.INFO)
    log_file = log_cfg["log_file"]

    # Ensure the logs/ directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),               # Print to console
            logging.FileHandler(log_file),         # Write to file
        ],
    )


# ---------------------------------------------------------------
# Step 4: The main bot loop
# ---------------------------------------------------------------
def run_bot(config: dict):
    """
    Initialize all modules and run the trading loop indefinitely.
    Press Ctrl+C to stop the bot gracefully.
    """
    logger = logging.getLogger("app")
    poll_interval = config["market_data"]["poll_interval_seconds"]

    # Instantiate each module — each one only needs the config dict
    fetcher = MarketDataFetcher(config)
    strategy = Strategy(config)
    risk_manager = RiskManager(config)
    trader = PaperTrader(config)

    logger.info("Bot started. Press Ctrl+C to stop.")
    logger.info(f"Scanning every {poll_interval}s. Paper-trading only.")

    cycle = 0

    try:
        while True:
            cycle += 1
            logger.info(f"--- Cycle {cycle} ---")

            # --- A: Fetch latest market data ---
            markets = fetcher.get_markets()

            if not markets:
                logger.warning("No market data received. Waiting before retry.")
                time.sleep(poll_interval)
                continue

            # --- A2: Print a compact market scan summary ---
            # Count how many markets fall into each category so you can
            # see at a glance what types of markets are in the feed.
            category_counts = Counter(m.get("category", "other") for m in markets)
            # Count how many markets are "moving" (recent price change)
            min_chg = config["strategy"].get("momentum", {}).get("min_price_change", 0.005)
            moving = sum(1 for m in markets if abs(m.get("price_change", 0.0)) >= min_chg)
            logger.info(
                f"Scan summary: {len(markets)} markets | "
                f"crypto={category_counts.get('crypto', 0)} "
                f"politics={category_counts.get('politics', 0)} "
                f"sports={category_counts.get('sports', 0)} "
                f"other={category_counts.get('other', 0)} | "
                f"moving (chg>={min_chg}): {moving}"
            )

            # --- B: Generate signals for all markets ---
            signals = strategy.evaluate(markets)

            # --- C: Process each actionable signal ---
            for signal in signals:
                if signal["signal"] == SIGNAL_HOLD:
                    continue  # strategy found no edge, skip

                # Ask the risk manager if it's safe to trade
                order = risk_manager.evaluate_signal(
                    signal=signal,
                    portfolio_balance=trader.balance,
                    open_positions=trader.open_positions,
                )

                if order is None:
                    continue  # risk manager rejected the trade

                # Attach extra analytics fields from the signal to the order.
                # These come from the strategy and are stored in trades.csv
                # so you can filter/analyze by category, momentum score, etc.
                order["category"]       = signal.get("category", "other")
                order["momentum_score"] = signal.get("momentum_score", 0.0)
                order["price_change"]   = signal.get("price_change", 0.0)
                # edge is already set by risk_manager but ensure it's current
                order["edge"]           = signal.get("edge", order.get("edge", 0.0))

                # Simulate opening the position
                trader.open_position(order)

            # --- D: Check open positions for stop/take-profit exits ---
            trader.monitor_positions(markets, risk_manager)

            # --- E: Print a quick status update ---
            logger.info(
                f"Balance: ${trader.balance:.2f} | "
                f"Open positions: {len(trader.open_positions)} | "
                f"Closed trades: {len(trader.trade_history)}"
            )

            # --- F: Wait before the next cycle ---
            logger.info(f"Sleeping {poll_interval}s until next scan...")
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        # -------------------------------------------------------
        # Graceful shutdown on Ctrl+C
        # -------------------------------------------------------
        # We do NOT force-close open positions — they remain open so
        # they can be resumed the next time the bot starts.
        # -------------------------------------------------------
        print("\n")  # blank line after the ^C character in the terminal
        logger.info("Shutdown requested (Ctrl+C). Saving state...")

        # Save open positions to open_positions.json and update
        # performance_summary.json with the final session stats.
        trader.save_state()

        logger.info("State saved.")

        # Print the clean final summary to the console
        trader.print_summary()


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    config = load_config()
    setup_logging(config)
    run_bot(config)
