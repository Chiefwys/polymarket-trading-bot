# ============================================================
# src/strategy.py
#
# PURPOSE:
#   Analyzes market data and decides whether to BUY YES,
#   BUY NO, or do NOTHING for each market.
#
# HOW IT WORKS:
#   This file implements a simple "edge-based" strategy with
#   short-term momentum scoring layered on top:
#
#   1. MOMENTUM SCORING (NEW): Before evaluating any market,
#      we assign each market a priority score based on:
#        a) Category: crypto (3) > politics (2) > sports (1) > other (0)
#        b) Recent price movement: +1 bonus if the price moved
#           at least `min_price_change` since the last scan.
#      Markets are then sorted highest-score-first so the bot
#      looks at the most active/relevant markets first.
#
#   2. FAIR VALUE ESTIMATE: We use the short-term simple
#      moving average (SMA) of the YES price as our best
#      guess of the "true" probability.
#
#   3. EDGE CALCULATION: Edge = our fair value - current price.
#      A positive edge means the market is UNDERPRICED (buy YES).
#      A negative edge means the market is OVERPRICED (buy NO).
#
#   4. SIGNAL: We only act if |edge| >= min_edge (from config).
#      This avoids trading on noise.
#
# KEY CONCEPTS FOR BEGINNERS:
#   - "Edge" is the advantage you have over the market price.
#     If you think something has a 70% chance but trades at 60%,
#     your edge is 10 percentage points.
#   - "Momentum" here means: the price moved recently, suggesting
#     something is happening in that market worth paying attention to.
#   - We use SMA as a simple fair-value proxy. In a real bot you
#     might use fundamentals, news sentiment, or a model instead.
# ============================================================

import logging

logger = logging.getLogger(__name__)

# These constants make the signal values easy to read throughout the code
SIGNAL_BUY_YES = "BUY_YES"
SIGNAL_BUY_NO = "BUY_NO"
SIGNAL_HOLD = "HOLD"


class Strategy:
    """
    Generates trade signals for each market based on price edge,
    with optional momentum scoring and category prioritization.

    Usage:
        strategy = Strategy(config)
        signals = strategy.evaluate(markets)
    """

    def __init__(self, config: dict):
        self.min_edge = config["strategy"]["min_edge"]
        self.min_price = config["strategy"]["min_price"]
        self.max_price = config["strategy"]["max_price"]

        # -------------------------------------------------------
        # Momentum settings (from the new `momentum:` block in
        # config.yaml). We use .get() with safe defaults so the
        # bot still works even if the block is missing.
        # -------------------------------------------------------
        momentum_cfg = config["strategy"].get("momentum", {})

        # True = sort markets by score before evaluating
        self.momentum_enabled = momentum_cfg.get("enabled", False)

        # Minimum absolute price change to count as "moving"
        self.min_price_change = momentum_cfg.get("min_price_change", 0.02)

        # Category → score mapping; higher = processed first
        raw_scores = momentum_cfg.get("category_scores", {})
        self.category_scores = {
            "crypto":   raw_scores.get("crypto", 3),
            "politics": raw_scores.get("politics", 2),
            "sports":   raw_scores.get("sports", 1),
            "other":    raw_scores.get("other", 0),
        }

        # Momentum entry threshold: minimum price move to fire a directional
        # signal even when there is not enough SMA edge yet.
        # This is the key setting for generating more short-term test trades.
        self.momentum_entry_threshold = momentum_cfg.get(
            "momentum_entry_threshold", 0.001
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def evaluate(self, markets: list[dict]) -> list[dict]:
        """
        Evaluate a list of markets and return a list of signal dicts.

        If momentum is enabled, markets are sorted by their momentum score
        (category priority + recent price movement bonus) before evaluation,
        so the best short-term candidates appear first.

        Each signal dict contains:
            - market_id, question, yes_price, signal, edge, fair_value
            - category, price_change, momentum_score  (new fields)
        """
        signals = []

        # -------------------------------------------------------
        # Sort markets by momentum score if the feature is on.
        # sorted() is stable and non-destructive (returns new list).
        # reverse=True means highest score first.
        # -------------------------------------------------------
        if self.momentum_enabled:
            sorted_markets = sorted(
                markets, key=self._score_market, reverse=True
            )
            logger.info(
                "Momentum scoring ON - markets ordered by "
                "category priority + recent price movement."
            )
        else:
            sorted_markets = markets

        for market in sorted_markets:
            signal = self._evaluate_single(market)
            if signal["signal"] != SIGNAL_HOLD:
                # Only log actionable signals to keep output clean
                logger.info(
                    f"Signal: {signal['signal']} | "
                    f"[{signal['category']:8s}] score={signal['momentum_score']:.1f} | "
                    f"chg={signal['price_change']:+.3f} | "
                    f"Market: {signal['question'][:55]} | "
                    f"Price: {signal['yes_price']:.3f} | "
                    f"Edge: {signal['edge']:+.3f}"
                )
            signals.append(signal)

        actionable = [s for s in signals if s["signal"] != SIGNAL_HOLD]
        logger.info(
            f"Strategy evaluated {len(markets)} markets -> "
            f"{len(actionable)} actionable signals."
        )
        return signals

    # ------------------------------------------------------------------
    # Private / internal methods
    # ------------------------------------------------------------------

    def _evaluate_single(self, market: dict) -> dict:
        """
        Run the edge calculation for one market and return a signal dict.

        New fields added to the result:
            category       — "crypto" / "politics" / "sports" / "other"
            price_change   — how much the YES price moved since last scan
            momentum_score — combined priority score for this market
        """
        yes_price = market["yes_price"]
        sma = market.get("sma")  # may be None if not enough history yet

        # Base signal dict — we'll fill in the details below
        result = {
            "market_id": market["id"],
            "question": market["question"],
            "yes_price": yes_price,
            "no_price": market["no_price"],
            "fair_value": sma,
            "edge": 0.0,
            "signal": SIGNAL_HOLD,
            # --- New momentum fields ---
            "category": market.get("category", "other"),
            "price_change": market.get("price_change", 0.0),
            "momentum_score": self._score_market(market),
        }

        # --- Guard: skip markets outside our tradeable price range ---
        if not (self.min_price <= yes_price <= self.max_price):
            return result  # price is too extreme to trade safely

        # -------------------------------------------------------
        # SIGNAL PATH 1: SMA edge
        # Fire if the gap between fair value (SMA) and current
        # price is large enough. Requires at least 2 price scans.
        # -------------------------------------------------------
        if sma is not None:
            edge = sma - yes_price
            result["edge"] = edge

            if edge >= self.min_edge:
                result["signal"] = SIGNAL_BUY_YES
            elif edge <= -self.min_edge:
                result["signal"] = SIGNAL_BUY_NO

        # -------------------------------------------------------
        # SIGNAL PATH 2: Momentum direction entry (NEW)
        #
        # If the signal is still HOLD after the SMA check, we look
        # at the raw price change between the last two scans.
        # If the market moved enough in one direction, we ride it.
        #
        #   price went UP   -> BUY_YES  (ride upward momentum)
        #   price went DOWN -> BUY_NO   (ride downward momentum / bet YES falls)
        #
        # This fires even on the very first scan where SMA is None,
        # as long as we have seen the price move. It is what makes the
        # bot produce more short-term test trades.
        #
        # Set momentum_entry_threshold: 0 in config to disable this path.
        # -------------------------------------------------------
        if (
            result["signal"] == SIGNAL_HOLD
            and self.momentum_enabled
            and self.momentum_entry_threshold > 0
        ):
            price_change = result["price_change"]
            if price_change >= self.momentum_entry_threshold:
                result["signal"] = SIGNAL_BUY_YES
            elif price_change <= -self.momentum_entry_threshold:
                result["signal"] = SIGNAL_BUY_NO

        return result

    def _score_market(self, market: dict) -> float:
        """
        Compute a short-term momentum priority score for a market.

        The score has two components:
          1. Category score — from config.yaml:
               crypto=3, politics=2, sports=1, other=0
          2. Movement bonus — +1.0 if |price_change| >= min_price_change

        Higher score = more interesting for short-term trading.
        The score is used in evaluate() to sort markets before processing.

        Example:
            A crypto market that just moved 3% scores 3 + 1 = 4.
            A politics market with no movement scores 2 + 0 = 2.
            A random "other" market with no movement scores 0.
        """
        category = market.get("category", "other")
        # Look up the base score for this category (default 0 if unknown)
        score = float(self.category_scores.get(category, 0))

        # Bonus point if the market price has moved recently
        price_change = abs(market.get("price_change", 0.0))
        if price_change >= self.min_price_change:
            score += 1.0

        return score
