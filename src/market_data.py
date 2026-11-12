# ============================================================
# src/market_data.py
#
# PURPOSE:
#   Fetches live market data from the Polymarket public API.
#   This is the only file that talks to the outside world.
#   Everything else in the bot consumes the clean data this
#   module produces.
#
# HOW IT WORKS:
#   1. We call the Polymarket Gamma API (free, no key needed).
#   2. We parse each market into a simple Python dict.
#   3. We maintain a short price history so the strategy can
#      calculate moving averages.
#
# KEY CONCEPTS FOR BEGINNERS:
#   - A "market" on Polymarket is a yes/no question (e.g.
#     "Will X happen by date Y?").
#   - The YES price is a probability between 0 and 1 (e.g.
#     0.65 means the crowd thinks there's a 65% chance of YES).
#   - NO price = 1 - YES price (they always sum to ~1).
# ============================================================

import json
import logging
import requests

logger = logging.getLogger(__name__)

# ============================================================
# Category keyword lists
#
# These are used by _detect_category() below to classify each
# market question into one of four buckets:
#   "crypto", "politics", "sports", or "other"
#
# We just check whether any of these words appear in the
# lowercased question text — simple but effective.
# ============================================================

CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "eth", "ethereum", "crypto", "solana", "sol",
    "doge", "dogecoin", "xrp", "ripple", "coinbase", "binance",
    "token", "nft", "defi", "blockchain", "altcoin", "usdc",
    "stablecoin", "bnb", "avalanche", "avax", "polygon", "matic",
]

POLITICS_KEYWORDS = [
    "trump", "biden", "harris", "election", "president", "congress",
    "senate", "republican", "democrat", "vote", "governor",
    "parliament", "prime minister", "white house", "ballot",
    "political", "party", "legislation", "bill", "impeach",
]

SPORTS_KEYWORDS = [
    "nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball",
    "baseball", "championship", "tournament", "super bowl", "playoffs",
    "world cup", "league", "match", "season", "team", "wins",
    "olympic", "formula 1", "f1", "tennis", "golf", "ufc", "mma",
]


class MarketDataFetcher:
    """
    Fetches and caches market data from Polymarket's public API.

    Usage:
        fetcher = MarketDataFetcher(config)
        markets = fetcher.get_markets()
    """

    def __init__(self, config: dict):
        # Pull settings out of the config dict (loaded from config.yaml)
        self.base_url = config["market_data"]["base_url"]
        self.limit = config["market_data"]["markets_limit"]
        self.sma_window = config["strategy"]["sma_window"]

        # Price history stores a list of recent prices for each market id.
        # Format: { market_id: [0.60, 0.62, 0.61, ...] }
        self._price_history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_markets(self) -> list[dict]:
        """
        Fetch active markets from the Polymarket API and return a list
        of cleaned market dicts.

        Returns [] on any network or parsing error so the bot keeps
        running instead of crashing.
        """
        raw_markets = self._fetch_active_markets()
        cleaned = []

        for raw in raw_markets:
            market = self._parse_market(raw)
            if market is None:
                continue  # skip markets we couldn't parse

            # Update the rolling price history for this market
            self._update_price_history(market["id"], market["yes_price"])

            # Attach the current simple moving average to the market dict
            market["sma"] = self._get_sma(market["id"])

            # Attach the category label (crypto / politics / sports / other)
            market["category"] = self._detect_category(market["question"])

            # Attach the latest price change (current price minus previous price).
            # Will be 0.0 on the first observation when there's no history yet.
            market["price_change"] = self._get_price_change(market["id"])

            cleaned.append(market)

        # -------------------------------------------------------
        # Print every market being scanned so you can see what
        # the bot is evaluating this cycle.
        # Format: [rank] [category] chg=+0.003 | question text
        # -------------------------------------------------------
        logger.info(f"--- Markets scanned this cycle ({len(cleaned)} total) ---")
        for i, m in enumerate(cleaned, 1):
            cat = m.get("category", "other")
            chg = m.get("price_change", 0.0)
            # Show sign on price change so +/- is immediately obvious
            chg_str = f"{chg:+.3f}" if chg != 0.0 else "new"
            logger.info(
                f"  [{i:3d}] [{cat:8s}] chg={chg_str} | {m['question'][:75]}"
            )
        logger.info(f"--- End of market list ---")

        logger.info(f"Fetched {len(cleaned)} active markets.")
        return cleaned

    # ------------------------------------------------------------------
    # Private / internal methods (prefixed with _)
    # ------------------------------------------------------------------

    def _fetch_active_markets(self) -> list[dict]:
        """Call the Polymarket API and return raw JSON market objects."""
        url = f"{self.base_url}/markets"
        params = {
            "active": "true",        # only open markets
            "closed": "false",
            "limit": self.limit,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # raises exception for 4xx/5xx errors
            data = response.json()

            # The API returns a list directly or nested — handle both shapes
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "markets" in data:
                return data["markets"]
            logger.warning(f"Unexpected API response shape: {type(data)}")
            return []

        except requests.exceptions.Timeout:
            logger.error("API request timed out.")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []

    def _parse_market(self, raw: dict) -> dict | None:
        """
        Extract only the fields we care about from a raw API market object.

        Returns None if the market data is missing required fields.
        """
        try:
            # The API returns outcomePrices as a JSON string — parse it first
            outcome_prices = raw.get("outcomePrices", '["0", "0"]')
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            yes_price = float(outcome_prices[0])
            no_price = float(outcome_prices[1])

            return {
                "id": raw["id"],
                "question": raw.get("question", "Unknown"),
                "yes_price": yes_price,
                "no_price": no_price,
                "volume": float(raw.get("volume", 0)),
                "end_date": raw.get("endDate", ""),
                "active": raw.get("active", False),
            }
        except (KeyError, ValueError, IndexError, TypeError) as e:
            logger.debug(f"Could not parse market: {e} | raw={raw}")
            return None

    def _update_price_history(self, market_id: str, price: float):
        """
        Append the latest price to the history list for this market,
        keeping only the last `sma_window` observations.
        """
        if market_id not in self._price_history:
            self._price_history[market_id] = []

        history = self._price_history[market_id]
        history.append(price)

        # Trim to the window size so we don't grow forever
        if len(history) > self.sma_window:
            self._price_history[market_id] = history[-self.sma_window:]

    def _get_sma(self, market_id: str) -> float | None:
        """
        Return the simple moving average of recent prices for a market,
        or None if we don't have enough data yet.
        """
        history = self._price_history.get(market_id, [])
        if len(history) < 2:
            return None  # not enough data points yet
        return sum(history) / len(history)

    def _get_price_change(self, market_id: str) -> float:
        """
        Return the price change between the two most recent observations.

        e.g. if the last two YES prices were [0.60, 0.63], this returns +0.03.
        Returns 0.0 if we don't have at least two data points yet (first scan).

        This is used by the strategy to find markets that are actively moving —
        a sign of short-term momentum worth trading.
        """
        history = self._price_history.get(market_id, [])
        if len(history) < 2:
            return 0.0
        return history[-1] - history[-2]  # latest minus previous

    def _detect_category(self, question: str) -> str:
        """
        Classify a market question into a category for momentum scoring.

        We convert the question to lowercase and check whether any of our
        keyword lists match. First match wins (crypto > politics > sports).
        Returns "other" if no keywords match.

        This is intentionally simple — it catches the most common cases
        without needing any external libraries or ML models.
        """
        q = question.lower()

        if any(kw in q for kw in CRYPTO_KEYWORDS):
            return "crypto"

        if any(kw in q for kw in POLITICS_KEYWORDS):
            return "politics"

        if any(kw in q for kw in SPORTS_KEYWORDS):
            return "sports"

        return "other"
