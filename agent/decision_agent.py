class DecisionAgent:
    """
    Chat-style, asset-aware investment decision agent
    """

    def __init__(
        self,
        capital: int,
        risk: str = "medium",
        asset: str = "general"
    ):
        self.capital = capital
        self.risk = risk.lower()
        self.asset = asset.lower()

        # Probability threshold by risk appetite
        self.thresholds = {
            "low": 0.50,
            "medium": 0.40,
            "high": 0.30
        }

        # Capital allocation by risk
        self.allocations = {
            "low": 0.30,
            "medium": 0.60,
            "high": 0.90
        }

    def decide(
        self,
        prob_up: float,
        live_price: float | None = None
    ) -> dict:
        """
        Decide BUY or HOLD using:
        - ML probability
        - Risk profile
        - Asset class
        - Optional real-time price
        """

        threshold = self.thresholds.get(self.risk, 0.40)
        allocation = self.allocations.get(self.risk, 0.60)

        # ----------------------------
        # Asset-specific guidance
        # ----------------------------
        asset_note = ""

        if self.asset == "commodity":
            asset_note = (
                "Commodities like gold and silver are typically used for "
                "inflation hedging and long-term stability, not aggressive gains."
            )

        elif self.asset == "crypto":
            asset_note = (
                "Cryptocurrencies are highly volatile assets. "
                "Invest only if you can tolerate sharp price fluctuations."
            )

        elif self.asset == "stock":
            asset_note = (
                "Equity investments depend on company fundamentals, "
                "earnings growth, and market trends."
            )

        # ----------------------------
        # HOLD decision
        # ----------------------------
        if prob_up < threshold:
            price_info = (
                f" Current price is approximately ₹{live_price}."
                if live_price is not None
                else ""
            )

            return {
                "action": "HOLD",
                "invest_amount": 0,
                "reply": (
                    f"I don’t recommend investing right now. "
                    f"The probability of price increase is "
                    f"{round(prob_up * 100, 2)}%, which is below your "
                    f"{self.risk}-risk comfort level."
                    f"{price_info} "
                    f"{asset_note}"
                )
            }

        # ----------------------------
        # BUY decision
        # ----------------------------
        invest_amount = int(self.capital * allocation)

        price_info = (
            f" The current market price is around ₹{live_price}."
            if live_price is not None
            else ""
        )

        return {
            "action": "BUY",
            "invest_amount": invest_amount,
            "reply": (
                f"Market conditions look favorable with a "
                f"{round(prob_up * 100, 2)}% probability of upside."
                f"{price_info} "
                f"Based on your {self.risk}-risk profile, "
                f"you could consider investing approximately "
                f"₹{invest_amount}. "
                f"{asset_note}"
            )
        }
