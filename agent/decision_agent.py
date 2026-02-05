class DecisionAgent:
    """
    Chat-style investment decision agent
    """

    def __init__(self, capital: int, risk: str = "medium"):
        self.capital = capital
        self.risk = risk

        # Probability threshold by risk appetite
        self.thresholds = {
            "low": 0.50,
            "medium": 0.40,
            "high": 0.30
        }

    def decide(self, prob_up: float) -> dict:
        """
        Decide BUY or HOLD based on probability
        """

        threshold = self.thresholds.get(self.risk, 0.40)

        if prob_up < threshold:
            return {
                "action": "HOLD",
                "invest_amount": 0,
                "reply": (
                    f"I don't recommend investing right now. "
                    f"The probability of price increase is only "
                    f"{round(prob_up * 100, 2)}%, which is below your "
                    f"{self.risk}-risk safety threshold."
                )
            }

        # Capital allocation based on risk
        if self.risk == "low":
            allocation = 0.3
        elif self.risk == "medium":
            allocation = 0.6
        else:
            allocation = 0.9

        invest_amount = int(self.capital * allocation)

        return {
            "action": "BUY",
            "invest_amount": invest_amount,
            "reply": (
                f"Based on current market trends, the probability of a price "
                f"increase is {round(prob_up * 100, 2)}%. "
                f"I recommend investing ₹{invest_amount} considering your "
                f"{self.risk}-risk profile."
            )
        }
