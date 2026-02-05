import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from services.dataset_builder import build_ml_dataset


class MLEngine:
    """
    ML Engine that trains once and predicts probabilities
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self._train()

    def _train(self):
        # Load dataset
        X_train, X_test, y_train, y_test = build_ml_dataset()

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train balanced model
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
        self.model.fit(X_train_scaled, y_train)

        # Keep last row for live prediction
        self.latest_features = X_test.iloc[-1]

    def predict_probability(self):
        """
        Predict probability of price going UP
        """
        features_scaled = self.scaler.transform(
            self.latest_features.values.reshape(1, -1)
        )

        prob_up = float(self.model.predict_proba(features_scaled)[0][1])
        return prob_up


if __name__ == "__main__":
    engine = MLEngine()
    prob = engine.predict_probability()
    print("Probability of price going UP:", round(prob * 100, 2), "%")
