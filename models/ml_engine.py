import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from services.dataset_builder import build_ml_dataset


class MLEngine:
    """
    Production-safe ML Engine
    - Never crashes on startup
    - Uses fallback when market data is unavailable
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.latest_features = None
        self._train_local()

    def _train_local(self):
        """
        Train model once at startup.
        If data is unavailable (Yahoo rate limit), fall back safely.
        """
        X_train, X_test, y_train, _ = build_ml_dataset()

        # ⚠️ Fallback mode (cloud / rate limit)
        if X_train is None or X_test is None:
            print("⚠️ ML dataset unavailable. Starting in fallback mode.")
            self.model = None
            self.scaler = None
            self.latest_features = None
            return

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
        self.model.fit(X_train_scaled, y_train)

        # Save latest feature row for live prediction
        self.latest_features = X_test.iloc[-1]

        print("✅ ML model trained successfully")

    def predict_probability(self):
        """
        Predict probability of price going UP.
        Always returns a value.
        """

        # 🛟 Fallback probability if model not trained
        if self.model is None or self.latest_features is None:
            return 0.4  # neutral-safe probability

        features_scaled = self.scaler.transform(
            self.latest_features.values.reshape(1, -1)
        )

        return float(self.model.predict_proba(features_scaled)[0][1])


if __name__ == "__main__":
    engine = MLEngine()
    prob = engine.predict_probability()
    print("Predicted probability:", prob)
