import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# NOTE: no Yahoo calls at startup
from services.dataset_builder import build_ml_dataset


class MLEngine:
    """
    Production-safe ML Engine
    (NO external API calls at startup)
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self._train_local()

    def _train_local(self):
        # Build dataset ONCE (historical data only)
        X_train, X_test, y_train, _ = build_ml_dataset()

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
        self.model.fit(X_train_scaled, y_train)

        # Use last known features as baseline
        self.latest_features = X_test.iloc[-1]

    def predict_probability(self):
        try:
            features_scaled = self.scaler.transform(
                self.latest_features.values.reshape(1, -1)
            )
            return float(self.model.predict_proba(features_scaled)[0][1])

        except Exception:
            # Fallback if anything breaks
            return 0.4
