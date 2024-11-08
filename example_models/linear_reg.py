from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Dict, List
from datetime import datetime


class PredictionModel:
    def __init__(self):
        """Initialize the linear regression model for cricket score prediction."""
        self.model = LinearRegression()
        self.is_trained = False

    def _extract_features(
        self, player_history: Dict, venue: str = None, opposition: str = None
    ) -> List[float]:
        """Extract relevant features from player history for prediction.

        Features:
        1. Recent form (avg of last 5 matches)
        2. Career average
        3. Venue performance
        4. Opposition performance
        5. Overall strike rate
        6. Recent strike rate
        7. Consistency (std dev of recent scores)
        """
        # Recent form
        recent_runs = (
            player_history["runs"][-5:]
            if len(player_history["runs"]) >= 5
            else player_history["runs"]
        )
        recent_form = np.mean(recent_runs) if recent_runs else 0

        # Career average
        career_average = (
            np.mean(player_history["runs"]) if player_history["runs"] else 0
        )

        # Venue performance
        venue_runs = (
            [
                r
                for r, v in zip(player_history["runs"], player_history["venues"])
                if v == venue
            ]
            if venue
            else []
        )
        venue_average = np.mean(venue_runs) if venue_runs else career_average

        # Opposition performance
        opposition_runs = (
            [
                r
                for r, t in zip(player_history["runs"], player_history["teams"])
                if t == opposition
            ]
            if opposition
            else []
        )
        opposition_average = (
            np.mean(opposition_runs) if opposition_runs else career_average
        )

        # Strike rate features
        overall_sr = (
            np.mean(player_history["strike_rate"])
            if player_history["strike_rate"]
            else 0
        )
        recent_sr = (
            np.mean(player_history["strike_rate"][-5:])
            if len(player_history["strike_rate"]) >= 5
            else overall_sr
        )

        # Consistency (lower std dev means more consistent)
        recent_consistency = np.std(recent_runs) if len(recent_runs) > 1 else 0

        return [
            recent_form,
            career_average,
            venue_average,
            opposition_average,
            overall_sr,
            recent_sr,
            recent_consistency,
        ]

    def train(self, training_data: Dict[str, Dict]) -> None:
        """Train the linear regression model using historical data."""
        X_train = []
        y_train = []

        for player, history in training_data.items():
            if not history["runs"]:
                continue

            # For each match except the last one
            for i in range(1, len(history["runs"])):
                # Create a temporary history up to this point
                temp_history = {
                    "runs": history["runs"][:i],
                    "venues": history["venues"][:i],
                    "teams": history["teams"][:i],
                    "strike_rate": history["strike_rate"][:i],
                }

                features = self._extract_features(
                    temp_history,
                    venue=history["venues"][i],
                    opposition=history["teams"][i],
                )

                X_train.append(features)
                y_train.append(history["runs"][i])

        if X_train and y_train:
            self.model.fit(X_train, y_train)
            self.is_trained = True

    def predict(
        self, player_history: Dict, venue: str = None, opposition: str = None
    ) -> float:
        """Generate predictions using the trained linear regression model."""
        if not self.is_trained:
            # Fallback to simple average if model isn't trained
            return np.mean(player_history["runs"]) if player_history["runs"] else 0

        features = self._extract_features(player_history, venue, opposition)
        prediction = self.model.predict([features])[0]

        # Ensure prediction is non-negative
        return max(0, prediction)
