# app.py
import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple
import importlib.util
import sys


class CricketDataProcessor:
    def __init__(self):
        self.player_stats = {}
        self.fantasy_weights = {
            "runs": 1,
            "fours": 1,
            "sixes": 2,
            "wickets": 25,
            "catches": 8,
            "stumpings": 12,
            "run_outs": 12,
        }

    def process_match_data(self, match_data: Dict) -> None:
        """Process individual match data and update player statistics."""
        match_date = datetime.strptime(match_data["info"]["dates"][0], "%Y-%m-%d")
        venue = match_data["info"].get("venue", "Unknown")

        for innings in match_data.get("innings", []):
            team = innings["team"]
            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    self._process_delivery(delivery, match_date, venue, team)

    def _process_delivery(
        self, delivery: Dict, match_date: datetime, venue: str, team: str
    ) -> None:
        """Process individual delivery data."""
        batter = delivery["batter"]
        runs = delivery["runs"]["batter"]
        extras = delivery["runs"].get("extras", 0)

        # Initialize player if not exists
        if batter not in self.player_stats:
            self.player_stats[batter] = self._initialize_player_stats()

        # Update batting statistics
        self._update_batting_stats(batter, match_date, venue, team, runs, extras)

        # Process wicket if exists
        if "wickets" in delivery:
            bowler = delivery["bowler"]
            if bowler not in self.player_stats:
                self.player_stats[bowler] = self._initialize_player_stats()
            self._update_bowling_stats(bowler, match_date, venue, team)

    def _initialize_player_stats(self) -> Dict:
        """Initialize player statistics dictionary."""
        return {
            "matches": [],
            "runs": [],
            "wickets": [],
            "dates": [],
            "venues": [],
            "teams": [],
            "batting_avg": [],
            "bowling_avg": [],
            "strike_rate": [],
        }

    def _update_batting_stats(
        self, player: str, date: datetime, venue: str, team: str, runs: int, extras: int
    ) -> None:
        """Update batting statistics for a player."""
        stats = self.player_stats[player]

        if date not in stats["dates"]:
            stats["dates"].append(date)
            stats["venues"].append(venue)
            stats["teams"].append(team)
            stats["matches"].append(1)
            stats["runs"].append(runs)
        else:
            stats["runs"][-1] += runs

        # Update batting average
        total_runs = sum(stats["runs"])
        matches = len(set(stats["dates"]))
        stats["batting_avg"].append(total_runs / max(matches, 1))

        # Update strike rate
        deliveries = len(stats["runs"])
        stats["strike_rate"].append((total_runs * 100) / max(deliveries, 1))

    def _update_bowling_stats(
        self, player: str, date: datetime, venue: str, team: str
    ) -> None:
        """Update bowling statistics for a player."""
        stats = self.player_stats[player]

        if date not in stats["dates"]:
            stats["dates"].append(date)
            stats["venues"].append(venue)
            stats["teams"].append(team)
            stats["wickets"].append(1)
            stats["runs"].append(0)

        else:
            if len(stats["wickets"]) == 0:
                stats["wickets"].append(0)
            stats["wickets"][-1] += 1

        # Update bowling average
        total_wickets = sum(stats["wickets"])
        total_runs_conceded = sum(stats.get("runs_conceded", [0]))

        stats["bowling_avg"].append(total_runs_conceded / max(total_wickets, 1))


class PredictionModel:
    def __init__(self):
        """Initialize prediction model with configurable weights."""
        self.weights = {
            "recent_form": 0.5,
            "career_average": 0.5,
            "venue_performance": 0.5,
            "opposition_factor": 0.5,
        }

    def predict(
        self, player_history: Dict, venue: str = None, opposition: str = None
    ) -> float:
        """Generate predictions using player history and context."""
        # Recent form (last 5 matches)
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

        # Opposition factor
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

        # Weighted prediction
        prediction = (
            recent_form * self.weights["recent_form"]
            + career_average * self.weights["career_average"]
            + venue_average * self.weights["venue_performance"]
            + opposition_average * self.weights["opposition_factor"]
        )

        return prediction


def load_custom_prediction_model(file_path: str):
    """Dynamically load a PredictionModel class from a user-provided file."""
    spec = importlib.util.spec_from_file_location("custom_model", file_path)
    custom_model_module = importlib.util.module_from_spec(spec)
    sys.modules["custom_model"] = custom_model_module
    spec.loader.exec_module(custom_model_module)
    return custom_model_module.PredictionModel()


class TestingFramework:
    def __init__(self, prediction_model):
        self.train_processor = CricketDataProcessor()
        self.test_processor = CricketDataProcessor()
        self.model = prediction_model
        self.metrics = {}

    def load_training_data(
        self, data_dir: str, start_date: datetime, end_date: datetime
    ) -> None:
        """Load and process training data within the specified date range."""
        self._load_data_within_date_range(
            data_dir, start_date, end_date, is_training=True
        )

    def load_testing_data(
        self, data_dir: str, start_date: datetime, end_date: datetime
    ) -> None:
        """Load and process testing data within the specified date range."""
        self._load_data_within_date_range(
            data_dir, start_date, end_date, is_training=False
        )

    def _load_data_within_date_range(
        self,
        directory: str,
        start_date: datetime,
        end_date: datetime,
        is_training: bool,
    ) -> None:
        """Load and process data files from the specified directory within the date range."""
        data_path = Path(directory)
        for json_file in data_path.glob("*.json"):
            with open(json_file) as f:
                match_data = json.load(f)
                match_date_str = match_data["info"]["dates"][0]
                match_date = datetime.strptime(match_date_str, "%Y-%m-%d")
                if start_date <= match_date.date() <= end_date:
                    if is_training:
                        self.train_processor.process_match_data(match_data)
                    else:
                        # Process test data separately
                        self.test_processor.process_match_data(match_data)

        print("Loaded:")
        print("num players in train:", len(self.train_processor.player_stats))
        print("num players in test:", len(self.test_processor.player_stats))

    def evaluate(self) -> Dict:
        """Evaluate model performance."""
        predictions = []
        actuals = []
        player_predictions = []

        for player in self.train_processor.player_stats:
            if (
                player in self.test_processor.player_stats
                and self.test_processor.player_stats[player]["runs"]
            ):
                train_history = self.train_processor.player_stats[player]
                assert len(train_history["runs"]) == len(
                    train_history["venues"]
                ), "Error: Length of 'runs' and 'venues' should match."
                assert len(train_history["runs"]) == len(
                    train_history["teams"]
                ), "Error: Length of 'runs' and 'teams' should match."
                assert len(train_history["runs"]) == len(
                    train_history["dates"]
                ), "Error: Length of 'runs' and 'dates' should match."
                prediction = self.model.predict(
                    train_history,
                    venue=self.test_processor.player_stats[player]["venues"][-1],
                    opposition=self.test_processor.player_stats[player]["teams"][-1],
                )
                actual = np.mean(self.test_processor.player_stats[player]["runs"])

                predictions.append(prediction)
                actuals.append(actual)
                player_predictions.append(
                    {
                        "player": player,
                        "predicted": prediction,
                        "actual": actual,
                        "error": abs(prediction - actual),
                    }
                )

        self.metrics = {
            "mae": mean_absolute_error(actuals, predictions),
            "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
            "r2": r2_score(actuals, predictions),
            "predictions": player_predictions,
        }

        return self.metrics


def main():

    st.title("Dream11 Model Testing Dashboard")

    st.sidebar.header("Data Configuration")
    train_start_date = st.sidebar.date_input(
        "Training Data Start Date", datetime(2000, 1, 1)
    )
    train_end_date = st.sidebar.date_input(
        "Training Data End Date", datetime(2004, 12, 30)
    )
    st.sidebar.write("---")
    test_start_date = st.sidebar.date_input(
        "Testing Data Start Date", datetime(2005, 1, 1)
    )
    test_end_date = st.sidebar.date_input(
        "Testing Data End Date", datetime(2023, 12, 30)
    )
    st.sidebar.write("---")
    custom_model_file = st.sidebar.file_uploader(
        "Upload Custom Prediction Model File", type="py", key="model_uploader"
    )
    if custom_model_file is not None:
        try:
            custom_model_path = f"{custom_model_file.name}"
            with open(custom_model_path, "wb") as f:
                f.write(custom_model_file.getbuffer())
            st.session_state.model_path = custom_model_path
            st.sidebar.success("Model uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error uploading model: {str(e)}")
            st.session_state.model_path = None

    data_dir = "data"
    if "run_analysis" not in st.session_state:
        st.session_state["run_analysis"] = False
    run = st.sidebar.button("Run Analysis", key="run_button")
    if run:
        st.session_state["run_analysis"] = True  # Set the state when clicked

    if st.session_state["run_analysis"]:
        print(st.session_state)
        if "model_path" in st.session_state and st.session_state.model_path:
            with st.spinner("Loading custom model..."):
                prediction_model = load_custom_prediction_model(
                    st.session_state.model_path
                )
                print("Custom Model Loaded")
        else:
            prediction_model = PredictionModel()

        framework = TestingFramework(prediction_model)

        with st.spinner("Loading training data..."):
            framework.load_training_data(data_dir, train_start_date, train_end_date)
        with st.spinner("Loading testing data..."):
            framework.load_testing_data(data_dir, test_start_date, test_end_date)
        metrics = None
        with st.spinner("Evaluating model..."):
            metrics = framework.evaluate()

        st.header("Model Performance Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MAE", f"{metrics['mae']:.2f}")
        with col2:
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
        with col3:
            st.metric("R² Score", f"{metrics['r2']:.2f}")

        # Display detailed predictions
        st.header("Player Predictions")
        predictions_df = pd.DataFrame(metrics["predictions"])
        predictions_df = predictions_df.sort_values("error", ascending=True)
        st.dataframe(predictions_df)

        # Visualization
        st.header("Actual vs Predicted Scores")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            [p["predicted"] for p in metrics["predictions"]],
            [p["actual"] for p in metrics["predictions"]],
            alpha=0.5,
        )

        min_val = min(
            min([p["predicted"] for p in metrics["predictions"]]),
            min([p["actual"] for p in metrics["predictions"]]),
        )
        max_val = max(
            max([p["predicted"] for p in metrics["predictions"]]),
            max([p["actual"] for p in metrics["predictions"]]),
        )
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            label="Perfect Prediction",
        )

        ax.set_xlabel("Predicted Scores")
        ax.set_ylabel("Actual Scores")
        ax.legend()
        st.pyplot(fig)

        # Error Distribution
        st.header("Error Distribution")
        errors = [p["error"] for p in metrics["predictions"]]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=20)
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.write(" Step 1: Select Start and End Dates for training and testing period.")
        st.write("Step 2: Select a custom model to evaluate(optional). ")
        st.write(
            "> If you don't upload a custom model, a test model with example weights will be evaluated"
        )


if __name__ == "__main__":
    main()
