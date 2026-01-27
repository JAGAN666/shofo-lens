import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from loguru import logger
from datetime import datetime
import random

# Optional ML dependencies
try:
    import xgboost as xgb
    import joblib
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
    logger.warning("xgboost/joblib not available, using demo mode")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    logger.warning("shap not available")


class EngagementPredictor:
    """Predictor for video engagement metrics using XGBoost."""

    FEATURE_COLUMNS = [
        "duration_ms",
        "hashtag_count",
        "description_length",
        "transcript_length",
        "has_transcript",
        "is_ai_generated",
        "is_ad",
        "hour_posted",
        "day_of_week",
        "sticker_count",
        "comment_count_sample",
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.explainer = None
        self.model_path = Path(model_path) if model_path else Path("models/engagement_model.joblib")
        self.demo_mode = not XGBOOST_AVAILABLE

        if not self.demo_mode and self.model_path.exists():
            self.load_model()

        if self.demo_mode:
            logger.info("EngagementPredictor running in demo mode")

    def extract_features(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from a single video's data."""
        # Parse engagement metrics if nested
        engagement = video_data.get("engagement_metrics", {})
        if isinstance(engagement, str):
            import json
            try:
                engagement = json.loads(engagement)
            except:
                engagement = {}

        # Parse date
        date_posted = video_data.get("date_posted")
        hour_posted = 12
        day_of_week = 0
        if date_posted:
            try:
                if isinstance(date_posted, str):
                    dt = datetime.fromisoformat(date_posted.replace("Z", "+00:00"))
                elif isinstance(date_posted, (int, float)):
                    dt = datetime.fromtimestamp(date_posted)
                else:
                    dt = date_posted
                hour_posted = dt.hour
                day_of_week = dt.weekday()
            except:
                pass

        # Parse hashtags
        hashtags = video_data.get("hashtags", [])
        if isinstance(hashtags, str):
            import json
            try:
                hashtags = json.loads(hashtags)
            except:
                hashtags = []
        hashtag_count = len(hashtags) if isinstance(hashtags, list) else 0

        # Parse stickers
        stickers = video_data.get("sticker_text", [])
        if isinstance(stickers, str):
            import json
            try:
                stickers = json.loads(stickers)
            except:
                stickers = []
        sticker_count = len(stickers) if isinstance(stickers, list) else 0

        # Parse comments
        comments = video_data.get("comments", [])
        if isinstance(comments, str):
            import json
            try:
                comments = json.loads(comments)
            except:
                comments = []
        comment_count_sample = len(comments) if isinstance(comments, list) else 0

        features = {
            "duration_ms": video_data.get("duration_ms", 0) or 0,
            "hashtag_count": hashtag_count,
            "description_length": len(video_data.get("description", "") or ""),
            "transcript_length": len(video_data.get("transcript", "") or ""),
            "has_transcript": 1 if video_data.get("transcript") else 0,
            "is_ai_generated": 1 if video_data.get("is_ai_generated") else 0,
            "is_ad": 1 if video_data.get("is_ad") else 0,
            "hour_posted": hour_posted,
            "day_of_week": day_of_week,
            "sticker_count": sticker_count,
            "comment_count_sample": comment_count_sample,
        }

        return features

    def extract_target(self, video_data: Dict[str, Any]) -> float:
        """Extract target variable (engagement score) from video data."""
        engagement = video_data.get("engagement_metrics", {})
        if isinstance(engagement, str):
            import json
            try:
                engagement = json.loads(engagement)
            except:
                engagement = {}

        play_count = engagement.get("play_count", 0) or 0
        like_count = engagement.get("like_count", 0) or 0
        share_count = engagement.get("share_count", 0) or 0
        comment_count = engagement.get("comment_count", 0) or 0

        # Engagement score: weighted combination normalized by views
        if play_count > 0:
            engagement_rate = (
                (like_count * 1.0 + share_count * 2.0 + comment_count * 1.5) / play_count
            )
        else:
            engagement_rate = 0

        # Log transform for better distribution
        return np.log1p(engagement_rate * 1000)

    def prepare_training_data(
        self, videos: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data from video records."""
        features_list = []
        targets = []

        for video in videos:
            features = self.extract_features(video)
            target = self.extract_target(video)
            features_list.append(features)
            targets.append(target)

        X = pd.DataFrame(features_list)
        y = np.array(targets)

        return X, y

    def train(
        self,
        videos: List[Dict[str, Any]],
        test_size: float = 0.2,
    ) -> Dict[str, float]:
        """Train the engagement prediction model."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        logger.info(f"Preparing training data from {len(videos)} videos")
        X, y = self.prepare_training_data(videos)

        # Remove any NaN values
        mask = ~(X.isna().any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        logger.info(f"Training data shape: {X.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

        logger.info("Training XGBoost model...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }

        logger.info(f"Model metrics: {metrics}")

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Save model
        self.save_model()

        return metrics

    def predict(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict engagement for a single video."""
        features = self.extract_features(video_data)

        # Demo mode: generate realistic predictions
        if self.demo_mode or self.model is None:
            # Generate score based on features
            base_score = 0.05
            if features.get("has_transcript"):
                base_score += 0.02
            if features.get("hashtag_count", 0) > 3:
                base_score += 0.01
            if 10000 < features.get("duration_ms", 0) < 60000:
                base_score += 0.015
            # Add some randomness
            engagement_score = base_score + random.uniform(-0.01, 0.03)

            return {
                "engagement_score": max(0.01, min(0.15, engagement_score)),
                "raw_prediction": np.log1p(engagement_score * 1000),
                "feature_importance": {col: random.uniform(-0.1, 0.1) for col in self.FEATURE_COLUMNS},
                "features_used": features,
            }

        X = pd.DataFrame([features])
        prediction = float(self.model.predict(X)[0])

        # Get SHAP explanation
        explanation = {}
        if self.explainer and SHAP_AVAILABLE:
            shap_values = self.explainer.shap_values(X)
            for i, col in enumerate(self.FEATURE_COLUMNS):
                if i < len(shap_values[0]):
                    explanation[col] = float(shap_values[0][i])

        # Convert back from log scale
        engagement_score = np.expm1(prediction) / 1000

        return {
            "engagement_score": engagement_score,
            "raw_prediction": prediction,
            "feature_importance": explanation,
            "features_used": features,
        }

    def batch_predict(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict engagement for multiple videos."""
        return [self.predict(video) for video in videos]

    def save_model(self):
        """Save the trained model."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.FEATURE_COLUMNS,
            },
            self.model_path,
        )
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load a trained model."""
        if self.model_path.exists():
            data = joblib.load(self.model_path)
            self.model = data["model"]
            if self.model:
                self.explainer = shap.TreeExplainer(self.model)
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.warning(f"No model found at {self.model_path}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if self.model is None:
            return {}

        importance = self.model.feature_importances_
        return {
            col: float(imp)
            for col, imp in zip(self.FEATURE_COLUMNS, importance)
        }
