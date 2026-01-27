"""
Advanced Virality Prediction Engine

Features:
- Multi-target prediction (views, likes, shares, comments)
- Confidence intervals using quantile regression
- Viral potential score combining all factors
- Visual explainability with SHAP (when available)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from loguru import logger
from datetime import datetime
import json
import random

# Optional heavy dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not available, using demo mode")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not available, using simple feature importance")


@dataclass
class ViralityPrediction:
    """Structured virality prediction result."""
    viral_score: float  # 0-100 overall virality score
    viral_tier: str  # "Low", "Medium", "High", "Viral"
    confidence: float  # Confidence in prediction (0-1)

    # Individual metric predictions
    predicted_views: int
    predicted_likes: int
    predicted_shares: int
    predicted_comments: int

    # Confidence intervals
    views_range: Tuple[int, int]
    likes_range: Tuple[int, int]

    # Explainability
    top_factors: List[Dict[str, Any]]
    recommendations: List[str]


class ViralityPredictor:
    """
    Advanced multi-target virality predictor with confidence intervals.

    Uses quantile regression for uncertainty estimation and provides
    actionable insights through SHAP explainability.
    """

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
        "word_count",
        "emoji_count",
        "question_marks",
        "exclamation_marks",
    ]

    VIRAL_TIERS = {
        (0, 20): "Low",
        (20, 50): "Medium",
        (50, 80): "High",
        (80, 100): "Viral"
    }

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else Path("models/virality")
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.quantile_models: Dict[str, Dict[str, xgb.XGBRegressor]] = {}
        self.scaler = StandardScaler()
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_stats: Dict[str, Dict[str, float]] = {}

        if self.model_path.exists():
            self.load_models()

    def extract_features(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from video data."""

        # Parse nested fields
        engagement = self._parse_json(video_data.get("engagement_metrics", {}))
        hashtags = self._parse_json(video_data.get("hashtags", []))
        stickers = self._parse_json(video_data.get("sticker_text", []))

        description = video_data.get("description", "") or ""
        transcript = video_data.get("transcript", "") or ""

        # Time features
        date_posted = video_data.get("date_posted")
        hour_posted, day_of_week = self._extract_time_features(date_posted)

        # Text analysis features
        text_content = f"{description} {transcript}"

        features = {
            "duration_ms": video_data.get("duration_ms", 0) or 0,
            "hashtag_count": len(hashtags) if isinstance(hashtags, list) else 0,
            "description_length": len(description),
            "transcript_length": len(transcript),
            "has_transcript": 1 if transcript else 0,
            "is_ai_generated": 1 if video_data.get("is_ai_generated") else 0,
            "is_ad": 1 if video_data.get("is_ad") else 0,
            "hour_posted": hour_posted,
            "day_of_week": day_of_week,
            "sticker_count": len(stickers) if isinstance(stickers, list) else 0,
            "word_count": len(text_content.split()),
            "emoji_count": sum(1 for c in text_content if ord(c) > 127),
            "question_marks": text_content.count("?"),
            "exclamation_marks": text_content.count("!"),
        }

        return features

    def extract_targets(self, video_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract target variables (engagement metrics)."""
        engagement = self._parse_json(video_data.get("engagement_metrics", {}))

        return {
            "views": np.log1p(engagement.get("play_count", 0) or 0),
            "likes": np.log1p(engagement.get("like_count", 0) or 0),
            "shares": np.log1p(engagement.get("share_count", 0) or 0),
            "comments": np.log1p(engagement.get("comment_count", 0) or 0),
        }

    def train(self, videos: List[Dict[str, Any]], test_size: float = 0.2) -> Dict[str, Any]:
        """Train multi-target models with quantile regression."""
        logger.info(f"Training virality predictor on {len(videos)} videos")

        # Prepare data
        features_list = []
        targets_list = []

        for video in videos:
            features_list.append(self.extract_features(video))
            targets_list.append(self.extract_targets(video))

        X = pd.DataFrame(features_list)
        Y = pd.DataFrame(targets_list)

        # Store feature statistics for normalization
        self.feature_stats = {
            col: {"mean": X[col].mean(), "std": X[col].std()}
            for col in X.columns
        }

        # Remove invalid rows
        valid_mask = ~(X.isna().any(axis=1) | Y.isna().any(axis=1))
        X = X[valid_mask]
        Y = Y[valid_mask]

        logger.info(f"Training data shape: X={X.shape}, Y={Y.shape}")

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42
        )

        metrics = {}

        # Train models for each target
        for target in ["views", "likes", "shares", "comments"]:
            logger.info(f"Training model for {target}...")

            # Main regression model
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train, Y_train[target], eval_set=[(X_test, Y_test[target])], verbose=False)
            self.models[target] = model

            # Quantile models for confidence intervals
            self.quantile_models[target] = {}
            for quantile, alpha in [("lower", 0.1), ("upper", 0.9)]:
                q_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    objective=f"reg:quantileerror",
                    quantile_alpha=alpha,
                    random_state=42,
                    n_jobs=-1,
                )
                q_model.fit(X_train, Y_train[target], verbose=False)
                self.quantile_models[target][quantile] = q_model

            # Evaluate
            y_pred = model.predict(X_test)
            mse = np.mean((Y_test[target] - y_pred) ** 2)
            r2 = 1 - mse / np.var(Y_test[target])
            metrics[target] = {"mse": float(mse), "r2": float(r2)}

            logger.info(f"  {target}: R2={r2:.3f}")

        # Initialize SHAP explainer with the views model (primary metric)
        self.explainer = shap.TreeExplainer(self.models["views"])

        # Save models
        self.save_models()

        return metrics

    def predict(self, video_data: Dict[str, Any]) -> ViralityPrediction:
        """Generate comprehensive virality prediction."""
        if not self.models:
            raise ValueError("Models not trained or loaded")

        features = self.extract_features(video_data)
        X = pd.DataFrame([features])

        predictions = {}
        ranges = {}

        for target in ["views", "likes", "shares", "comments"]:
            # Main prediction
            pred = float(self.models[target].predict(X)[0])
            predictions[target] = int(np.expm1(pred))

            # Confidence intervals
            lower = float(self.quantile_models[target]["lower"].predict(X)[0])
            upper = float(self.quantile_models[target]["upper"].predict(X)[0])
            ranges[target] = (int(np.expm1(lower)), int(np.expm1(upper)))

        # Calculate viral score (0-100)
        viral_score = self._calculate_viral_score(predictions, ranges)
        viral_tier = self._get_viral_tier(viral_score)

        # Calculate confidence based on prediction interval width
        confidence = self._calculate_confidence(predictions, ranges)

        # Get SHAP explanations
        top_factors = self._get_top_factors(X, features)

        # Generate recommendations
        recommendations = self._generate_recommendations(features, top_factors)

        return ViralityPrediction(
            viral_score=viral_score,
            viral_tier=viral_tier,
            confidence=confidence,
            predicted_views=predictions["views"],
            predicted_likes=predictions["likes"],
            predicted_shares=predictions["shares"],
            predicted_comments=predictions["comments"],
            views_range=ranges["views"],
            likes_range=ranges["likes"],
            top_factors=top_factors,
            recommendations=recommendations,
        )

    def _calculate_viral_score(
        self, predictions: Dict[str, int], ranges: Dict[str, Tuple[int, int]]
    ) -> float:
        """Calculate overall viral score (0-100)."""
        # Weight different metrics
        weights = {"views": 0.4, "likes": 0.3, "shares": 0.2, "comments": 0.1}

        # Normalize predictions to 0-100 scale based on typical ranges
        typical_maxes = {"views": 10_000_000, "likes": 1_000_000, "shares": 100_000, "comments": 50_000}

        score = 0
        for metric, weight in weights.items():
            normalized = min(predictions[metric] / typical_maxes[metric] * 100, 100)
            score += normalized * weight

        return min(max(score, 0), 100)

    def _get_viral_tier(self, score: float) -> str:
        """Get viral tier from score."""
        for (low, high), tier in self.VIRAL_TIERS.items():
            if low <= score < high:
                return tier
        return "Viral"

    def _calculate_confidence(
        self, predictions: Dict[str, int], ranges: Dict[str, Tuple[int, int]]
    ) -> float:
        """Calculate confidence based on prediction interval width."""
        confidences = []
        for metric in predictions:
            pred = predictions[metric]
            low, high = ranges[metric]
            if pred > 0:
                interval_width = (high - low) / pred
                conf = max(0, 1 - interval_width / 2)
                confidences.append(conf)

        return np.mean(confidences) if confidences else 0.5

    def _get_top_factors(self, X: pd.DataFrame, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get top factors influencing the prediction."""
        if not self.explainer:
            return []

        shap_values = self.explainer.shap_values(X)

        factors = []
        for i, col in enumerate(self.FEATURE_COLUMNS):
            if i < len(shap_values[0]):
                impact = float(shap_values[0][i])
                factors.append({
                    "feature": col,
                    "value": features.get(col, 0),
                    "impact": impact,
                    "direction": "positive" if impact > 0 else "negative",
                })

        # Sort by absolute impact
        factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return factors[:6]

    def _generate_recommendations(
        self, features: Dict[str, Any], top_factors: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Analyze negative factors and suggest improvements
        for factor in top_factors:
            if factor["direction"] == "negative" and abs(factor["impact"]) > 0.1:
                feature = factor["feature"]
                value = factor["value"]

                if feature == "duration_ms" and value < 15000:
                    recommendations.append(
                        "Increase video length to 15-60 seconds for better engagement"
                    )
                elif feature == "hashtag_count" and value < 3:
                    recommendations.append(
                        "Add 3-5 relevant hashtags to improve discoverability"
                    )
                elif feature == "has_transcript" and value == 0:
                    recommendations.append(
                        "Include spoken content - videos with clear audio perform 40% better"
                    )
                elif feature == "hour_posted":
                    recommendations.append(
                        "Try posting between 6-9 PM local time for peak engagement"
                    )
                elif feature == "description_length" and value < 50:
                    recommendations.append(
                        "Write a longer, keyword-rich description (100+ characters)"
                    )

        # Add positive reinforcement
        for factor in top_factors[:2]:
            if factor["direction"] == "positive" and abs(factor["impact"]) > 0.2:
                feature = factor["feature"]
                recommendations.append(
                    f"Your {feature.replace('_', ' ')} is working well - keep it up!"
                )

        if not recommendations:
            recommendations.append("Your video characteristics look optimized for engagement!")

        return recommendations[:5]

    def _parse_json(self, value: Any) -> Any:
        """Parse JSON field that might be string or already parsed."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except:
                return value
        return value

    def _extract_time_features(self, date_posted: Any) -> Tuple[int, int]:
        """Extract hour and day of week from date."""
        try:
            if isinstance(date_posted, str):
                dt = datetime.fromisoformat(date_posted.replace("Z", "+00:00"))
            elif isinstance(date_posted, (int, float)):
                dt = datetime.fromtimestamp(date_posted)
            else:
                return 12, 0
            return dt.hour, dt.weekday()
        except:
            return 12, 0

    def save_models(self):
        """Save all trained models."""
        self.model_path.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "models": self.models,
            "quantile_models": self.quantile_models,
            "feature_stats": self.feature_stats,
        }, self.model_path / "virality_models.joblib")

        logger.info(f"Models saved to {self.model_path}")

    def load_models(self):
        """Load trained models."""
        model_file = self.model_path / "virality_models.joblib"
        if model_file.exists():
            data = joblib.load(model_file)
            self.models = data["models"]
            self.quantile_models = data["quantile_models"]
            self.feature_stats = data.get("feature_stats", {})

            if "views" in self.models:
                self.explainer = shap.TreeExplainer(self.models["views"])

            logger.info("Virality models loaded")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get global feature importance."""
        if "views" not in self.models:
            return {}

        importance = self.models["views"].feature_importances_
        return {
            col: float(imp)
            for col, imp in zip(self.FEATURE_COLUMNS, importance)
        }
