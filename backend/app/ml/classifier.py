"""
Zero-Shot Content Classifier

Automatically labels videos into categories without any training data.
Uses transformer models for multi-label classification.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
from transformers import pipeline
from loguru import logger
import json


@dataclass
class ContentLabel:
    """A single content label with confidence."""
    category: str
    confidence: float
    subcategory: Optional[str] = None


@dataclass
class ClassificationResult:
    """Complete classification result for a video."""
    primary_category: str
    primary_confidence: float
    all_labels: List[ContentLabel]
    content_type: str  # "Entertainment", "Educational", "Commercial"
    mood: str  # "Funny", "Serious", "Inspirational", etc.


class ContentClassifier:
    """
    Zero-shot content classifier using transformer models.

    Classifies videos into categories based on transcript, description,
    and hashtags without requiring any training data.
    """

    # Main content categories
    CATEGORIES = [
        "Comedy & Entertainment",
        "Dance & Music",
        "Tutorial & How-To",
        "Product Review & Unboxing",
        "Lifestyle & Vlog",
        "Food & Cooking",
        "Fashion & Beauty",
        "Fitness & Health",
        "Gaming",
        "Pets & Animals",
        "Education & Learning",
        "News & Current Events",
        "Art & Creativity",
        "Travel & Adventure",
        "Technology & Gadgets",
    ]

    # Content types
    CONTENT_TYPES = [
        "Entertainment",
        "Educational",
        "Commercial/Promotional",
        "Personal/Authentic",
    ]

    # Mood categories
    MOODS = [
        "Funny & Humorous",
        "Serious & Informative",
        "Inspirational & Motivational",
        "Relaxing & Calm",
        "Exciting & Energetic",
        "Emotional & Touching",
    ]

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """Initialize the classifier with a zero-shot model."""
        logger.info(f"Loading zero-shot classifier: {model_name}")
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1,  # CPU, change to 0 for GPU
            )
            self.is_loaded = True
            logger.info("Zero-shot classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load classifier: {e}")
            self.classifier = None
            self.is_loaded = False

    def classify(self, video_data: Dict[str, Any]) -> ClassificationResult:
        """
        Classify a video based on its content.

        Args:
            video_data: Dict containing transcript, description, hashtags, etc.

        Returns:
            ClassificationResult with all labels and confidence scores
        """
        if not self.is_loaded:
            return self._fallback_classification(video_data)

        # Build text representation
        text = self._build_text_representation(video_data)

        if not text or len(text.strip()) < 10:
            return self._fallback_classification(video_data)

        # Classify main category
        category_result = self.classifier(
            text,
            candidate_labels=self.CATEGORIES,
            multi_label=True,
        )

        # Classify content type
        type_result = self.classifier(
            text,
            candidate_labels=self.CONTENT_TYPES,
            multi_label=False,
        )

        # Classify mood
        mood_result = self.classifier(
            text,
            candidate_labels=self.MOODS,
            multi_label=False,
        )

        # Build labels list
        all_labels = []
        for label, score in zip(category_result["labels"], category_result["scores"]):
            if score > 0.1:  # Only include labels with >10% confidence
                all_labels.append(ContentLabel(
                    category=label,
                    confidence=round(score, 3),
                ))

        # Get primary category
        primary_category = category_result["labels"][0]
        primary_confidence = category_result["scores"][0]

        return ClassificationResult(
            primary_category=primary_category,
            primary_confidence=round(primary_confidence, 3),
            all_labels=all_labels[:5],  # Top 5 labels
            content_type=type_result["labels"][0],
            mood=mood_result["labels"][0].replace(" & ", "/"),
        )

    def classify_batch(
        self, videos: List[Dict[str, Any]], batch_size: int = 8
    ) -> List[ClassificationResult]:
        """Classify multiple videos efficiently."""
        results = []
        for video in videos:
            results.append(self.classify(video))
        return results

    def get_category_distribution(
        self, videos: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Get distribution of categories across videos."""
        distribution = {cat: 0 for cat in self.CATEGORIES}

        for video in videos:
            result = self.classify(video)
            distribution[result.primary_category] += 1

        return distribution

    def _build_text_representation(self, video_data: Dict[str, Any]) -> str:
        """Build a comprehensive text representation of the video."""
        parts = []

        # Add description
        description = video_data.get("description", "") or ""
        if description:
            parts.append(f"Description: {description[:500]}")

        # Add transcript
        transcript = video_data.get("transcript", "") or ""
        if transcript:
            parts.append(f"Transcript: {transcript[:1000]}")

        # Add hashtags
        hashtags = self._parse_json(video_data.get("hashtags", []))
        if hashtags and isinstance(hashtags, list):
            parts.append(f"Hashtags: {' '.join(hashtags[:15])}")

        # Add sticker text
        stickers = self._parse_json(video_data.get("sticker_text", []))
        if stickers and isinstance(stickers, list):
            parts.append(f"Text: {' '.join(stickers[:10])}")

        return " | ".join(parts)

    def _fallback_classification(self, video_data: Dict[str, Any]) -> ClassificationResult:
        """Fallback classification based on hashtags and keywords."""
        hashtags = self._parse_json(video_data.get("hashtags", [])) or []
        description = (video_data.get("description", "") or "").lower()

        # Simple keyword matching for fallback
        keyword_map = {
            "Comedy & Entertainment": ["funny", "comedy", "joke", "lol", "meme"],
            "Dance & Music": ["dance", "music", "song", "choreography", "dj"],
            "Tutorial & How-To": ["tutorial", "howto", "diy", "learn", "tips"],
            "Food & Cooking": ["food", "recipe", "cooking", "foodie", "yummy"],
            "Fashion & Beauty": ["fashion", "beauty", "makeup", "style", "ootd"],
            "Fitness & Health": ["fitness", "workout", "gym", "health", "exercise"],
            "Gaming": ["gaming", "game", "gamer", "twitch", "esports"],
            "Pets & Animals": ["pet", "dog", "cat", "animal", "puppy", "kitten"],
        }

        all_text = " ".join(hashtags).lower() + " " + description

        scores = {}
        for category, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw in all_text)
            scores[category] = score / len(keywords)

        if max(scores.values()) > 0:
            primary = max(scores, key=scores.get)
            confidence = min(scores[primary] * 2, 0.9)
        else:
            primary = "Comedy & Entertainment"
            confidence = 0.3

        return ClassificationResult(
            primary_category=primary,
            primary_confidence=confidence,
            all_labels=[ContentLabel(primary, confidence)],
            content_type="Entertainment",
            mood="Funny/Humorous",
        )

    def _parse_json(self, value: Any) -> Any:
        """Parse JSON field."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except:
                return value
        return value


# Singleton instance for efficiency
_classifier_instance: Optional[ContentClassifier] = None


def get_classifier() -> ContentClassifier:
    """Get or create classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ContentClassifier()
    return _classifier_instance
