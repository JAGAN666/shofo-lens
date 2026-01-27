"""
Topic Clustering with BERTopic

Automatically discovers content themes in the dataset using
neural topic modeling.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger
import json

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logger.warning("BERTopic not available. Topic clustering will use fallback.")


@dataclass
class Topic:
    """A discovered topic."""
    id: int
    name: str
    keywords: List[str]
    size: int  # Number of documents
    representative_docs: List[str]
    coherence_score: float


@dataclass
class TopicAnalysis:
    """Complete topic analysis result."""
    topics: List[Topic]
    topic_distribution: Dict[int, int]
    total_documents: int
    outliers: int  # Documents not assigned to any topic


class TopicModeler:
    """
    Topic modeling using BERTopic for automatic theme discovery.

    Falls back to keyword-based clustering if BERTopic is not available.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.topic_model: Optional[BERTopic] = None
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self.is_fitted = False

        if BERTOPIC_AVAILABLE:
            logger.info("Initializing BERTopic...")
            try:
                # Use sentence transformers for embeddings
                sentence_model = SentenceTransformer(embedding_model)
                self.topic_model = BERTopic(
                    embedding_model=sentence_model,
                    language="english",
                    calculate_probabilities=True,
                    verbose=False,
                    nr_topics="auto",
                    min_topic_size=10,
                )
                logger.info("BERTopic initialized")
            except Exception as e:
                logger.warning(f"Could not initialize BERTopic: {e}")
                self.topic_model = None

    def fit(self, videos: List[Dict[str, Any]]) -> TopicAnalysis:
        """
        Fit the topic model on video content.

        Args:
            videos: List of video data dictionaries

        Returns:
            TopicAnalysis with discovered topics
        """
        # Build documents from video content
        self.documents = []
        for video in videos:
            doc = self._build_document(video)
            if doc and len(doc) > 20:
                self.documents.append(doc)

        if len(self.documents) < 50:
            logger.warning(f"Only {len(self.documents)} documents, using fallback")
            return self._fallback_analysis(videos)

        if self.topic_model is None:
            return self._fallback_analysis(videos)

        try:
            logger.info(f"Fitting BERTopic on {len(self.documents)} documents")
            topics, probs = self.topic_model.fit_transform(self.documents)
            self.is_fitted = True

            return self._build_analysis(topics, probs)

        except Exception as e:
            logger.error(f"BERTopic fitting failed: {e}")
            return self._fallback_analysis(videos)

    def get_topic_for_video(self, video_data: Dict[str, Any]) -> Tuple[int, float]:
        """Get the most likely topic for a video."""
        if not self.is_fitted or self.topic_model is None:
            return -1, 0.0

        doc = self._build_document(video_data)
        if not doc:
            return -1, 0.0

        try:
            topics, probs = self.topic_model.transform([doc])
            return int(topics[0]), float(probs[0].max()) if len(probs) > 0 else 0.0
        except:
            return -1, 0.0

    def get_similar_topics(self, topic_id: int, n: int = 5) -> List[int]:
        """Get topics similar to a given topic."""
        if not self.is_fitted or self.topic_model is None:
            return []

        try:
            similar = self.topic_model.find_topics(
                self.topic_model.get_topic(topic_id)[0][0],  # Use top keyword
                top_n=n + 1
            )
            return [t for t in similar[0] if t != topic_id][:n]
        except:
            return []

    def _build_document(self, video_data: Dict[str, Any]) -> str:
        """Build a text document from video data."""
        parts = []

        description = video_data.get("description", "") or ""
        if description:
            parts.append(description[:500])

        transcript = video_data.get("transcript", "") or ""
        if transcript:
            parts.append(transcript[:1000])

        hashtags = self._parse_json(video_data.get("hashtags", []))
        if hashtags and isinstance(hashtags, list):
            parts.append(" ".join(str(h) for h in hashtags[:15]))

        return " ".join(parts)

    def _build_analysis(self, topics: List[int], probs: np.ndarray) -> TopicAnalysis:
        """Build topic analysis from BERTopic results."""
        topic_info = self.topic_model.get_topic_info()
        topic_list = []

        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:  # Skip outlier topic
                continue

            # Get topic keywords
            topic_words = self.topic_model.get_topic(topic_id)
            keywords = [word for word, _ in topic_words[:10]]

            # Get representative documents
            try:
                rep_docs = self.topic_model.get_representative_docs(topic_id)[:3]
            except:
                rep_docs = []

            # Create topic name from top keywords
            name = " | ".join(keywords[:3]).title()

            topic_list.append(Topic(
                id=topic_id,
                name=name,
                keywords=keywords,
                size=row["Count"],
                representative_docs=[doc[:200] for doc in rep_docs],
                coherence_score=0.0,  # Would need additional calculation
            ))

        # Calculate distribution
        topic_distribution = {}
        for t in topics:
            topic_distribution[t] = topic_distribution.get(t, 0) + 1

        outliers = topic_distribution.get(-1, 0)

        return TopicAnalysis(
            topics=topic_list,
            topic_distribution=topic_distribution,
            total_documents=len(self.documents),
            outliers=outliers,
        )

    def _fallback_analysis(self, videos: List[Dict[str, Any]]) -> TopicAnalysis:
        """Fallback topic analysis using keyword clustering."""
        # Define topic keywords
        topic_definitions = {
            0: ("Entertainment & Comedy", ["funny", "comedy", "meme", "viral", "lol"]),
            1: ("Dance & Music", ["dance", "music", "song", "dj", "remix", "beat"]),
            2: ("Food & Cooking", ["food", "recipe", "cooking", "eat", "delicious"]),
            3: ("Beauty & Fashion", ["beauty", "makeup", "fashion", "style", "outfit"]),
            4: ("Fitness & Health", ["fitness", "workout", "gym", "health", "exercise"]),
            5: ("Education & Tips", ["learn", "tips", "how", "tutorial", "hack"]),
            6: ("Lifestyle & Vlog", ["life", "vlog", "day", "routine", "story"]),
            7: ("Pets & Animals", ["pet", "dog", "cat", "animal", "cute"]),
            8: ("Gaming", ["game", "gaming", "gamer", "play", "stream"]),
            9: ("Travel", ["travel", "trip", "adventure", "explore", "vacation"]),
        }

        topic_distribution = {i: 0 for i in range(-1, 10)}
        topic_docs: Dict[int, List[str]] = {i: [] for i in range(10)}

        for video in videos:
            doc = self._build_document(video)
            if not doc:
                topic_distribution[-1] += 1
                continue

            doc_lower = doc.lower()

            # Find best matching topic
            best_topic = -1
            best_score = 0

            for topic_id, (_, keywords) in topic_definitions.items():
                score = sum(1 for kw in keywords if kw in doc_lower)
                if score > best_score:
                    best_score = score
                    best_topic = topic_id

            if best_score == 0:
                best_topic = -1

            topic_distribution[best_topic] += 1
            if best_topic >= 0 and len(topic_docs[best_topic]) < 3:
                topic_docs[best_topic].append(doc[:200])

        # Build topics
        topic_list = []
        for topic_id, (name, keywords) in topic_definitions.items():
            if topic_distribution[topic_id] > 0:
                topic_list.append(Topic(
                    id=topic_id,
                    name=name,
                    keywords=keywords,
                    size=topic_distribution[topic_id],
                    representative_docs=topic_docs[topic_id],
                    coherence_score=0.5,
                ))

        return TopicAnalysis(
            topics=topic_list,
            topic_distribution=topic_distribution,
            total_documents=len(videos),
            outliers=topic_distribution[-1],
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


# Singleton instance
_topic_modeler: Optional[TopicModeler] = None


def get_topic_modeler() -> TopicModeler:
    """Get or create topic modeler instance."""
    global _topic_modeler
    if _topic_modeler is None:
        _topic_modeler = TopicModeler()
    return _topic_modeler
