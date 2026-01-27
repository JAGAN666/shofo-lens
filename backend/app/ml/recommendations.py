"""
Similar Video Recommendations

Hybrid recommendation system combining:
- Content-based filtering (embedding similarity)
- Engagement-based filtering (similar performance)
- Explainable recommendations
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger
import json


@dataclass
class Recommendation:
    """A single video recommendation."""
    video_id: int
    similarity_score: float
    explanation: str
    shared_hashtags: List[str]
    engagement_similarity: float


@dataclass
class RecommendationResult:
    """Complete recommendation result."""
    source_video_id: int
    recommendations: List[Recommendation]
    recommendation_type: str  # "content", "engagement", "hybrid"


class RecommendationEngine:
    """
    Hybrid recommendation engine for similar videos.

    Combines:
    - Embedding similarity for content matching
    - Engagement pattern matching
    - Hashtag overlap
    """

    def __init__(self):
        self.video_embeddings: Dict[int, np.ndarray] = {}
        self.video_metadata: Dict[int, Dict[str, Any]] = {}
        self.hashtag_index: Dict[str, List[int]] = {}  # hashtag -> video_ids

    def index_videos(
        self,
        videos: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None
    ):
        """
        Index videos for recommendation.

        Args:
            videos: List of video data
            embeddings: Pre-computed embeddings (optional)
        """
        logger.info(f"Indexing {len(videos)} videos for recommendations")

        for i, video in enumerate(videos):
            video_id = video.get("video_id", i)

            # Store metadata
            self.video_metadata[video_id] = video

            # Store embedding if provided
            if embeddings is not None and i < len(embeddings):
                self.video_embeddings[video_id] = embeddings[i]

            # Build hashtag index
            hashtags = self._parse_json(video.get("hashtags", []))
            if isinstance(hashtags, list):
                for tag in hashtags[:20]:
                    tag = str(tag).lower()
                    if tag not in self.hashtag_index:
                        self.hashtag_index[tag] = []
                    self.hashtag_index[tag].append(video_id)

        logger.info(f"Indexed {len(self.video_metadata)} videos, {len(self.hashtag_index)} hashtags")

    def get_similar(
        self,
        video_id: int,
        n: int = 10,
        method: str = "hybrid"
    ) -> RecommendationResult:
        """
        Get similar videos.

        Args:
            video_id: Source video ID
            n: Number of recommendations
            method: "content", "engagement", or "hybrid"

        Returns:
            RecommendationResult with similar videos
        """
        if video_id not in self.video_metadata:
            return RecommendationResult(
                source_video_id=video_id,
                recommendations=[],
                recommendation_type=method,
            )

        source = self.video_metadata[video_id]

        if method == "content":
            recommendations = self._content_based(video_id, source, n)
        elif method == "engagement":
            recommendations = self._engagement_based(video_id, source, n)
        else:
            recommendations = self._hybrid(video_id, source, n)

        return RecommendationResult(
            source_video_id=video_id,
            recommendations=recommendations,
            recommendation_type=method,
        )

    def _content_based(
        self,
        video_id: int,
        source: Dict[str, Any],
        n: int
    ) -> List[Recommendation]:
        """Content-based recommendations using embeddings and hashtags."""
        candidates: Dict[int, float] = {}

        # Embedding similarity
        if video_id in self.video_embeddings:
            source_emb = self.video_embeddings[video_id]
            for vid, emb in self.video_embeddings.items():
                if vid != video_id:
                    similarity = np.dot(source_emb, emb) / (
                        np.linalg.norm(source_emb) * np.linalg.norm(emb) + 1e-8
                    )
                    candidates[vid] = float(similarity)

        # Hashtag overlap boost
        source_hashtags = set(
            str(h).lower()
            for h in self._parse_json(source.get("hashtags", [])) or []
        )

        for tag in source_hashtags:
            if tag in self.hashtag_index:
                for vid in self.hashtag_index[tag]:
                    if vid != video_id:
                        candidates[vid] = candidates.get(vid, 0) + 0.1

        # Sort and get top N
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        recommendations = []
        for vid, score in sorted_candidates:
            target = self.video_metadata.get(vid, {})
            target_hashtags = set(
                str(h).lower()
                for h in self._parse_json(target.get("hashtags", [])) or []
            )
            shared = list(source_hashtags & target_hashtags)[:5]

            explanation = self._generate_explanation(source, target, shared, "content")

            recommendations.append(Recommendation(
                video_id=vid,
                similarity_score=round(min(score, 1.0), 3),
                explanation=explanation,
                shared_hashtags=shared,
                engagement_similarity=0.0,
            ))

        return recommendations

    def _engagement_based(
        self,
        video_id: int,
        source: Dict[str, Any],
        n: int
    ) -> List[Recommendation]:
        """Engagement-based recommendations."""
        source_eng = self._parse_json(source.get("engagement_metrics", {})) or {}
        source_views = source_eng.get("play_count", 0) or 0
        source_likes = source_eng.get("like_count", 0) or 0
        source_rate = source_likes / source_views if source_views > 0 else 0

        candidates = []

        for vid, video in self.video_metadata.items():
            if vid == video_id:
                continue

            eng = self._parse_json(video.get("engagement_metrics", {})) or {}
            views = eng.get("play_count", 0) or 0
            likes = eng.get("like_count", 0) or 0
            rate = likes / views if views > 0 else 0

            # Calculate engagement similarity
            view_sim = 1 - abs(np.log1p(views) - np.log1p(source_views)) / 20
            rate_sim = 1 - abs(rate - source_rate)

            similarity = (view_sim + rate_sim) / 2
            candidates.append((vid, similarity, rate_sim))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for vid, score, eng_sim in candidates[:n]:
            target = self.video_metadata.get(vid, {})
            explanation = self._generate_explanation(source, target, [], "engagement")

            recommendations.append(Recommendation(
                video_id=vid,
                similarity_score=round(max(0, min(score, 1.0)), 3),
                explanation=explanation,
                shared_hashtags=[],
                engagement_similarity=round(eng_sim, 3),
            ))

        return recommendations

    def _hybrid(
        self,
        video_id: int,
        source: Dict[str, Any],
        n: int
    ) -> List[Recommendation]:
        """Hybrid recommendations combining content and engagement."""
        # Get both types of recommendations
        content_recs = self._content_based(video_id, source, n * 2)
        engagement_recs = self._engagement_based(video_id, source, n * 2)

        # Create score dictionary
        scores: Dict[int, Dict[str, float]] = {}

        for rec in content_recs:
            scores[rec.video_id] = {
                "content": rec.similarity_score,
                "engagement": 0,
                "shared_hashtags": rec.shared_hashtags,
            }

        for rec in engagement_recs:
            if rec.video_id in scores:
                scores[rec.video_id]["engagement"] = rec.engagement_similarity
            else:
                scores[rec.video_id] = {
                    "content": 0,
                    "engagement": rec.engagement_similarity,
                    "shared_hashtags": [],
                }

        # Calculate hybrid scores (weighted average)
        hybrid_scores = []
        for vid, s in scores.items():
            hybrid = s["content"] * 0.6 + s["engagement"] * 0.4
            hybrid_scores.append((vid, hybrid, s))

        # Sort by hybrid score
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for vid, score, s in hybrid_scores[:n]:
            target = self.video_metadata.get(vid, {})
            explanation = self._generate_explanation(
                source, target, s["shared_hashtags"], "hybrid"
            )

            recommendations.append(Recommendation(
                video_id=vid,
                similarity_score=round(score, 3),
                explanation=explanation,
                shared_hashtags=s["shared_hashtags"],
                engagement_similarity=round(s["engagement"], 3),
            ))

        return recommendations

    def _generate_explanation(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
        shared_hashtags: List[str],
        method: str
    ) -> str:
        """Generate human-readable explanation for recommendation."""
        explanations = []

        if shared_hashtags:
            explanations.append(f"Shares #{', #'.join(shared_hashtags[:3])}")

        source_eng = self._parse_json(source.get("engagement_metrics", {})) or {}
        target_eng = self._parse_json(target.get("engagement_metrics", {})) or {}

        source_views = source_eng.get("play_count", 0) or 0
        target_views = target_eng.get("play_count", 0) or 0

        if target_views > 0 and source_views > 0:
            ratio = target_views / source_views
            if 0.5 < ratio < 2:
                explanations.append("Similar reach")
            elif ratio >= 2:
                explanations.append("Higher performing")

        if method == "content" and not explanations:
            explanations.append("Similar content style")
        elif method == "engagement" and not explanations:
            explanations.append("Similar engagement pattern")
        elif not explanations:
            explanations.append("Related content")

        return " | ".join(explanations)

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
_recommendation_engine: Optional[RecommendationEngine] = None


def get_recommendation_engine() -> RecommendationEngine:
    """Get or create recommendation engine instance."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine
