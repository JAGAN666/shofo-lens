#!/usr/bin/env python3
"""
Setup script to load the Shofo dataset, generate embeddings, and index into Qdrant.
Run this before starting the server.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from app.core.config import settings
from app.ml.embeddings import EmbeddingService, create_combined_text
from app.ml.search import SearchService
from app.ml.predictor import EngagementPredictor


def parse_json_field(value: Any) -> Any:
    """Parse a JSON field that might be a string or already parsed."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return value
    return value


def load_shofo_dataset(limit: int = None) -> List[Dict[str, Any]]:
    """Load the Shofo TikTok dataset from HuggingFace."""
    logger.info(f"Loading dataset: {settings.DATASET_NAME}")

    dataset = load_dataset(settings.DATASET_NAME, split="train")

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    logger.info(f"Loaded {len(dataset)} videos")
    return [dict(item) for item in dataset]


def generate_embeddings(
    videos: List[Dict[str, Any]],
    embedding_service: EmbeddingService,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate text embeddings for all videos."""
    logger.info("Generating text embeddings...")

    texts = []
    for video in tqdm(videos, desc="Preparing texts"):
        text = create_combined_text(
            transcript=video.get("transcript"),
            description=video.get("description"),
            hashtags=parse_json_field(video.get("hashtags")),
            sticker_text=parse_json_field(video.get("sticker_text")),
        )
        texts.append(text)

    logger.info(f"Encoding {len(texts)} texts...")
    embeddings = embedding_service.encode(texts, batch_size=batch_size)
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")

    return embeddings


def prepare_metadata(video: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare video metadata for Qdrant indexing."""
    engagement = parse_json_field(video.get("engagement_metrics", {})) or {}
    language_info = parse_json_field(video.get("language", {})) or {}
    hashtags = parse_json_field(video.get("hashtags", [])) or []

    return {
        "video_id": video.get("video_id"),
        "web_url": video.get("web_url"),
        "creator": video.get("creator"),
        "description": video.get("description", "")[:500] if video.get("description") else None,
        "transcript": video.get("transcript", "")[:1000] if video.get("transcript") else None,
        "has_transcript": bool(video.get("transcript")),
        "duration_ms": video.get("duration_ms"),
        "resolution": video.get("resolution"),
        "fps": video.get("fps"),
        "is_ai_generated": bool(video.get("is_ai_generated")),
        "is_ad": bool(video.get("is_ad")),
        "language": language_info.get("desc_language") if isinstance(language_info, dict) else None,
        "play_count": engagement.get("play_count", 0) or 0,
        "like_count": engagement.get("like_count", 0) or 0,
        "share_count": engagement.get("share_count", 0) or 0,
        "comment_count": engagement.get("comment_count", 0) or 0,
        "hashtag_count": len(hashtags) if isinstance(hashtags, list) else 0,
    }


def index_to_qdrant(
    videos: List[Dict[str, Any]],
    embeddings: np.ndarray,
    search_service: SearchService,
):
    """Index videos and embeddings into Qdrant."""
    logger.info("Indexing to Qdrant...")

    video_ids = []
    metadata = []

    for i, video in enumerate(videos):
        vid = video.get("video_id", i)
        video_ids.append(vid)
        metadata.append(prepare_metadata(video))

    search_service.index_videos(
        video_ids=video_ids,
        embeddings=embeddings,
        metadata=metadata,
    )

    logger.info("Indexing complete!")


def train_predictor(
    videos: List[Dict[str, Any]],
    predictor: EngagementPredictor,
):
    """Train the engagement prediction model."""
    logger.info("Training engagement predictor...")

    metrics = predictor.train(videos)

    logger.info(f"Training complete! Metrics: {metrics}")
    return metrics


def save_videos_json(videos: List[Dict[str, Any]], path: str = "data/videos.json"):
    """Save videos to JSON for the API to load."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    serializable = []
    for v in videos:
        item = {}
        for key, value in v.items():
            if isinstance(value, (dict, list)):
                item[key] = value
            elif hasattr(value, "isoformat"):
                item[key] = value.isoformat()
            else:
                item[key] = value
        serializable.append(item)

    with open(path, "w") as f:
        json.dump(serializable, f)

    logger.info(f"Saved {len(videos)} videos to {path}")


def main(
    limit: int = None,
    skip_embeddings: bool = False,
    skip_qdrant: bool = False,
    skip_training: bool = False,
):
    """Main setup function."""
    logger.info("=" * 60)
    logger.info("ShofoLens Data Setup")
    logger.info("=" * 60)

    # Load dataset
    videos = load_shofo_dataset(limit=limit)

    # Save for API
    save_videos_json(videos)

    if not skip_embeddings:
        # Initialize services
        embedding_service = EmbeddingService()

        # Generate embeddings
        embeddings = generate_embeddings(videos, embedding_service)

        # Save embeddings
        np.save("data/embeddings.npy", embeddings)
        logger.info("Saved embeddings to data/embeddings.npy")

        if not skip_qdrant:
            try:
                search_service = SearchService()
                index_to_qdrant(videos, embeddings, search_service)
            except Exception as e:
                logger.warning(f"Could not index to Qdrant: {e}")
                logger.warning("Run Qdrant with: docker run -p 6333:6333 qdrant/qdrant")

    if not skip_training:
        predictor = EngagementPredictor()
        train_predictor(videos, predictor)

    logger.info("=" * 60)
    logger.info("Setup complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup ShofoLens data")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant indexing")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")

    args = parser.parse_args()

    main(
        limit=args.limit,
        skip_embeddings=args.skip_embeddings,
        skip_qdrant=args.skip_qdrant,
        skip_training=args.skip_training,
    )
