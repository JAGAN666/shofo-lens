#!/usr/bin/env python3
"""
Load real Shofo data from metadata.parquet (no video decoding needed).
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("QDRANT_HOST", "4594413c-b3c6-4a73-b706-69f0a7b5c73a.us-east4-0.gcp.cloud.qdrant.io")
os.environ.setdefault("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.95JBAqrvCOLLLrYA7Fob1qSBOFnMbVos36FKBWd8buE")


def download_metadata(limit=1000):
    """Download and load metadata.parquet directly."""
    logger.info("Downloading metadata.parquet from Shofo dataset...")

    parquet_path = hf_hub_download(
        repo_id="Shofo/shofo-tiktok-general-small",
        filename="metadata.parquet",
        repo_type="dataset",
    )

    logger.info(f"Loading parquet from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    logger.info(f"Dataset has {len(df)} total records")
    logger.info(f"Columns: {list(df.columns)}")

    if limit:
        df = df.head(limit)

    logger.info(f"Using {len(df)} records")

    # Convert to list of dicts
    videos = df.to_dict('records')
    return videos


def save_and_index(videos):
    """Save to JSON and index to Qdrant Cloud."""
    from tqdm import tqdm
    from app.ml.embeddings import EmbeddingService, create_combined_text
    from app.ml.search import SearchService

    # Save JSON
    Path("data").mkdir(exist_ok=True)

    serializable = []
    for v in videos:
        item = {}
        for key, value in v.items():
            if isinstance(value, bytes):
                continue
            if pd.isna(value):
                item[key] = None
            elif hasattr(value, "isoformat"):
                item[key] = value.isoformat()
            elif hasattr(value, 'item'):
                item[key] = value.item()
            else:
                item[key] = value
        serializable.append(item)

    with open("data/videos.json", "w") as f:
        json.dump(serializable, f, default=str)
    logger.info(f"Saved {len(videos)} REAL videos to data/videos.json")

    # Generate embeddings
    logger.info("Loading sentence-transformers model...")
    embedding_service = EmbeddingService()

    texts = []
    for video in tqdm(videos, desc="Preparing texts"):
        def is_null(val):
            if val is None:
                return True
            try:
                if pd.isna(val):
                    return True
            except:
                pass
            return False

        hashtags = video.get("hashtags", [])
        if is_null(hashtags):
            hashtags = []
        elif isinstance(hashtags, str):
            try:
                hashtags = json.loads(hashtags)
            except:
                hashtags = []

        stickers = video.get("sticker_text", [])
        if is_null(stickers):
            stickers = []
        elif isinstance(stickers, str):
            try:
                stickers = json.loads(stickers)
            except:
                stickers = []

        transcript = video.get("transcript", "")
        if is_null(transcript):
            transcript = ""

        description = video.get("description", "")
        if is_null(description):
            description = ""

        text = create_combined_text(
            transcript=transcript,
            description=description,
            hashtags=hashtags if isinstance(hashtags, list) else [],
            sticker_text=stickers if isinstance(stickers, list) else [],
        )
        texts.append(text)

    logger.info(f"Generating embeddings for {len(texts)} videos...")
    embeddings = embedding_service.encode(texts, batch_size=32)
    np.save("data/embeddings.npy", embeddings)
    logger.info(f"Embeddings saved: {embeddings.shape}")

    # Index to Qdrant Cloud
    logger.info("Connecting to Qdrant Cloud...")
    search_service = SearchService()

    video_ids = []
    metadata = []

    for i, video in enumerate(videos):
        vid = video.get("video_id", i)
        if pd.isna(vid):
            vid = i
        if hasattr(vid, 'item'):
            vid = vid.item()
        video_ids.append(int(vid) if vid else i)

        engagement = video.get("engagement_metrics", {})
        if isinstance(engagement, str):
            try:
                engagement = json.loads(engagement)
            except:
                engagement = {}
        if pd.isna(engagement) if not isinstance(engagement, dict) else False:
            engagement = {}

        language_info = video.get("language", {})
        if isinstance(language_info, str):
            try:
                language_info = json.loads(language_info)
            except:
                language_info = {}

        hashtags = video.get("hashtags", [])
        if isinstance(hashtags, str):
            try:
                hashtags = json.loads(hashtags)
            except:
                hashtags = []

        def safe_val(val, default=0):
            if val is None or (hasattr(pd, 'isna') and pd.isna(val)):
                return default
            if hasattr(val, 'item'):
                return val.item()
            return val

        def safe_str(val):
            if val is None or (hasattr(pd, 'isna') and pd.isna(val)):
                return ""
            return str(val)

        meta = {
            "video_id": video_ids[-1],
            "web_url": safe_str(video.get("web_url")),
            "creator": safe_str(video.get("creator")),
            "description": safe_str(video.get("description"))[:500],
            "transcript": safe_str(video.get("transcript"))[:1000],
            "has_transcript": bool(video.get("transcript") and not pd.isna(video.get("transcript"))),
            "duration_ms": int(safe_val(video.get("duration_ms"), 0)),
            "is_ai_generated": bool(safe_val(video.get("is_ai_generated"), False)),
            "is_ad": bool(safe_val(video.get("is_ad"), False)),
            "language": language_info.get("desc_language") if isinstance(language_info, dict) else None,
            "play_count": int(safe_val(engagement.get("play_count", 0) if isinstance(engagement, dict) else 0)),
            "like_count": int(safe_val(engagement.get("like_count", 0) if isinstance(engagement, dict) else 0)),
            "share_count": int(safe_val(engagement.get("share_count", 0) if isinstance(engagement, dict) else 0)),
            "comment_count": int(safe_val(engagement.get("comment_count", 0) if isinstance(engagement, dict) else 0)),
            "hashtag_count": len(hashtags) if isinstance(hashtags, list) else 0,
        }
        metadata.append(meta)

    logger.info("Indexing to Qdrant Cloud...")
    search_service.index_videos(
        video_ids=video_ids,
        embeddings=embeddings,
        metadata=metadata,
    )
    logger.info("Qdrant Cloud indexing complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Loading REAL Shofo TikTok Metadata")
    logger.info("=" * 60)

    videos = download_metadata(limit=args.limit)
    save_and_index(videos)

    logger.info("=" * 60)
    logger.info("REAL DATA SETUP COMPLETE!")
    logger.info(f"Loaded {len(videos)} real TikTok videos from Shofo dataset")
    logger.info("=" * 60)
