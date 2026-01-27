import json
from pathlib import Path
from loguru import logger


def load_video_cache():
    """Load videos from JSON file into the video cache."""
    from app.api.routes.videos import load_videos_to_cache

    data_path = Path("data/videos.json")

    if data_path.exists():
        logger.info(f"Loading videos from {data_path}")
        with open(data_path, "r") as f:
            videos = json.load(f)
        load_videos_to_cache(videos)
        logger.info(f"Loaded {len(videos)} videos into cache")
    else:
        logger.warning(f"No video data found at {data_path}")
        logger.warning("Run 'python scripts/setup_data.py' to download and process the dataset")
