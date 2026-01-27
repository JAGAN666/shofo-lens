import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger


def generate_demo_videos(count: int = 100) -> list:
    """Generate realistic demo video data for demonstration purposes."""

    creators = [
        "dancing_queen", "foodie_mike", "tech_guru_sam", "comedy_king",
        "fitness_jane", "beauty_belle", "travel_tom", "gamer_pro",
        "diy_dan", "pet_lover", "music_master", "art_studio",
        "life_hacks", "news_now", "fashion_fwd", "cook_with_me"
    ]

    hashtag_pools = {
        "dance": ["#dance", "#choreography", "#viral", "#fyp", "#trending", "#moves"],
        "food": ["#foodie", "#recipe", "#cooking", "#yummy", "#homemade", "#foodtok"],
        "tech": ["#tech", "#gadgets", "#review", "#unboxing", "#innovation", "#tips"],
        "comedy": ["#funny", "#comedy", "#lol", "#humor", "#jokes", "#memes"],
        "fitness": ["#fitness", "#workout", "#gym", "#health", "#motivation", "#gains"],
        "beauty": ["#beauty", "#makeup", "#skincare", "#tutorial", "#glam", "#ootd"],
        "travel": ["#travel", "#adventure", "#explore", "#wanderlust", "#vacation", "#views"],
        "gaming": ["#gaming", "#gamer", "#gameplay", "#streamer", "#esports", "#games"],
        "diy": ["#diy", "#crafts", "#howto", "#tutorial", "#creative", "#hack"],
        "pets": ["#pets", "#dogs", "#cats", "#cute", "#animals", "#adorable"],
    }

    descriptions = {
        "dance": [
            "New dance challenge! Can you keep up? 💃",
            "This choreography took me 3 hours to learn",
            "When the beat drops just right 🔥",
            "POV: you finally nail the moves",
        ],
        "food": [
            "The easiest 5-minute pasta you'll ever make 🍝",
            "My grandma's secret recipe finally revealed",
            "Wait until the end... trust me 😋",
            "This changed my breakfast game forever",
        ],
        "tech": [
            "This gadget is a game changer 📱",
            "5 iPhone tricks you didn't know existed",
            "Honest review after 30 days of use",
            "The future is here and I'm not ready",
        ],
        "comedy": [
            "POV: Monday morning vibes 😂",
            "When your friend says 'just 5 more minutes'",
            "Nobody: ... Me at 3am:",
            "Acting challenge gone wrong 💀",
        ],
        "fitness": [
            "30-day transformation results 💪",
            "5 exercises you're doing wrong",
            "No gym? No problem. Try this at home",
            "The workout that changed my life",
        ],
        "beauty": [
            "Everyday makeup in under 5 minutes ✨",
            "This product is SO underrated",
            "Get ready with me for date night",
            "Skincare routine that cleared my acne",
        ],
        "travel": [
            "Found a hidden gem in Italy 🇮🇹",
            "This view was worth the 8-hour hike",
            "Travel hack: how I fly for almost free",
            "Places you NEED to visit before you die",
        ],
        "gaming": [
            "Insane clutch in the final round 🎮",
            "When the random teammate is actually good",
            "This game broke me emotionally",
            "Speed run attempt number 47...",
        ],
        "diy": [
            "Turn your old jeans into something amazing",
            "Home hack that saves $100/month",
            "You won't believe this transformation",
            "Finally organized my entire room",
        ],
        "pets": [
            "My dog's reaction to seeing me after 2 weeks 🐕",
            "Cat logic will never make sense",
            "The most dramatic pet on the internet",
            "Teaching my puppy tricks - day 15",
        ],
    }

    transcripts = {
        "dance": "Okay so first you start with your right foot, step step, then spin, and hit that pose. Ready? Let's go!",
        "food": "You're gonna need two cups of flour, a pinch of salt, and here's the secret ingredient that makes all the difference.",
        "tech": "So I've been using this for about a month now and I have to say, it completely exceeded my expectations. Let me show you why.",
        "comedy": "So there I was, minding my own business, when suddenly... you're not gonna believe what happened next.",
        "fitness": "Push through it, you've got this! Five more reps, let's go! Remember to keep your core tight and breathe.",
        "beauty": "So the key is to blend in circular motions, don't drag it. And always set your makeup, that's the secret to making it last all day.",
        "travel": "This place is absolutely incredible. The locals are so friendly, the food is amazing, and look at this view!",
        "gaming": "Come on come on, we've got this. Watch the flank, watch the flank! Yes! Let's go, that was insane!",
        "diy": "All you need is some hot glue, cardboard, and a little bit of creativity. Let me show you step by step.",
        "pets": "Who's a good boy? Who wants a treat? Look at those eyes, I can't say no to that face!",
    }

    videos = []
    categories = list(hashtag_pools.keys())
    base_time = datetime.now() - timedelta(days=30)

    for i in range(count):
        category = random.choice(categories)
        creator = random.choice(creators)

        # Generate realistic engagement numbers (following power law distribution)
        base_views = random.randint(1000, 500000)
        engagement_rate = random.uniform(0.02, 0.15)

        play_count = base_views + random.randint(0, base_views)
        like_count = int(play_count * engagement_rate * random.uniform(0.8, 1.2))
        share_count = int(like_count * random.uniform(0.05, 0.2))
        comment_count = int(like_count * random.uniform(0.02, 0.1))

        # Random posting time
        hours_ago = random.randint(0, 720)  # Up to 30 days ago
        posted_time = base_time + timedelta(hours=hours_ago)

        video = {
            "video_id": 1000 + i,
            "web_url": f"https://www.tiktok.com/@{creator}/video/{1000000 + i}",
            "creator": creator,
            "description": random.choice(descriptions[category]),
            "transcript": transcripts[category] if random.random() > 0.3 else None,
            "hashtags": random.sample(hashtag_pools[category], k=random.randint(3, 6)) + ["#fyp", "#viral"],
            "duration_ms": random.randint(5000, 180000),  # 5s to 3min
            "resolution": random.choice(["1080x1920", "720x1280", "540x960"]),
            "fps": random.choice([24.0, 30.0, 60.0]),
            "is_ai_generated": random.random() < 0.05,
            "is_ad": random.random() < 0.1,
            "date_posted": posted_time.isoformat(),
            "language": {
                "desc_language": random.choice(["en", "en", "en", "es", "pt", "fr"]),
                "confidence": random.uniform(0.85, 0.99),
            },
            "sticker_text": [f"Text overlay {j}" for j in range(random.randint(0, 3))],
            "comments": [
                {"text": "This is amazing! 🔥", "likes": random.randint(10, 1000)},
                {"text": "How do you do this?", "likes": random.randint(5, 500)},
                {"text": "Following for more!", "likes": random.randint(1, 200)},
            ] if random.random() > 0.2 else [],
            "engagement_metrics": {
                "play_count": play_count,
                "like_count": like_count,
                "share_count": share_count,
                "comment_count": comment_count,
                "save_count": int(like_count * random.uniform(0.1, 0.3)),
            },
        }
        videos.append(video)

    return videos


def load_video_cache():
    """Load videos from JSON file or generate demo data."""
    from app.api.routes.videos import load_videos_to_cache

    data_path = Path("data/videos.json")

    if data_path.exists():
        logger.info(f"Loading videos from {data_path}")
        with open(data_path, "r") as f:
            videos = json.load(f)
        load_videos_to_cache(videos)
        logger.info(f"Loaded {len(videos)} videos into cache")
    else:
        logger.info("No video data file found, generating demo data...")
        videos = generate_demo_videos(100)
        load_videos_to_cache(videos)
        logger.info(f"Generated and loaded {len(videos)} demo videos into cache")
