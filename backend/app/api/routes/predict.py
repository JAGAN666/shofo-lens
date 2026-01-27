from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

router = APIRouter()


class VideoInput(BaseModel):
    duration_ms: int = Field(..., description="Video duration in milliseconds")
    description: Optional[str] = Field(None, description="Video description/caption")
    transcript: Optional[str] = Field(None, description="Video transcript")
    hashtags: Optional[List[str]] = Field(None, description="List of hashtags")
    sticker_text: Optional[List[str]] = Field(None, description="Text overlays")
    is_ai_generated: bool = Field(False, description="Whether video is AI generated")
    is_ad: bool = Field(False, description="Whether video is an advertisement")
    hour_posted: Optional[int] = Field(12, ge=0, le=23, description="Hour of posting (0-23)")
    day_of_week: Optional[int] = Field(0, ge=0, le=6, description="Day of week (0=Monday)")


class PredictionResponse(BaseModel):
    engagement_score: float = Field(..., description="Predicted engagement rate (0-1)")
    engagement_percentile: float = Field(..., description="Percentile compared to dataset")
    feature_importance: Dict[str, float] = Field(..., description="SHAP values for each feature")
    recommendations: List[str] = Field(..., description="Tips to improve engagement")


class BatchPredictionRequest(BaseModel):
    videos: List[VideoInput]


@router.post("/predict", response_model=PredictionResponse)
async def predict_engagement(request: Request, video: VideoInput):
    """
    Predict engagement for a video based on its characteristics.

    Returns:
    - engagement_score: Predicted engagement rate
    - feature_importance: SHAP values showing which features impact prediction
    - recommendations: Tips to improve engagement
    """
    predictor = request.app.state.predictor

    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Engagement prediction model not loaded. Please run the training script first."
        )

    # Prepare video data
    video_data = {
        "duration_ms": video.duration_ms,
        "description": video.description or "",
        "transcript": video.transcript or "",
        "hashtags": video.hashtags or [],
        "sticker_text": video.sticker_text or [],
        "is_ai_generated": video.is_ai_generated,
        "is_ad": video.is_ad,
        "date_posted": None,
        "comments": [],
        "engagement_metrics": {},
    }

    # Override date features
    video_data["_hour_posted"] = video.hour_posted
    video_data["_day_of_week"] = video.day_of_week

    result = predictor.predict(video_data)

    # Generate recommendations based on feature importance
    recommendations = _generate_recommendations(result["feature_importance"], video_data)

    # Calculate percentile (simplified - in production would compare against distribution)
    percentile = min(100, max(0, result["engagement_score"] * 100))

    return PredictionResponse(
        engagement_score=result["engagement_score"],
        engagement_percentile=percentile,
        feature_importance=result["feature_importance"],
        recommendations=recommendations,
    )


@router.post("/predict/batch")
async def predict_batch(request: Request, batch: BatchPredictionRequest):
    """Predict engagement for multiple videos."""
    predictor = request.app.state.predictor

    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Engagement prediction model not loaded."
        )

    results = []
    for video in batch.videos:
        video_data = {
            "duration_ms": video.duration_ms,
            "description": video.description or "",
            "transcript": video.transcript or "",
            "hashtags": video.hashtags or [],
            "sticker_text": video.sticker_text or [],
            "is_ai_generated": video.is_ai_generated,
            "is_ad": video.is_ad,
            "date_posted": None,
            "comments": [],
            "engagement_metrics": {},
        }
        result = predictor.predict(video_data)
        results.append({
            "engagement_score": result["engagement_score"],
            "raw_prediction": result["raw_prediction"],
        })

    return {"predictions": results}


@router.get("/predict/feature-importance")
async def get_feature_importance(request: Request):
    """Get global feature importance from the trained model."""
    predictor = request.app.state.predictor

    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Engagement prediction model not loaded."
        )

    importance = predictor.get_feature_importance()

    # Sort by importance
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return {
        "feature_importance": sorted_importance,
        "description": {
            "duration_ms": "Video duration in milliseconds",
            "hashtag_count": "Number of hashtags used",
            "description_length": "Length of video description",
            "transcript_length": "Length of video transcript",
            "has_transcript": "Whether video has a transcript",
            "is_ai_generated": "Whether video is AI generated",
            "is_ad": "Whether video is an advertisement",
            "hour_posted": "Hour of day when posted",
            "day_of_week": "Day of week when posted",
            "sticker_count": "Number of text overlays/stickers",
            "comment_count_sample": "Sample comment count",
        }
    }


def _generate_recommendations(
    feature_importance: Dict[str, float],
    video_data: Dict[str, Any]
) -> List[str]:
    """Generate actionable recommendations based on feature importance."""
    recommendations = []

    # Sort features by negative impact
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1]
    )

    for feature, impact in sorted_features[:3]:
        if impact < -0.1:
            if feature == "duration_ms" and video_data.get("duration_ms", 0) < 15000:
                recommendations.append("Consider making longer videos (15-60 seconds tend to perform better)")
            elif feature == "hashtag_count":
                recommendations.append("Try using 3-5 relevant hashtags to improve discoverability")
            elif feature == "description_length":
                recommendations.append("Add a more detailed description with keywords")
            elif feature == "has_transcript":
                recommendations.append("Videos with clear audio/speech tend to get more engagement")
            elif feature == "hour_posted":
                recommendations.append("Try posting during peak hours (6-9 PM local time)")

    if not recommendations:
        recommendations.append("Your video characteristics look good for engagement!")

    return recommendations[:5]
