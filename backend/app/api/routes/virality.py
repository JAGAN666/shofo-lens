"""
Virality Prediction API Endpoints

Advanced virality prediction with confidence intervals and explainability.
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple

router = APIRouter()


class VideoInput(BaseModel):
    duration_ms: int = Field(..., description="Video duration in milliseconds")
    description: Optional[str] = Field(None, description="Video description")
    transcript: Optional[str] = Field(None, description="Video transcript")
    hashtags: Optional[List[str]] = Field(None, description="List of hashtags")
    sticker_text: Optional[List[str]] = Field(None, description="Text overlays")
    is_ai_generated: bool = Field(False)
    is_ad: bool = Field(False)
    hour_posted: Optional[int] = Field(12, ge=0, le=23)
    day_of_week: Optional[int] = Field(0, ge=0, le=6)


class ViralityResponse(BaseModel):
    viral_score: float = Field(..., description="Overall virality score (0-100)")
    viral_tier: str = Field(..., description="Viral tier: Low/Medium/High/Viral")
    confidence: float = Field(..., description="Prediction confidence (0-1)")

    predicted_views: int
    predicted_likes: int
    predicted_shares: int
    predicted_comments: int

    views_range: Tuple[int, int] = Field(..., description="80% confidence interval for views")
    likes_range: Tuple[int, int] = Field(..., description="80% confidence interval for likes")

    top_factors: List[Dict[str, Any]] = Field(..., description="Top influencing factors")
    recommendations: List[str] = Field(..., description="Actionable recommendations")


@router.post("/predict/virality", response_model=ViralityResponse)
async def predict_virality(request: Request, video: VideoInput):
    """
    Predict video virality with confidence intervals.

    Returns:
    - viral_score: Overall virality score (0-100)
    - viral_tier: Categorical tier (Low/Medium/High/Viral)
    - Predicted metrics with confidence intervals
    - Top factors influencing the prediction
    - Actionable recommendations to improve virality
    """
    # Check if virality predictor is available
    virality_predictor = getattr(request.app.state, "virality_predictor", None)

    if virality_predictor is None or not virality_predictor.models:
        # Fallback to basic predictor if virality model not loaded
        raise HTTPException(
            status_code=503,
            detail="Virality predictor not loaded. Run setup_data.py first."
        )

    video_data = {
        "duration_ms": video.duration_ms,
        "description": video.description or "",
        "transcript": video.transcript or "",
        "hashtags": video.hashtags or [],
        "sticker_text": video.sticker_text or [],
        "is_ai_generated": video.is_ai_generated,
        "is_ad": video.is_ad,
        "date_posted": None,
        "engagement_metrics": {},
    }

    prediction = virality_predictor.predict(video_data)

    return ViralityResponse(
        viral_score=round(prediction.viral_score, 1),
        viral_tier=prediction.viral_tier,
        confidence=round(prediction.confidence, 2),
        predicted_views=prediction.predicted_views,
        predicted_likes=prediction.predicted_likes,
        predicted_shares=prediction.predicted_shares,
        predicted_comments=prediction.predicted_comments,
        views_range=prediction.views_range,
        likes_range=prediction.likes_range,
        top_factors=prediction.top_factors,
        recommendations=prediction.recommendations,
    )


@router.get("/predict/virality/feature-importance")
async def get_virality_feature_importance(request: Request):
    """Get global feature importance for virality prediction."""
    virality_predictor = getattr(request.app.state, "virality_predictor", None)

    if virality_predictor is None:
        return {"feature_importance": {}, "error": "Model not loaded"}

    importance = virality_predictor.get_feature_importance()

    return {
        "feature_importance": dict(sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )),
    }
