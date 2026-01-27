import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
from loguru import logger

from app.core.config import settings


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        if not texts:
            return np.array([])

        # Handle None or empty strings
        processed_texts = [t if t else "" for t in texts]

        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding."""
        return self.encode([text])[0]

    def combine_embeddings(
        self,
        text_embedding: np.ndarray,
        weight_text: float = 1.0,
    ) -> np.ndarray:
        """Combine text embeddings with optional weighting."""
        combined = text_embedding * weight_text
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined


def create_combined_text(
    transcript: Optional[str],
    description: Optional[str],
    hashtags: Optional[List[str]],
    sticker_text: Optional[List[str]],
) -> str:
    """Create a combined text representation from video metadata."""
    parts = []

    if transcript:
        parts.append(f"Transcript: {transcript[:1000]}")

    if description:
        parts.append(f"Description: {description}")

    if hashtags:
        if isinstance(hashtags, list):
            parts.append(f"Hashtags: {' '.join(hashtags[:20])}")
        else:
            parts.append(f"Hashtags: {hashtags}")

    if sticker_text:
        if isinstance(sticker_text, list):
            parts.append(f"Text: {' '.join(sticker_text[:10])}")
        else:
            parts.append(f"Text: {sticker_text}")

    return " | ".join(parts) if parts else "No text content"
