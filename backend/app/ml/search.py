from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from app.core.config import settings


class SearchService:
    """Service for semantic search using Qdrant vector database."""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self.collection_name = settings.QDRANT_COLLECTION
        self.dimension = settings.EMBEDDING_DIMENSION
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the collection exists, create if not."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection {self.collection_name} created")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}")
            logger.warning("Search functionality will be limited until Qdrant is available")

    def index_videos(
        self,
        video_ids: List[int],
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        batch_size: int = 100,
    ):
        """Index videos with their embeddings and metadata."""
        points = []
        for i, (vid, emb, meta) in enumerate(zip(video_ids, embeddings, metadata)):
            points.append(
                PointStruct(
                    id=int(vid),
                    vector=emb.tolist(),
                    payload=meta,
                )
            )

            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                logger.info(f"Indexed {i + 1} videos")
                points = []

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        logger.info(f"Indexing complete. Total: {len(video_ids)} videos")

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar videos."""
        filter_conditions = None

        if filters:
            conditions = []
            if "min_views" in filters:
                conditions.append(
                    models.FieldCondition(
                        key="play_count",
                        range=models.Range(gte=filters["min_views"]),
                    )
                )
            if "language" in filters:
                conditions.append(
                    models.FieldCondition(
                        key="language",
                        match=models.MatchValue(value=filters["language"]),
                    )
                )
            if "has_transcript" in filters and filters["has_transcript"]:
                conditions.append(
                    models.FieldCondition(
                        key="has_transcript",
                        match=models.MatchValue(value=True),
                    )
                )
            if conditions:
                filter_conditions = models.Filter(must=conditions)

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=filter_conditions,
                with_payload=True,
            )

            return [
                {
                    "video_id": hit.id,
                    "score": hit.score,
                    **hit.payload,
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
