# ShofoLens v2.0 - SPECTACULAR Edition

> Advanced Multimodal Video Intelligence Platform built on the [Shofo TikTok Dataset](https://huggingface.co/datasets/Shofo/shofo-tiktok-general-small)

**Demo:** [Live URL] | **API Docs:** [Swagger UI]

---

## What Makes This SPECTACULAR

| Feature | Description |
|---------|-------------|
| **Semantic Search** | Natural language search across 58K+ videos using embeddings |
| **Virality Prediction** | Multi-target prediction with confidence intervals and SHAP explanations |
| **Trend Detection** | Real-time trending hashtags, emerging topics, and optimal posting times |
| **Content Auto-Labeling** | Zero-shot classification into 15+ categories |
| **Topic Clustering** | Automatic theme discovery using BERTopic |
| **Similar Videos** | Hybrid content + engagement recommendations |

---

## Features

### 1. Virality Prediction Engine
- **Multi-target prediction**: Views, likes, shares, comments
- **Confidence intervals**: 80% prediction ranges using quantile regression
- **Explainability**: SHAP values showing WHY a video will go viral
- **Actionable recommendations**: Tips to improve engagement

### 2. Zero-Shot Content Classification
- Automatically labels videos into categories (Comedy, Dance, Tutorial, etc.)
- Multi-label support (a video can be "Comedy + Dance")
- Content type detection (Entertainment, Educational, Commercial)
- Mood analysis (Funny, Serious, Inspirational)

### 3. Trend Detection
- **Trending hashtags** with velocity metrics
- **Emerging trends** (rising fast but not yet mainstream)
- **Declining trends** (losing momentum)
- **Best posting times** based on engagement analysis

### 4. Topic Clustering
- Automatic theme discovery using BERTopic
- Interactive topic visualization
- Representative videos for each topic
- Keyword extraction per topic

### 5. Similar Video Recommendations
- Content-based filtering (embedding similarity)
- Engagement-based filtering (similar performance)
- Hybrid approach combining both
- Explainable recommendations

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11+ |
| ML | sentence-transformers, XGBoost, BERTopic, SHAP |
| Vector DB | Qdrant |
| Deployment | Railway (backend), Vercel (frontend), Qdrant Cloud |

---

## Quick Start

### Option 1: Docker Compose (Local)

```bash
cd shofo-lens
docker-compose up -d

# First time: setup data
docker-compose exec backend python scripts/setup_data.py --limit 5000

# Access
open http://localhost:3000
```

### Option 2: Local Development

**Backend:**
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Setup data
python scripts/setup_data.py --limit 5000

# Run server
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## API Endpoints

### Core Features
| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/search` | Semantic video search |
| `POST /api/v1/predict/virality` | Advanced virality prediction |
| `POST /api/v1/classify` | Content classification |
| `GET /api/v1/trends` | Trend analysis |
| `GET /api/v1/topics` | Topic clusters |
| `GET /api/v1/videos/{id}/similar` | Recommendations |

### Example: Virality Prediction
```bash
curl -X POST http://localhost:8000/api/v1/predict/virality \
  -H "Content-Type: application/json" \
  -d '{
    "duration_ms": 30000,
    "description": "Check out this amazing recipe!",
    "hashtags": ["food", "recipe", "cooking"],
    "hour_posted": 18
  }'
```

Response:
```json
{
  "viral_score": 72.5,
  "viral_tier": "High",
  "confidence": 0.85,
  "predicted_views": 125000,
  "predicted_likes": 8500,
  "views_range": [45000, 280000],
  "top_factors": [...],
  "recommendations": [...]
}
```

---

## Deployment

### Railway (Backend)
```bash
npm install -g @railway/cli
railway login
cd backend
railway init
railway up
```

### Vercel (Frontend)
```bash
npm install -g vercel
cd frontend
vercel --prod
```

### Qdrant Cloud
1. Sign up at https://cloud.qdrant.io
2. Create a free cluster
3. Update `QDRANT_HOST` and `QDRANT_API_KEY` in Railway env vars

---

## Project Structure

```
shofo-lens/
├── backend/
│   ├── app/
│   │   ├── api/routes/     # API endpoints
│   │   ├── core/           # Configuration
│   │   └── ml/             # ML modules
│   │       ├── virality.py     # Virality prediction
│   │       ├── classifier.py   # Zero-shot classification
│   │       ├── trends.py       # Trend detection
│   │       ├── topics.py       # BERTopic clustering
│   │       └── recommendations.py  # Similar videos
│   ├── scripts/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   │   ├── search-section.tsx
│   │   │   ├── virality-section.tsx
│   │   │   ├── trends-section.tsx
│   │   │   └── analytics-section.tsx
│   │   └── lib/
│   └── package.json
├── docker-compose.yml
└── README.md
```

---

## What This Demonstrates

1. **Advanced ML Engineering**
   - Multi-target regression with confidence intervals
   - Zero-shot classification without training data
   - Topic modeling with BERTopic
   - SHAP explainability

2. **Full-Stack Development**
   - Modern React/Next.js frontend
   - FastAPI async backend
   - Docker containerization
   - Cloud deployment

3. **Understanding of Shofo's Business**
   - Built on their actual dataset
   - Semantic search (their core offering)
   - Video intelligence features they need

---

## Performance

- **Search latency**: <500ms
- **Virality prediction**: <100ms
- **Classification**: <200ms
- **Trend analysis**: <1s (full dataset)

---

## Acknowledgments

- Dataset: [Shofo TikTok General (Small)](https://huggingface.co/datasets/Shofo/shofo-tiktok-general-small)
- Embeddings: [Sentence Transformers](https://www.sbert.net/)
- Topic Modeling: [BERTopic](https://maartengr.github.io/BERTopic/)
- Vector Search: [Qdrant](https://qdrant.tech/)

---

**Built to demonstrate video intelligence capabilities for Shofo.ai**
