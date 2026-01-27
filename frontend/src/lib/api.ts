const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export interface SearchResult {
  video_id: number;
  score: number;
  description: string | null;
  transcript: string | null;
  creator: string | null;
  web_url: string | null;
  play_count: number | null;
  like_count: number | null;
  duration_ms: number | null;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
}

export interface PredictionResponse {
  engagement_score: number;
  engagement_percentile: number;
  feature_importance: Record<string, number>;
  recommendations: string[];
}

export interface AnalyticsOverview {
  total_videos: number;
  total_views: number;
  total_likes: number;
  avg_duration_sec: number;
  videos_with_transcript: number;
  transcript_percentage: number;
  ai_generated_videos: number;
  ad_videos: number;
}

export interface HashtagData {
  tag: string;
  count: number;
}

export interface LanguageData {
  code: string;
  count: number;
  percentage: number;
}

export interface DurationBucket {
  bucket: string;
  count: number;
  percentage: number;
}

export async function searchVideos(
  query: string,
  options?: {
    limit?: number;
    min_views?: number;
    language?: string;
    has_transcript?: boolean;
  }
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE}/api/v1/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      limit: options?.limit || 20,
      min_views: options?.min_views,
      language: options?.language,
      has_transcript: options?.has_transcript,
    }),
  });

  if (!response.ok) {
    throw new Error("Search failed");
  }

  return response.json();
}

export async function predictEngagement(videoData: {
  duration_ms: number;
  description?: string;
  transcript?: string;
  hashtags?: string[];
  sticker_text?: string[];
  is_ai_generated?: boolean;
  is_ad?: boolean;
  hour_posted?: number;
  day_of_week?: number;
}): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(videoData),
  });

  if (!response.ok) {
    throw new Error("Prediction failed");
  }

  return response.json();
}

export async function getAnalyticsOverview(): Promise<AnalyticsOverview> {
  const response = await fetch(`${API_BASE}/api/v1/analytics/overview`);
  if (!response.ok) throw new Error("Failed to fetch analytics");
  return response.json();
}

export async function getTopHashtags(
  limit = 20
): Promise<{ hashtags: HashtagData[]; total_unique: number }> {
  const response = await fetch(
    `${API_BASE}/api/v1/analytics/top-hashtags?limit=${limit}`
  );
  if (!response.ok) throw new Error("Failed to fetch hashtags");
  return response.json();
}

export async function getLanguageDistribution(
  limit = 15
): Promise<{ languages: LanguageData[]; total_languages: number }> {
  const response = await fetch(
    `${API_BASE}/api/v1/analytics/language-distribution?limit=${limit}`
  );
  if (!response.ok) throw new Error("Failed to fetch languages");
  return response.json();
}

export async function getDurationDistribution(): Promise<{
  distribution: DurationBucket[];
  total_videos: number;
}> {
  const response = await fetch(
    `${API_BASE}/api/v1/analytics/duration-distribution`
  );
  if (!response.ok) throw new Error("Failed to fetch duration distribution");
  return response.json();
}

export async function getEngagementDistribution(): Promise<{
  views: { percentiles: Record<string, number>; mean: number };
  likes: { percentiles: Record<string, number>; mean: number };
  engagement_rate: { percentiles: Record<string, number>; mean: number };
}> {
  const response = await fetch(
    `${API_BASE}/api/v1/analytics/engagement-distribution`
  );
  if (!response.ok) throw new Error("Failed to fetch engagement distribution");
  return response.json();
}

export async function getFeatureImportance(): Promise<{
  feature_importance: Record<string, number>;
  description: Record<string, string>;
}> {
  const response = await fetch(`${API_BASE}/api/v1/predict/feature-importance`);
  if (!response.ok) throw new Error("Failed to fetch feature importance");
  return response.json();
}

// ============== SPECTACULAR FEATURES ==============

// Virality Prediction
export interface ViralityResponse {
  viral_score: number;
  viral_tier: string;
  confidence: number;
  predicted_views: number;
  predicted_likes: number;
  predicted_shares: number;
  predicted_comments: number;
  views_range: [number, number];
  likes_range: [number, number];
  top_factors: Array<{
    feature: string;
    value: number;
    impact: number;
    direction: string;
  }>;
  recommendations: string[];
}

export async function predictVirality(videoData: {
  duration_ms: number;
  description?: string;
  transcript?: string;
  hashtags?: string[];
  is_ai_generated?: boolean;
  is_ad?: boolean;
  hour_posted?: number;
  day_of_week?: number;
}): Promise<ViralityResponse> {
  const response = await fetch(`${API_BASE}/api/v1/predict/virality`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(videoData),
  });
  if (!response.ok) throw new Error("Virality prediction failed");
  return response.json();
}

// Content Classification
export interface ClassificationResponse {
  primary_category: string;
  primary_confidence: number;
  all_labels: Array<{ category: string; confidence: number }>;
  content_type: string;
  mood: string;
}

export async function classifyContent(videoData: {
  description?: string;
  transcript?: string;
  hashtags?: string[];
}): Promise<ClassificationResponse> {
  const response = await fetch(`${API_BASE}/api/v1/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(videoData),
  });
  if (!response.ok) throw new Error("Classification failed");
  return response.json();
}

// Trends
export interface TrendItem {
  name: string;
  count: number;
  velocity: number;
  avg_engagement: number;
  trend_status: string;
  rank_change: number;
}

export interface TrendResponse {
  trending_hashtags: TrendItem[];
  trending_topics: string[];
  emerging_hashtags: TrendItem[];
  declining_hashtags: TrendItem[];
  best_posting_times: Array<{
    hour: number;
    formatted: string;
    video_count: number;
    avg_engagement: number;
  }>;
  engagement_insights: Record<string, number>;
}

export async function getTrends(): Promise<TrendResponse> {
  const response = await fetch(`${API_BASE}/api/v1/trends`);
  if (!response.ok) throw new Error("Failed to fetch trends");
  return response.json();
}

// Topics
export interface Topic {
  id: number;
  name: string;
  keywords: string[];
  size: number;
  representative_docs: string[];
}

export interface TopicsResponse {
  topics: Topic[];
  total_documents: number;
  outliers: number;
  topic_distribution: Record<string, number>;
}

export async function getTopics(): Promise<TopicsResponse> {
  const response = await fetch(`${API_BASE}/api/v1/topics`);
  if (!response.ok) throw new Error("Failed to fetch topics");
  return response.json();
}

// Similar Videos
export interface SimilarVideo {
  video_id: number;
  similarity_score: number;
  explanation: string;
  shared_hashtags: string[];
  engagement_similarity: number;
}

export async function getSimilarVideos(
  videoId: number,
  limit = 10,
  method = "hybrid"
): Promise<{ recommendations: SimilarVideo[] }> {
  const response = await fetch(
    `${API_BASE}/api/v1/videos/${videoId}/similar?limit=${limit}&method=${method}`
  );
  if (!response.ok) throw new Error("Failed to fetch similar videos");
  return response.json();
}
