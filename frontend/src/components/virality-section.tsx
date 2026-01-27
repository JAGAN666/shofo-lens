"use client";

import { useState } from "react";
import {
  Rocket,
  Loader2,
  TrendingUp,
  Eye,
  Heart,
  Share2,
  MessageCircle,
  Lightbulb,
  Target,
  Sparkles,
} from "lucide-react";
import { predictVirality, ViralityResponse } from "@/lib/api";
import { formatNumber } from "@/lib/utils";

export function ViralitySection() {
  const [formData, setFormData] = useState({
    duration_ms: 30000,
    description: "",
    transcript: "",
    hashtags: "",
    is_ai_generated: false,
    is_ad: false,
    hour_posted: 18,
    day_of_week: 3,
  });
  const [prediction, setPrediction] = useState<ViralityResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await predictVirality({
        duration_ms: formData.duration_ms,
        description: formData.description || undefined,
        transcript: formData.transcript || undefined,
        hashtags: formData.hashtags
          ? formData.hashtags.split(",").map((h) => h.trim())
          : undefined,
        is_ai_generated: formData.is_ai_generated,
        is_ad: formData.is_ad,
        hour_posted: formData.hour_posted,
        day_of_week: formData.day_of_week,
      });
      setPrediction(result);
    } catch (err) {
      setError("Prediction failed. Make sure the backend is running with trained models.");
    } finally {
      setLoading(false);
    }
  };

  const daysOfWeek = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

  const getTierColor = (tier: string) => {
    switch (tier) {
      case "Viral": return "text-purple-500 bg-purple-500/20";
      case "High": return "text-green-500 bg-green-500/20";
      case "Medium": return "text-yellow-500 bg-yellow-500/20";
      default: return "text-gray-500 bg-gray-500/20";
    }
  };

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-purple-500/20 to-pink-500/20 text-purple-400 mb-4">
          <Sparkles className="w-4 h-4" />
          <span className="text-sm font-medium">Advanced ML Powered</span>
        </div>
        <h2 className="text-2xl font-bold mb-2">Virality Prediction Engine</h2>
        <p className="text-muted-foreground">
          Predict how viral your video will be with confidence intervals and actionable insights
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="space-y-4">
          <div className="p-6 rounded-2xl border border-border bg-card">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-primary" />
              Video Details
            </h3>

            <div className="space-y-4">
              {/* Duration */}
              <div>
                <label className="text-sm font-medium mb-2 block">Duration</label>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    min="5"
                    max="180"
                    value={formData.duration_ms / 1000}
                    onChange={(e) =>
                      setFormData({ ...formData, duration_ms: parseInt(e.target.value) * 1000 })
                    }
                    className="flex-1 accent-primary"
                  />
                  <span className="text-sm font-mono w-12 text-right">
                    {Math.round(formData.duration_ms / 1000)}s
                  </span>
                </div>
              </div>

              {/* Description */}
              <div>
                <label className="text-sm font-medium mb-2 block">Description / Caption</label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  placeholder="What's your video about?"
                  rows={2}
                  className="w-full px-3 py-2 rounded-lg border border-border bg-background resize-none focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>

              {/* Transcript */}
              <div>
                <label className="text-sm font-medium mb-2 block">Transcript / Speech</label>
                <textarea
                  value={formData.transcript}
                  onChange={(e) => setFormData({ ...formData, transcript: e.target.value })}
                  placeholder="What do you say in the video?"
                  rows={2}
                  className="w-full px-3 py-2 rounded-lg border border-border bg-background resize-none focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>

              {/* Hashtags */}
              <div>
                <label className="text-sm font-medium mb-2 block">Hashtags</label>
                <input
                  type="text"
                  value={formData.hashtags}
                  onChange={(e) => setFormData({ ...formData, hashtags: e.target.value })}
                  placeholder="fyp, viral, trending (comma-separated)"
                  className="w-full px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>

              {/* Time */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Hour Posted</label>
                  <select
                    value={formData.hour_posted}
                    onChange={(e) => setFormData({ ...formData, hour_posted: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 rounded-lg border border-border bg-background"
                  >
                    {Array.from({ length: 24 }, (_, i) => (
                      <option key={i} value={i}>{i.toString().padStart(2, "0")}:00</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Day of Week</label>
                  <select
                    value={formData.day_of_week}
                    onChange={(e) => setFormData({ ...formData, day_of_week: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 rounded-lg border border-border bg-background"
                  >
                    {daysOfWeek.map((day, i) => (
                      <option key={i} value={i}>{day}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full mt-6 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 text-white font-medium disabled:opacity-50 hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <>
                  <Rocket className="w-5 h-5" />
                  Predict Virality
                </>
              )}
            </button>

            {error && (
              <p className="mt-4 text-sm text-red-500 text-center">{error}</p>
            )}
          </div>
        </div>

        {/* Results */}
        <div className="space-y-4">
          {prediction ? (
            <>
              {/* Viral Score */}
              <div className="p-6 rounded-2xl border border-border bg-card">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-green-500" />
                    Virality Score
                  </h3>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getTierColor(prediction.viral_tier)}`}>
                    {prediction.viral_tier}
                  </span>
                </div>

                <div className="text-center py-4">
                  <div className="text-6xl font-bold bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">
                    {prediction.viral_score.toFixed(0)}
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">
                    out of 100 • {(prediction.confidence * 100).toFixed(0)}% confidence
                  </p>

                  {/* Score bar */}
                  <div className="mt-6 h-4 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-1000"
                      style={{ width: `${prediction.viral_score}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Predicted Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <MetricCard
                  icon={<Eye className="w-5 h-5" />}
                  label="Views"
                  value={formatNumber(prediction.predicted_views)}
                  range={`${formatNumber(prediction.views_range[0])} - ${formatNumber(prediction.views_range[1])}`}
                  color="blue"
                />
                <MetricCard
                  icon={<Heart className="w-5 h-5" />}
                  label="Likes"
                  value={formatNumber(prediction.predicted_likes)}
                  range={`${formatNumber(prediction.likes_range[0])} - ${formatNumber(prediction.likes_range[1])}`}
                  color="red"
                />
                <MetricCard
                  icon={<Share2 className="w-5 h-5" />}
                  label="Shares"
                  value={formatNumber(prediction.predicted_shares)}
                  color="green"
                />
                <MetricCard
                  icon={<MessageCircle className="w-5 h-5" />}
                  label="Comments"
                  value={formatNumber(prediction.predicted_comments)}
                  color="orange"
                />
              </div>

              {/* Top Factors */}
              <div className="p-6 rounded-2xl border border-border bg-card">
                <h3 className="text-lg font-semibold mb-4">Impact Factors (SHAP)</h3>
                <div className="space-y-3">
                  {prediction.top_factors.slice(0, 5).map((factor) => (
                    <div key={factor.feature} className="flex items-center gap-3">
                      <span className="text-sm text-muted-foreground w-32 truncate">
                        {factor.feature.replace(/_/g, " ")}
                      </span>
                      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className={`h-full ${factor.direction === "positive" ? "bg-green-500" : "bg-red-500"}`}
                          style={{ width: `${Math.min(Math.abs(factor.impact) * 100, 100)}%` }}
                        />
                      </div>
                      <span className={`text-xs font-mono w-12 text-right ${factor.direction === "positive" ? "text-green-500" : "text-red-500"}`}>
                        {factor.impact > 0 ? "+" : ""}{factor.impact.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommendations */}
              <div className="p-6 rounded-2xl border border-border bg-card">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Lightbulb className="w-5 h-5 text-yellow-500" />
                  Recommendations
                </h3>
                <ul className="space-y-2">
                  {prediction.recommendations.map((rec, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <span className="text-primary mt-0.5">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            </>
          ) : (
            <div className="h-full flex items-center justify-center p-12 rounded-2xl border border-dashed border-border">
              <div className="text-center text-muted-foreground">
                <Rocket className="w-16 h-16 mx-auto mb-4 opacity-30" />
                <p className="text-lg font-medium">Ready to predict</p>
                <p className="text-sm">Fill in video details and click predict</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  range,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  range?: string;
  color: string;
}) {
  const colorClasses: Record<string, string> = {
    blue: "bg-blue-500/10 text-blue-500",
    red: "bg-red-500/10 text-red-500",
    green: "bg-green-500/10 text-green-500",
    orange: "bg-orange-500/10 text-orange-500",
  };

  return (
    <div className="p-4 rounded-xl border border-border bg-card">
      <div className={`w-10 h-10 rounded-lg ${colorClasses[color]} flex items-center justify-center mb-2`}>
        {icon}
      </div>
      <p className="text-2xl font-bold">{value}</p>
      <p className="text-sm text-muted-foreground">{label}</p>
      {range && <p className="text-xs text-muted-foreground mt-1">Range: {range}</p>}
    </div>
  );
}
