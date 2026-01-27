"use client";

import { useState, useEffect } from "react";
import {
  Zap,
  Loader2,
  TrendingUp,
  Lightbulb,
  Hash,
  Clock,
  FileText,
  Info,
} from "lucide-react";
import { predictEngagement, getFeatureImportance, PredictionResponse } from "@/lib/api";

export function PredictSection() {
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
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [featureImportance, setFeatureImportance] = useState<Record<string, number>>({});

  useEffect(() => {
    getFeatureImportance()
      .then((data) => setFeatureImportance(data.feature_importance))
      .catch(console.error);
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const result = await predictEngagement({
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
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const daysOfWeek = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

  return (
    <div className="max-w-4xl mx-auto">
      <div className="grid md:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="space-y-6">
          <div className="p-6 rounded-2xl border border-border bg-card">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <FileText className="w-5 h-5 text-primary" />
              Video Details
            </h3>

            <div className="space-y-4">
              {/* Duration */}
              <div>
                <label className="text-sm font-medium mb-2 block">
                  Duration (seconds)
                </label>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    min="5"
                    max="180"
                    value={formData.duration_ms / 1000}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        duration_ms: parseInt(e.target.value) * 1000,
                      })
                    }
                    className="flex-1 accent-primary"
                  />
                  <span className="text-sm font-mono w-12">
                    {Math.round(formData.duration_ms / 1000)}s
                  </span>
                </div>
              </div>

              {/* Description */}
              <div>
                <label className="text-sm font-medium mb-2 block">Description</label>
                <textarea
                  value={formData.description}
                  onChange={(e) =>
                    setFormData({ ...formData, description: e.target.value })
                  }
                  placeholder="Video caption or description..."
                  rows={3}
                  className="w-full px-3 py-2 rounded-lg border border-border bg-background resize-none focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>

              {/* Transcript */}
              <div>
                <label className="text-sm font-medium mb-2 block">
                  Transcript/Speech
                </label>
                <textarea
                  value={formData.transcript}
                  onChange={(e) =>
                    setFormData({ ...formData, transcript: e.target.value })
                  }
                  placeholder="What is said in the video..."
                  rows={3}
                  className="w-full px-3 py-2 rounded-lg border border-border bg-background resize-none focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>

              {/* Hashtags */}
              <div>
                <label className="text-sm font-medium mb-2 block flex items-center gap-2">
                  <Hash className="w-4 h-4" />
                  Hashtags
                </label>
                <input
                  type="text"
                  value={formData.hashtags}
                  onChange={(e) =>
                    setFormData({ ...formData, hashtags: e.target.value })
                  }
                  placeholder="fyp, viral, trending (comma-separated)"
                  className="w-full px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>

              {/* Time */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    Hour Posted
                  </label>
                  <select
                    value={formData.hour_posted}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        hour_posted: parseInt(e.target.value),
                      })
                    }
                    className="w-full px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary/50"
                  >
                    {Array.from({ length: 24 }, (_, i) => (
                      <option key={i} value={i}>
                        {i.toString().padStart(2, "0")}:00
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Day of Week</label>
                  <select
                    value={formData.day_of_week}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        day_of_week: parseInt(e.target.value),
                      })
                    }
                    className="w-full px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary/50"
                  >
                    {daysOfWeek.map((day, i) => (
                      <option key={i} value={i}>
                        {day}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Flags */}
              <div className="flex gap-6">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formData.is_ai_generated}
                    onChange={(e) =>
                      setFormData({ ...formData, is_ai_generated: e.target.checked })
                    }
                    className="rounded accent-primary"
                  />
                  <span className="text-sm">AI Generated</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formData.is_ad}
                    onChange={(e) =>
                      setFormData({ ...formData, is_ad: e.target.checked })
                    }
                    className="rounded accent-primary"
                  />
                  <span className="text-sm">Advertisement</span>
                </label>
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full mt-6 py-3 rounded-xl gradient-bg text-white font-medium disabled:opacity-50 hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  Predict Engagement
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="space-y-6">
          {prediction ? (
            <>
              {/* Score */}
              <div className="p-6 rounded-2xl border border-border bg-card">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-500" />
                  Engagement Prediction
                </h3>

                <div className="text-center py-4">
                  <div className="text-5xl font-bold gradient-text mb-2">
                    {(prediction.engagement_score * 100).toFixed(1)}%
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Predicted engagement rate
                  </p>

                  {/* Score bar */}
                  <div className="mt-6 h-3 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full gradient-bg transition-all duration-500"
                      style={{
                        width: `${Math.min(prediction.engagement_score * 100, 100)}%`,
                      }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>Low</span>
                    <span>Average</span>
                    <span>High</span>
                  </div>
                </div>
              </div>

              {/* Feature Impact */}
              <div className="p-6 rounded-2xl border border-border bg-card">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Info className="w-5 h-5 text-blue-500" />
                  Feature Impact (SHAP)
                </h3>

                <div className="space-y-3">
                  {Object.entries(prediction.feature_importance)
                    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                    .slice(0, 6)
                    .map(([feature, impact]) => (
                      <div key={feature} className="flex items-center gap-3">
                        <span className="text-sm text-muted-foreground w-32 truncate">
                          {feature.replace(/_/g, " ")}
                        </span>
                        <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                          <div
                            className={`h-full ${
                              impact > 0 ? "bg-green-500" : "bg-red-500"
                            }`}
                            style={{
                              width: `${Math.min(Math.abs(impact) * 100, 100)}%`,
                              marginLeft: impact < 0 ? "auto" : 0,
                            }}
                          />
                        </div>
                        <span
                          className={`text-xs font-mono w-12 text-right ${
                            impact > 0 ? "text-green-500" : "text-red-500"
                          }`}
                        >
                          {impact > 0 ? "+" : ""}
                          {impact.toFixed(2)}
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
                    <li
                      key={i}
                      className="flex items-start gap-2 text-sm text-muted-foreground"
                    >
                      <span className="text-primary">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            </>
          ) : (
            <div className="h-full flex items-center justify-center p-12 rounded-2xl border border-dashed border-border">
              <div className="text-center text-muted-foreground">
                <Zap className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Fill in video details and click predict to see engagement estimates</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Global Feature Importance */}
      {Object.keys(featureImportance).length > 0 && (
        <div className="mt-8 p-6 rounded-2xl border border-border bg-card">
          <h3 className="text-lg font-semibold mb-4">
            Global Feature Importance (Model-wide)
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            {Object.entries(featureImportance)
              .sort((a, b) => b[1] - a[1])
              .map(([feature, importance]) => (
                <div key={feature} className="flex items-center gap-3">
                  <span className="text-sm text-muted-foreground w-40 truncate">
                    {feature.replace(/_/g, " ")}
                  </span>
                  <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary"
                      style={{ width: `${importance * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-muted-foreground w-12">
                    {(importance * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
