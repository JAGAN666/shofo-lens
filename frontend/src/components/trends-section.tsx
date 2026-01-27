"use client";

import { useState, useEffect } from "react";
import {
  TrendingUp,
  TrendingDown,
  Flame,
  Loader2,
  Clock,
  Hash,
  Sparkles,
  ArrowUp,
  ArrowDown,
  Minus,
} from "lucide-react";
import { getTrends, TrendResponse, TrendItem } from "@/lib/api";
import { formatNumber } from "@/lib/utils";

export function TrendsSection() {
  const [loading, setLoading] = useState(true);
  const [trends, setTrends] = useState<TrendResponse | null>(null);
  const [activeTab, setActiveTab] = useState<"trending" | "emerging" | "declining">("trending");

  useEffect(() => {
    getTrends()
      .then(setTrends)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!trends) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <p>Unable to load trends. Make sure the backend is running.</p>
      </div>
    );
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "Rising":
        return <ArrowUp className="w-4 h-4 text-green-500" />;
      case "Hot":
        return <Flame className="w-4 h-4 text-orange-500" />;
      case "Declining":
        return <ArrowDown className="w-4 h-4 text-red-500" />;
      default:
        return <Minus className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Rising":
        return "bg-green-500/10 text-green-500";
      case "Hot":
        return "bg-orange-500/10 text-orange-500";
      case "Emerging":
        return "bg-purple-500/10 text-purple-500";
      case "Declining":
        return "bg-red-500/10 text-red-500";
      default:
        return "bg-gray-500/10 text-gray-500";
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-orange-500/20 to-red-500/20 text-orange-400 mb-4">
          <Flame className="w-4 h-4" />
          <span className="text-sm font-medium">Real-time Analysis</span>
        </div>
        <h2 className="text-2xl font-bold mb-2">Trend Detection</h2>
        <p className="text-muted-foreground">
          Discover what's trending, emerging, and declining in the dataset
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex justify-center gap-2">
        {[
          { id: "trending", label: "Trending", icon: <Flame className="w-4 h-4" />, count: trends.trending_hashtags.length },
          { id: "emerging", label: "Emerging", icon: <Sparkles className="w-4 h-4" />, count: trends.emerging_hashtags.length },
          { id: "declining", label: "Declining", icon: <TrendingDown className="w-4 h-4" />, count: trends.declining_hashtags.length },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${
              activeTab === tab.id
                ? "bg-primary text-white"
                : "bg-muted hover:bg-muted/80 text-muted-foreground"
            }`}
          >
            {tab.icon}
            {tab.label}
            <span className="text-xs opacity-70">({tab.count})</span>
          </button>
        ))}
      </div>

      {/* Content Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Hashtag List */}
        <div className="lg:col-span-2 p-6 rounded-2xl border border-border bg-card">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Hash className="w-5 h-5 text-primary" />
            {activeTab === "trending" && "Trending Hashtags"}
            {activeTab === "emerging" && "Emerging Hashtags"}
            {activeTab === "declining" && "Declining Hashtags"}
          </h3>

          <div className="space-y-3">
            {(activeTab === "trending"
              ? trends.trending_hashtags
              : activeTab === "emerging"
              ? trends.emerging_hashtags
              : trends.declining_hashtags
            )
              .slice(0, 12)
              .map((hashtag, index) => (
                <HashtagRow key={hashtag.name} hashtag={hashtag} rank={index + 1} />
              ))}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Best Posting Times */}
          <div className="p-6 rounded-2xl border border-border bg-card">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5 text-blue-500" />
              Best Posting Times
            </h3>
            <div className="space-y-2">
              {trends.best_posting_times.slice(0, 5).map((time, i) => (
                <div
                  key={time.hour}
                  className="flex items-center justify-between p-2 rounded-lg bg-muted/50"
                >
                  <div className="flex items-center gap-2">
                    <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                      i === 0 ? "bg-yellow-500/20 text-yellow-500" : "bg-muted text-muted-foreground"
                    }`}>
                      {i + 1}
                    </span>
                    <span className="font-mono">{time.formatted}</span>
                  </div>
                  <span className="text-sm text-green-500">
                    {time.avg_engagement.toFixed(1)}% eng
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Trending Topics */}
          <div className="p-6 rounded-2xl border border-border bg-card">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-green-500" />
              Trending Topics
            </h3>
            <div className="flex flex-wrap gap-2">
              {trends.trending_topics.slice(0, 8).map((topic) => (
                <span
                  key={topic}
                  className="px-3 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-medium"
                >
                  {topic}
                </span>
              ))}
            </div>
          </div>

          {/* Engagement Insights */}
          <div className="p-6 rounded-2xl border border-border bg-card">
            <h3 className="text-lg font-semibold mb-4">Dataset Insights</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Videos</span>
                <span className="font-medium">{formatNumber(trends.engagement_insights.total_videos || 0)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Views</span>
                <span className="font-medium">{formatNumber(trends.engagement_insights.total_views || 0)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Avg Engagement</span>
                <span className="font-medium text-green-500">
                  {(trends.engagement_insights.avg_engagement_rate || 0).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function HashtagRow({ hashtag, rank }: { hashtag: TrendItem; rank: number }) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "Rising": return "text-green-500 bg-green-500/10";
      case "Hot": return "text-orange-500 bg-orange-500/10";
      case "Emerging": return "text-purple-500 bg-purple-500/10";
      case "Declining": return "text-red-500 bg-red-500/10";
      default: return "text-gray-500 bg-gray-500/10";
    }
  };

  return (
    <div className="flex items-center gap-4 p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors">
      <span className="w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center text-sm font-bold">
        {rank}
      </span>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium truncate">#{hashtag.name}</span>
          <span className={`px-2 py-0.5 rounded-full text-xs ${getStatusColor(hashtag.trend_status)}`}>
            {hashtag.trend_status}
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
          <span>{formatNumber(hashtag.count)} videos</span>
          <span>•</span>
          <span>{hashtag.avg_engagement.toFixed(1)}% engagement</span>
        </div>
      </div>

      <div className="text-right">
        <div className={`text-sm font-medium ${hashtag.velocity > 0 ? "text-green-500" : "text-red-500"}`}>
          {hashtag.velocity > 0 ? "+" : ""}{(hashtag.velocity * 100).toFixed(0)}%
        </div>
        <div className="text-xs text-muted-foreground">velocity</div>
      </div>
    </div>
  );
}
