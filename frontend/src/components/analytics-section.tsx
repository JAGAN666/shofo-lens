"use client";

import { useState, useEffect } from "react";
import {
  BarChart3,
  Play,
  Heart,
  Clock,
  Hash,
  Globe,
  Loader2,
  TrendingUp,
  Users,
  FileText,
  Bot,
  Megaphone,
} from "lucide-react";
import {
  getAnalyticsOverview,
  getTopHashtags,
  getLanguageDistribution,
  getDurationDistribution,
  AnalyticsOverview,
  HashtagData,
  LanguageData,
  DurationBucket,
} from "@/lib/api";
import { formatNumber } from "@/lib/utils";

export function AnalyticsSection() {
  const [loading, setLoading] = useState(true);
  const [overview, setOverview] = useState<AnalyticsOverview | null>(null);
  const [hashtags, setHashtags] = useState<HashtagData[]>([]);
  const [languages, setLanguages] = useState<LanguageData[]>([]);
  const [durations, setDurations] = useState<DurationBucket[]>([]);

  useEffect(() => {
    Promise.all([
      getAnalyticsOverview(),
      getTopHashtags(15),
      getLanguageDistribution(10),
      getDurationDistribution(),
    ])
      .then(([overviewData, hashtagData, languageData, durationData]) => {
        setOverview(overviewData);
        setHashtags(hashtagData.hashtags);
        setLanguages(languageData.languages);
        setDurations(durationData.distribution);
      })
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

  if (!overview) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <p>Unable to load analytics. Make sure the backend is running.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Overview Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={<Play className="w-5 h-5" />}
          label="Total Videos"
          value={formatNumber(overview.total_videos)}
          color="purple"
        />
        <StatCard
          icon={<TrendingUp className="w-5 h-5" />}
          label="Total Views"
          value={formatNumber(overview.total_views)}
          color="blue"
        />
        <StatCard
          icon={<Heart className="w-5 h-5" />}
          label="Total Likes"
          value={formatNumber(overview.total_likes)}
          color="red"
        />
        <StatCard
          icon={<Clock className="w-5 h-5" />}
          label="Avg Duration"
          value={`${overview.avg_duration_sec.toFixed(1)}s`}
          color="green"
        />
      </div>

      {/* Secondary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={<FileText className="w-5 h-5" />}
          label="With Transcript"
          value={`${overview.transcript_percentage}%`}
          subtext={`${formatNumber(overview.videos_with_transcript)} videos`}
          color="orange"
        />
        <StatCard
          icon={<Bot className="w-5 h-5" />}
          label="AI Generated"
          value={formatNumber(overview.ai_generated_videos)}
          color="cyan"
        />
        <StatCard
          icon={<Megaphone className="w-5 h-5" />}
          label="Advertisements"
          value={formatNumber(overview.ad_videos)}
          color="pink"
        />
        <StatCard
          icon={<Users className="w-5 h-5" />}
          label="Engagement Rate"
          value={`${((overview.total_likes / overview.total_views) * 100).toFixed(2)}%`}
          color="emerald"
        />
      </div>

      {/* Charts Row */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Top Hashtags */}
        <div className="p-6 rounded-2xl border border-border bg-card">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Hash className="w-5 h-5 text-primary" />
            Top Hashtags
          </h3>
          <div className="space-y-3">
            {hashtags.slice(0, 10).map((tag, index) => (
              <div key={tag.tag} className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground w-6">
                  #{index + 1}
                </span>
                <span className="text-sm font-medium flex-1 truncate">
                  #{tag.tag}
                </span>
                <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full gradient-bg"
                    style={{
                      width: `${(tag.count / hashtags[0].count) * 100}%`,
                    }}
                  />
                </div>
                <span className="text-xs text-muted-foreground w-16 text-right">
                  {formatNumber(tag.count)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Language Distribution */}
        <div className="p-6 rounded-2xl border border-border bg-card">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Globe className="w-5 h-5 text-blue-500" />
            Language Distribution
          </h3>
          <div className="space-y-3">
            {languages.map((lang) => (
              <div key={lang.code} className="flex items-center gap-3">
                <span className="text-sm font-mono uppercase w-8">
                  {lang.code}
                </span>
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500"
                    style={{ width: `${lang.percentage}%` }}
                  />
                </div>
                <span className="text-xs text-muted-foreground w-16 text-right">
                  {lang.percentage}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Duration Distribution */}
      <div className="p-6 rounded-2xl border border-border bg-card">
        <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-green-500" />
          Duration Distribution
        </h3>
        <div className="flex items-end justify-between gap-4 h-48">
          {durations.map((bucket) => (
            <div
              key={bucket.bucket}
              className="flex-1 flex flex-col items-center gap-2"
            >
              <div className="w-full flex flex-col items-center">
                <span className="text-xs text-muted-foreground mb-1">
                  {bucket.percentage}%
                </span>
                <div
                  className="w-full max-w-16 gradient-bg rounded-t-lg transition-all duration-500"
                  style={{ height: `${Math.max(bucket.percentage * 2, 8)}px` }}
                />
              </div>
              <span className="text-xs text-muted-foreground">{bucket.bucket}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Data Info */}
      <div className="p-4 rounded-xl bg-muted/50 border border-border">
        <p className="text-sm text-muted-foreground text-center">
          Analytics based on{" "}
          <span className="font-semibold text-foreground">
            {formatNumber(overview.total_videos)} videos
          </span>{" "}
          from the{" "}
          <a
            href="https://huggingface.co/datasets/Shofo/shofo-tiktok-general-small"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            Shofo TikTok Dataset
          </a>
        </p>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  subtext,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  subtext?: string;
  color: string;
}) {
  const colorClasses: Record<string, string> = {
    purple: "bg-purple-500/10 text-purple-500",
    blue: "bg-blue-500/10 text-blue-500",
    red: "bg-red-500/10 text-red-500",
    green: "bg-green-500/10 text-green-500",
    orange: "bg-orange-500/10 text-orange-500",
    cyan: "bg-cyan-500/10 text-cyan-500",
    pink: "bg-pink-500/10 text-pink-500",
    emerald: "bg-emerald-500/10 text-emerald-500",
  };

  return (
    <div className="p-4 rounded-xl border border-border bg-card hover:border-primary/30 transition-colors">
      <div className={`w-10 h-10 rounded-lg ${colorClasses[color]} flex items-center justify-center mb-3`}>
        {icon}
      </div>
      <p className="text-2xl font-bold">{value}</p>
      <p className="text-sm text-muted-foreground">{label}</p>
      {subtext && <p className="text-xs text-muted-foreground mt-1">{subtext}</p>}
    </div>
  );
}
