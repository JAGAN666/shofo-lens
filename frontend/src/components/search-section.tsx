"use client";

import { useState } from "react";
import {
  Search,
  Play,
  Heart,
  ExternalLink,
  Filter,
  Loader2,
  User,
  Clock,
} from "lucide-react";
import { searchVideos, SearchResult } from "@/lib/api";
import { formatNumber, formatDuration } from "@/lib/utils";

export function SearchSection() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    min_views: undefined as number | undefined,
    has_transcript: undefined as boolean | undefined,
  });

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setHasSearched(true);

    try {
      const response = await searchVideos(query, {
        limit: 20,
        ...filters,
      });
      setResults(response.results);
    } catch (error) {
      console.error("Search failed:", error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const exampleQueries = [
    "cooking tutorials with hand movements",
    "funny dance transitions",
    "product reviews and unboxing",
    "life hacks and tips",
    "cute pet videos",
  ];

  return (
    <div className="space-y-6">
      {/* Search Input */}
      <div className="max-w-3xl mx-auto">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search videos by content, topic, or description..."
            className="w-full pl-12 pr-32 py-4 rounded-2xl border border-border bg-card text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 text-lg"
          />
          <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`p-2 rounded-lg transition-colors ${
                showFilters
                  ? "bg-primary/20 text-primary"
                  : "hover:bg-muted text-muted-foreground"
              }`}
            >
              <Filter className="w-5 h-5" />
            </button>
            <button
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              className="px-6 py-2 rounded-xl gradient-bg text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                "Search"
              )}
            </button>
          </div>
        </div>

        {/* Filters */}
        {showFilters && (
          <div className="mt-4 p-4 rounded-xl border border-border bg-card/50 flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm text-muted-foreground">Min views:</label>
              <select
                value={filters.min_views || ""}
                onChange={(e) =>
                  setFilters({
                    ...filters,
                    min_views: e.target.value ? parseInt(e.target.value) : undefined,
                  })
                }
                className="px-3 py-1.5 rounded-lg border border-border bg-background text-sm"
              >
                <option value="">Any</option>
                <option value="1000">1K+</option>
                <option value="10000">10K+</option>
                <option value="100000">100K+</option>
                <option value="1000000">1M+</option>
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-muted-foreground">Has transcript:</label>
              <select
                value={
                  filters.has_transcript === undefined
                    ? ""
                    : filters.has_transcript.toString()
                }
                onChange={(e) =>
                  setFilters({
                    ...filters,
                    has_transcript:
                      e.target.value === "" ? undefined : e.target.value === "true",
                  })
                }
                className="px-3 py-1.5 rounded-lg border border-border bg-background text-sm"
              >
                <option value="">Any</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
            </div>
          </div>
        )}

        {/* Example queries */}
        {!hasSearched && (
          <div className="mt-6">
            <p className="text-sm text-muted-foreground mb-3">Try searching for:</p>
            <div className="flex flex-wrap gap-2">
              {exampleQueries.map((example) => (
                <button
                  key={example}
                  onClick={() => {
                    setQuery(example);
                    setTimeout(() => {
                      setLoading(true);
                      setHasSearched(true);
                      searchVideos(example, { limit: 20 })
                        .then((response) => setResults(response.results))
                        .catch(() => setResults([]))
                        .finally(() => setLoading(false));
                    }, 0);
                  }}
                  className="px-4 py-2 rounded-full border border-border hover:border-primary hover:text-primary transition-colors text-sm"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      {hasSearched && (
        <div className="mt-8">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
          ) : results.length > 0 ? (
            <>
              <p className="text-sm text-muted-foreground mb-4">
                Found {results.length} results for "{query}"
              </p>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {results.map((result) => (
                  <VideoCard key={result.video_id} video={result} />
                ))}
              </div>
            </>
          ) : (
            <div className="text-center py-12">
              <p className="text-muted-foreground">No results found for "{query}"</p>
              <p className="text-sm text-muted-foreground mt-1">
                Try a different search term or adjust your filters
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function VideoCard({ video }: { video: SearchResult }) {
  const similarityPercent = Math.round(video.score * 100);

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden hover:border-primary/50 transition-colors group">
      {/* Similarity score badge */}
      <div className="px-4 py-3 border-b border-border/50 flex items-center justify-between bg-muted/30">
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{
              backgroundColor: `hsl(${Math.round(video.score * 120)}, 80%, 50%)`,
            }}
          />
          <span className="text-sm font-medium">{similarityPercent}% match</span>
        </div>
        {video.web_url && (
          <a
            href={video.web_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-primary transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        )}
      </div>

      <div className="p-4 space-y-3">
        {/* Creator */}
        {video.creator && (
          <div className="flex items-center gap-2 text-sm">
            <User className="w-4 h-4 text-muted-foreground" />
            <span className="text-muted-foreground">@{video.creator}</span>
          </div>
        )}

        {/* Description */}
        {video.description && (
          <p className="text-sm line-clamp-3">{video.description}</p>
        )}

        {/* Transcript preview */}
        {video.transcript && (
          <div className="p-2 rounded-lg bg-muted/50 text-xs text-muted-foreground line-clamp-2">
            "{video.transcript}"
          </div>
        )}

        {/* Stats */}
        <div className="flex items-center gap-4 pt-2 text-sm text-muted-foreground">
          {video.play_count !== null && (
            <div className="flex items-center gap-1">
              <Play className="w-3.5 h-3.5" />
              {formatNumber(video.play_count)}
            </div>
          )}
          {video.like_count !== null && (
            <div className="flex items-center gap-1">
              <Heart className="w-3.5 h-3.5" />
              {formatNumber(video.like_count)}
            </div>
          )}
          {video.duration_ms !== null && (
            <div className="flex items-center gap-1">
              <Clock className="w-3.5 h-3.5" />
              {formatDuration(video.duration_ms)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
