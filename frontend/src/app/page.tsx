"use client";

import { useState } from "react";
import {
  Search,
  Sparkles,
  BarChart3,
  Rocket,
  Database,
  Brain,
  TrendingUp,
  Flame,
  Layers,
  Zap,
} from "lucide-react";
import { SearchSection } from "@/components/search-section";
import { ViralitySection } from "@/components/virality-section";
import { PredictSection } from "@/components/predict-section";
import { TrendsSection } from "@/components/trends-section";
import { AnalyticsSection } from "@/components/analytics-section";

type Tab = "search" | "virality" | "predict" | "trends" | "analytics";

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("search");

  const tabs = [
    { id: "search" as Tab, label: "Semantic Search", icon: Search },
    { id: "virality" as Tab, label: "Virality Predictor", icon: Rocket },
    { id: "predict" as Tab, label: "Engagement", icon: Zap },
    { id: "trends" as Tab, label: "Trends", icon: Flame },
    { id: "analytics" as Tab, label: "Analytics", icon: BarChart3 },
  ];

  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm bg-background/80 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-pink-600 flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">
                  ShofoLens
                </h1>
                <p className="text-xs text-muted-foreground">v2.0 SPECTACULAR Edition</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <span className="hidden md:flex items-center gap-1 text-xs text-muted-foreground bg-muted px-2 py-1 rounded-full">
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                Live Demo
              </span>
              <a
                href="https://huggingface.co/datasets/Shofo/shofo-tiktok-general-small"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
              >
                <Database className="w-4 h-4" />
                <span className="hidden sm:inline">Dataset</span>
              </a>
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                API
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-12 border-b border-border/50 bg-gradient-to-b from-purple-500/5 to-transparent">
        <div className="container mx-auto px-4 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-purple-500/30 bg-purple-500/10 text-purple-400 text-sm mb-6">
            <Sparkles className="w-4 h-4" />
            Built for Shofo.ai
          </div>

          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Video Intelligence{" "}
            <span className="bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">
              Reimagined
            </span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
            Advanced ML-powered semantic search, virality prediction, trend detection,
            and content analytics on 58,000+ TikTok videos.
          </p>

          {/* Feature pills */}
          <div className="flex flex-wrap justify-center gap-3">
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-purple-500/10 text-purple-400 text-sm font-medium">
              <Brain className="w-4 h-4" />
              Semantic Search
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-pink-500/10 text-pink-400 text-sm font-medium">
              <Rocket className="w-4 h-4" />
              Virality Prediction
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-orange-500/10 text-orange-400 text-sm font-medium">
              <TrendingUp className="w-4 h-4" />
              Trend Detection
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-400 text-sm font-medium">
              <Layers className="w-4 h-4" />
              Auto Classification
            </div>
          </div>
        </div>
      </section>

      {/* Tab Navigation */}
      <nav className="border-b border-border/50 sticky top-[73px] z-40 bg-background/80 backdrop-blur-sm">
        <div className="container mx-auto px-4">
          <div className="flex gap-1 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 md:px-6 py-4 font-medium transition-all relative whitespace-nowrap ${
                  activeTab === tab.id
                    ? "text-primary"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <div className="flex items-center gap-2">
                  <tab.icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </div>
                {activeTab === tab.id && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-purple-500 to-pink-500" />
                )}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        {activeTab === "search" && <SearchSection />}
        {activeTab === "virality" && <ViralitySection />}
        {activeTab === "predict" && <PredictSection />}
        {activeTab === "trends" && <TrendsSection />}
        {activeTab === "analytics" && <AnalyticsSection />}
      </div>

      {/* Footer */}
      <footer className="border-t border-border/50 py-8 mt-12 bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-600 to-pink-600 flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div>
                <p className="font-medium">ShofoLens v2.0</p>
                <p className="text-xs text-muted-foreground">Multimodal Video Intelligence Platform</p>
              </div>
            </div>

            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <a
                href="https://huggingface.co/datasets/Shofo/shofo-tiktok-general-small"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-foreground transition-colors"
              >
                Dataset
              </a>
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-foreground transition-colors"
              >
                API Docs
              </a>
              <a
                href="https://www.shofo.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-foreground transition-colors"
              >
                Shofo.ai
              </a>
            </div>
          </div>

          <div className="mt-6 pt-6 border-t border-border/50 text-center text-xs text-muted-foreground">
            <p>
              Built with FastAPI, Next.js, sentence-transformers, XGBoost, BERTopic, and Qdrant
            </p>
            <p className="mt-1">
              Demonstrating advanced ML capabilities for video data intelligence
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
