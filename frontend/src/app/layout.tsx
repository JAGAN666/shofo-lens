import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ShofoLens - Multimodal Video Intelligence",
  description: "Semantic search and engagement intelligence for TikTok video data",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
          {children}
        </div>
      </body>
    </html>
  );
}
