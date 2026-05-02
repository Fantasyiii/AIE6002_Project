import type { Metadata } from "next";
import "./globals.css";
import { sora } from "./fonts";

export const metadata: Metadata = {
  title: "VibeMatch - AI Movie Recommendations",
  description: "RAG-powered semantic movie recommendation system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${sora.className} bg-black flex items-center text-white`}
      >
        {children}
      </body>
    </html>
  );
}
