"use client";

import { useState, useCallback } from "react";

export type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: MovieSource[];
  latency_ms?: number;
};

export type MovieSource = {
  title: string;
  year: string;
  genres: string;
  overview: string;
};

export type ChatResponse = {
  answer: string;
  sources: MovieSource[];
  retrieval_mode: string;
  latency_ms?: number;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useMovieSearch() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const search = useCallback(async (prompt: string) => {
    if (!prompt.trim()) return;

    const userMessage: Message = { role: "user", content: prompt };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: prompt,
          retrieval_mode: "similarity",
          top_k: 5,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ChatResponse = await response.json();

      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        latency_ms: data.latency_ms,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Search error:", error);
      const errorMessage: Message = {
        role: "assistant",
        content: "Sorry, I encountered an error while searching for movies. Please make sure the backend server is running.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isLoading, search, clearMessages };
}
