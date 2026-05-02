"use client";

import { jetbrainsMono } from "./fonts";
import { useRef, useState } from "react";
import { useMovieSearch } from "./useMovieSearch";

type Props = {
  shouldShowSuggestions: boolean;
};

export function SearchForm({ shouldShowSuggestions }: Props) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const { search, isLoading } = useMovieSearch();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    search(input).then(() => {
      setInput("");
      inputRef.current?.focus();
    });
  };

  return (
    <form className="grid gap-4 w-full" onSubmit={handleSubmit}>
      <div className="relative">
        <input
          type="text"
          ref={inputRef}
          autoFocus
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
          className={`${jetbrainsMono.className} focus:text-white focus:border-white text-[grey] w-full p-4 pr-12 rounded-lg bg-black border border-[#404040] disabled:opacity-50`}
          placeholder="Describe the vibe you're looking for..."
        />
        {isLoading && (
          <div className="absolute right-4 top-1/2 -translate-y-1/2">
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          </div>
        )}
      </div>
      {shouldShowSuggestions && (
        <div className="text-sm text-[grey]">
          Try: "sci-fi about time travel" or "romantic comedy for weekend"
        </div>
      )}
    </form>
  );
}
