"use client";

import { jetbrainsMono } from "./fonts";
import { useRef, useState } from "react";
import { Translations } from "./i18n";

type Props = {
  shouldShowSuggestions: boolean;
  onSearch: (query: string) => void;
  isLoading: boolean;
  t: Translations;
};

export function SearchForm({ shouldShowSuggestions, onSearch, isLoading, t }: Props) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSearch(input);
    setInput("");
    inputRef.current?.focus();
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
          placeholder={t.placeholder}
        />
        {isLoading && (
          <div className="absolute right-4 top-1/2 -translate-y-1/2">
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          </div>
        )}
      </div>
      {shouldShowSuggestions && (
        <div className="text-sm text-[grey]">{t.suggestions}</div>
      )}
    </form>
  );
}
