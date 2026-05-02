"use client";

import { Logo } from "./Logo";
import { SearchForm } from "./SearchForm";
import { useMovieSearch } from "./useMovieSearch";
import { Markdown } from "./Markdown";
import { MovieSource } from "./useMovieSearch";

function SourceCard({ source }: { source: MovieSource }) {
  return (
    <div className="border border-[#404040] rounded-lg p-3 bg-[#111] hover:bg-[#1a1a1a] transition-colors">
      <div className="font-bold text-white text-sm">{source.title}</div>
      <div className="text-xs text-[grey] mt-1">
        {source.year} · {source.genres}
      </div>
      <div className="text-xs text-[#888] mt-2 line-clamp-2">
        {source.overview}
      </div>
    </div>
  );
}

export default function Home() {
  const { messages, isLoading } = useMovieSearch();
  const hasMessages = messages.length > 0;

  return (
    <main
      className={`mx-auto w-full md:flex grid gap-4 ${
        !hasMessages ? "items-center h-screen" : "flex-col-reverse"
      }`}
    >
      <header
        className={`
        h-[155px] md:h-[122px] grid max-w-screen-lg mx-auto md:flex w-full items-center gap-4 md:gap-8 p-4 md:p-8 bg-black bg-opacity-80 backdrop-blur-lg`}
      >
        <Logo />
        <SearchForm shouldShowSuggestions={!hasMessages} />
      </header>

      {hasMessages && (
        <div className="row-start-1 p-4 md:p-8 w-full flex h-[calc(100vh-170px)] md:h-[calc(100vh-122px)] flex-col text-left overflow-auto max-w-screen-lg mx-auto gap-6">
          {messages.map((msg, index) => (
            <div key={index} className="flex flex-col gap-3">
              {msg.role === "user" ? (
                <div className="flex justify-end">
                  <div className="bg-[#222] text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-[80%]">
                    {msg.content}
                  </div>
                </div>
              ) : (
                <div className="flex flex-col gap-3">
                  <div className="prose prose-invert max-w-none">
                    <Markdown>{msg.content}</Markdown>
                  </div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div>
                      <div className="text-xs text-[grey] uppercase tracking-wider mb-2">
                        Sources
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {msg.sources.map((source, sidx) => (
                          <SourceCard key={sidx} source={source} />
                        ))}
                      </div>
                    </div>
                  )}
                  {msg.latency_ms && (
                    <div className="text-xs text-[#555]">
                      Response time: {msg.latency_ms.toFixed(0)}ms
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex items-center gap-3 text-[grey]">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span className="text-sm">Thinking...</span>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
