"use client";

import { LangProvider } from "./LangContext";

export function Providers({ children }: { children: React.ReactNode }) {
  return <LangProvider>{children}</LangProvider>;
}
