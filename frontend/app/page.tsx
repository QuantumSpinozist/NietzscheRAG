"use client";

import { useState, useRef, useEffect } from "react";
import { Message, Period, QueryResponse } from "@/types";
import ChatInput from "@/components/ChatInput";
import FilterBar from "@/components/FilterBar";
import MessageList from "@/components/MessageList";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState<Period>(null);
  const [slug, setSlug] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit() {
    const question = input.trim();
    if (!question || loading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          filter_period: period,
          filter_slug: slug,
        }),
      });

      const data: QueryResponse = await res.json();

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
        sources: data.sources,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "Failed to reach the server. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col max-w-3xl mx-auto px-4 py-12">
      {/* Header */}
      <header className="mb-10 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-[var(--accent)] mb-1">
          Nietzsche
        </h1>
        <p className="text-sm text-[var(--muted)] tracking-widest uppercase font-mono-ui">
          Retrieval-Augmented Corpus
        </p>
        <div className="mt-1 h-px bg-[var(--border)] w-24 mx-auto" />
      </header>

      {/* Empty state */}
      {messages.length === 0 && (
        <div className="flex-1 flex items-center justify-center text-center mb-8">
          <div className="space-y-2 text-[var(--muted)]">
            <p className="text-lg italic">&ldquo;God is dead. God remains dead. And we have killed him.&rdquo;</p>
            <p className="text-xs font-mono-ui tracking-wide">— The Gay Science, §125</p>
            <p className="mt-6 text-sm">Ask anything about Nietzsche&apos;s philosophy.</p>
          </div>
        </div>
      )}

      {/* Messages */}
      {messages.length > 0 && (
        <div className="flex-1 mb-8">
          <MessageList messages={messages} />
          {loading && (
            <div className="mt-8 border-l-2 border-[var(--accent-dim)] pl-4">
              <span className="text-[var(--muted)] italic text-sm animate-pulse">
                Searching the corpus…
              </span>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      )}

      {/* Input area */}
      <div className="mt-auto space-y-3">
        <FilterBar
          period={period}
          slug={slug}
          onPeriodChange={setPeriod}
          onSlugChange={setSlug}
          disabled={loading}
        />
        <ChatInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          loading={loading}
        />
        <p className="text-xs text-[var(--muted)] text-center font-mono-ui">
          Shift+Enter for newline · Enter to submit
        </p>
      </div>
    </div>
  );
}
