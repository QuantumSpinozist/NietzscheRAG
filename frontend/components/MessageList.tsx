"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Message } from "@/types";
import SourceCard from "./SourceCard";

function AssistantMessage({ message }: { message: Message }) {
  const [showSources, setShowSources] = useState(false);
  const hasSources = message.sources && message.sources.length > 0;

  const sortedSources = hasSources
    ? [...message.sources!].sort((a, b) => Number(b.used) - Number(a.used))
    : [];
  const citedCount = sortedSources.filter((s) => s.used).length;
  const totalCount = sortedSources.length;

  const toggleLabel = showSources
    ? "▲ hide sources"
    : citedCount === totalCount
    ? `▼ ${citedCount} cited`
    : `▼ ${citedCount} cited · ${totalCount - citedCount} more`;

  return (
    <div className="space-y-3">
      <div className="prose-nietzsche">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {message.content}
        </ReactMarkdown>
      </div>

      {hasSources && (
        <div>
          <button
            onClick={() => setShowSources((s) => !s)}
            className="text-xs text-[var(--muted)] hover:text-[var(--accent)] transition-colors font-mono-ui tracking-wide uppercase"
          >
            {toggleLabel}
          </button>

          {showSources && (
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {sortedSources.map((src, i) => (
                <SourceCard key={i} source={src} used={src.used} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function MessageList({ messages }: MessageListProps) {
  return (
    <div className="space-y-8">
      {messages.map((msg) => (
        <div key={msg.id} className={msg.role === "user" ? "flex justify-end" : ""}>
          {msg.role === "user" ? (
            <div className="max-w-xl px-4 py-3 rounded bg-[var(--surface-2)] border border-[var(--border)] text-[var(--foreground)] leading-relaxed">
              {msg.content}
            </div>
          ) : (
            <div className="border-l-2 border-[var(--accent-dim)] pl-4">
              <AssistantMessage message={msg} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

interface MessageListProps {
  messages: Message[];
}
