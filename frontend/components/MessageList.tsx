"use client";

import { useState } from "react";
import { Message } from "@/types";
import SourceCard from "./SourceCard";

function AssistantMessage({ message }: { message: Message }) {
  const [showSources, setShowSources] = useState(false);
  const hasSources = message.sources && message.sources.length > 0;

  return (
    <div className="space-y-3">
      <div className="text-[var(--foreground)] leading-relaxed whitespace-pre-wrap">
        {message.content}
      </div>

      {hasSources && (
        <div>
          <button
            onClick={() => setShowSources((s) => !s)}
            className="text-xs text-[var(--muted)] hover:text-[var(--accent)] transition-colors font-mono-ui tracking-wide uppercase"
          >
            {showSources
              ? "▲ hide sources"
              : `▼ ${message.sources!.length} source${message.sources!.length > 1 ? "s" : ""}`}
          </button>

          {showSources && (
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {message.sources!.map((src, i) => (
                <SourceCard key={i} source={src} />
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
