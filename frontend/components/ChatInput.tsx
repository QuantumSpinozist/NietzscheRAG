"use client";

import { useRef, KeyboardEvent } from "react";

interface ChatInputProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  loading: boolean;
}

export default function ChatInput({ value, onChange, onSubmit, loading }: ChatInputProps) {
  const ref = useRef<HTMLTextAreaElement>(null);

  function handleKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!loading && value.trim()) onSubmit();
    }
  }

  return (
    <div className="flex gap-3 items-end">
      <textarea
        ref={ref}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKey}
        placeholder="Ask a question about Nietzsche…"
        rows={1}
        disabled={loading}
        className="flex-1 resize-none bg-[var(--surface-2)] border border-[var(--border)] rounded px-4 py-3 text-[var(--foreground)] placeholder-[var(--muted)] focus:outline-none focus:border-[var(--accent-dim)] disabled:opacity-40 leading-relaxed"
        style={{ minHeight: "48px", maxHeight: "200px" }}
        onInput={(e) => {
          const el = e.currentTarget;
          el.style.height = "auto";
          el.style.height = Math.min(el.scrollHeight, 200) + "px";
        }}
      />
      <button
        onClick={onSubmit}
        disabled={loading || !value.trim()}
        className="px-5 py-3 rounded bg-[var(--accent-dim)] hover:bg-[var(--accent)] text-[var(--background)] font-semibold text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
      >
        {loading ? "…" : "Ask"}
      </button>
    </div>
  );
}
