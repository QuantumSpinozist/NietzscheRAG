"use client";

import { useState, useEffect } from "react";
import { SourceResult } from "@/types";

interface SourceCardProps {
  source: SourceResult;
}

function SourceModal({ source, onClose }: { source: SourceResult; onClose: () => void }) {
  const section = source.section_number != null ? `§${source.section_number}` : null;

  // Close on Escape
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  // Prevent body scroll while open
  useEffect(() => {
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = ""; };
  }, []);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-2xl max-h-[85vh] flex flex-col rounded border border-[var(--border)] bg-[var(--surface)] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-baseline justify-between px-5 py-3 border-b border-[var(--border)] shrink-0">
          <div className="flex items-baseline gap-2 min-w-0">
            <span className="text-[var(--accent)] font-semibold truncate">
              {source.work_title}
            </span>
            {section && (
              <span className="text-[var(--muted)] font-mono-ui text-xs shrink-0">{section}</span>
            )}
          </div>
          <div className="flex items-center gap-3 ml-3 shrink-0">
            <span className="text-xs text-[var(--muted)] capitalize">{source.chunk_type}</span>
            <button
              onClick={onClose}
              aria-label="Close"
              className="text-[var(--muted)] hover:text-[var(--foreground)] transition-colors text-lg leading-none"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Scrollable content */}
        <div className="overflow-y-auto px-5 py-4 text-[var(--foreground)] leading-relaxed italic text-sm">
          &ldquo;{source.content.trim()}&rdquo;
        </div>

        {/* Footer */}
        <div className="px-5 py-2 border-t border-[var(--border)] shrink-0">
          <span className="text-xs text-[var(--muted)] font-mono-ui">
            similarity {source.similarity.toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  );
}

export default function SourceCard({ source }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const section = source.section_number != null ? `§${source.section_number}` : null;

  return (
    <>
      <button
        onClick={() => setExpanded(true)}
        className="w-full text-left rounded border border-[var(--border)] bg-[var(--surface-2)] overflow-hidden text-sm hover:border-[var(--accent-dim)] transition-colors group"
      >
        {/* Header */}
        <div className="flex items-baseline justify-between px-4 py-2 border-b border-[var(--border)] bg-[var(--surface)]">
          <div className="flex items-baseline gap-2 min-w-0">
            <span className="text-[var(--accent)] font-semibold truncate">
              {source.work_title}
            </span>
            {section && (
              <span className="text-[var(--muted)] font-mono-ui text-xs shrink-0">{section}</span>
            )}
          </div>
          <div className="flex items-center gap-2 shrink-0 ml-2">
            <span className="text-xs text-[var(--muted)] capitalize">{source.chunk_type}</span>
            <span className="text-xs text-[var(--muted)] opacity-0 group-hover:opacity-100 transition-opacity font-mono-ui">
              ⤢
            </span>
          </div>
        </div>

        {/* Content preview */}
        <div className="px-4 py-3 text-[var(--foreground)] leading-relaxed italic line-clamp-4 text-sm">
          &ldquo;{source.content.slice(0, 280).trim()}
          {source.content.length > 280 ? "…" : ""}&rdquo;
        </div>
      </button>

      {expanded && (
        <SourceModal source={source} onClose={() => setExpanded(false)} />
      )}
    </>
  );
}
