"use client";

import { SourceResult } from "@/types";

interface SourceCardProps {
  source: SourceResult;
}

export default function SourceCard({ source }: SourceCardProps) {
  const section = source.section_number != null ? `§${source.section_number}` : null;

  return (
    <div className="rounded border border-[var(--border)] bg-[var(--surface-2)] overflow-hidden text-sm">
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
        <span className="text-xs text-[var(--muted)] capitalize shrink-0 ml-2">
          {source.chunk_type}
        </span>
      </div>

      {/* Content */}
      <div className="px-4 py-3 text-[var(--foreground)] leading-relaxed italic line-clamp-4 text-sm">
        &ldquo;{source.content.slice(0, 280).trim()}
        {source.content.length > 280 ? "…" : ""}&rdquo;
      </div>
    </div>
  );
}
