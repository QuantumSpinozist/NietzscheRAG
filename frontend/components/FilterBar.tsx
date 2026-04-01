"use client";

import { Period, WORKS } from "@/types";

interface FilterBarProps {
  period: Period;
  slug: string | null;
  onPeriodChange: (p: Period) => void;
  onSlugChange: (s: string | null) => void;
  disabled: boolean;
}

export default function FilterBar({
  period,
  slug,
  onPeriodChange,
  onSlugChange,
  disabled,
}: FilterBarProps) {
  return (
    <div className="flex gap-3 flex-wrap">
      <select
        value={period ?? ""}
        onChange={(e) => onPeriodChange((e.target.value as Period) || null)}
        disabled={disabled}
        className="bg-[var(--surface-2)] border border-[var(--border)] text-[var(--foreground)] text-sm rounded px-3 py-1.5 focus:outline-none focus:border-[var(--accent-dim)] disabled:opacity-40 cursor-pointer"
      >
        <option value="">All periods</option>
        <option value="early">Early period</option>
        <option value="middle">Middle period</option>
        <option value="late">Late period</option>
      </select>

      <select
        value={slug ?? ""}
        onChange={(e) => onSlugChange(e.target.value || null)}
        disabled={disabled}
        className="bg-[var(--surface-2)] border border-[var(--border)] text-[var(--foreground)] text-sm rounded px-3 py-1.5 focus:outline-none focus:border-[var(--accent-dim)] disabled:opacity-40 cursor-pointer"
      >
        <option value="">All works</option>
        {WORKS.map((w) => (
          <option key={w.slug} value={w.slug}>
            {w.label}
          </option>
        ))}
      </select>
    </div>
  );
}
