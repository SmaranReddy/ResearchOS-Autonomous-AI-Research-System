"use client";

interface Props {
  value: number; // 0.0 – 1.0
}

function label(v: number) {
  if (v >= 0.85) return { text: "Very high", color: "text-emerald-300", bar: "bg-emerald-400" };
  if (v >= 0.70) return { text: "High",      color: "text-emerald-400", bar: "bg-emerald-500" };
  if (v >= 0.40) return { text: "Moderate",  color: "text-amber-400",   bar: "bg-amber-500"   };
  return              { text: "Low",       color: "text-red-400",    bar: "bg-red-500"     };
}

export default function ConfidenceBadge({ value }: Props) {
  const { text, color, bar } = label(value);
  const pct = Math.round(value * 100);

  return (
    <div className="mt-3 pt-3 border-t border-white/5">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-500 uppercase tracking-wider">Confidence</span>
        <span className={`text-xs font-semibold ${color}`}>
          {text} &middot; {pct}%
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${bar}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
