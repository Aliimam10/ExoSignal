import { cn } from "@/lib/utils";

const tones: Record<string, string> = {
  PASS: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300",
  WARNING: "border-amber-300/30 bg-amber-300/10 text-amber-200",
  FAIL: "border-rose-400/30 bg-rose-400/10 text-rose-300",
  "NOT AVAILABLE": "border-slate-500/40 bg-slate-500/10 text-slate-300",
};

export function StatusBadge({ status }: { status?: string }) {
  const label = status || "NOT AVAILABLE";
  return <span className={cn("inline-flex rounded-full border px-2.5 py-1 text-[11px] font-semibold tracking-[.12em]", tones[label] || "border-cyan/30 bg-cyan/10 text-cyan")}>{label}</span>;
}
