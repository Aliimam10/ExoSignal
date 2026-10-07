"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { JsonMap } from "@/lib/outputs";
import { formatNumber, percent } from "@/lib/utils";
import { Card } from "@/components/ui/card";

type Target = JsonMap & { tic_id: number; toi: number; benchmark_class: string; planet_like_benchmark_score: number; actual_catalogue_disposition: string };

function label(target: Target) {
  return target.benchmark_class === "STRONGLY_PLANET_LIKE" ? "Strongly planet-like" : target.benchmark_class === "STRONGLY_FP_EB_LIKE" ? "Strongly FP / EB-like" : "Uncertain";
}

export function CatalogueList({ targets }: { targets: JsonMap[] }) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => targets.filter((item) => {
    const target = item as Target; const text = query.trim().toLowerCase();
    return !text || String(target.tic_id).includes(text) || String(target.toi).includes(text) || target.actual_catalogue_disposition.toLowerCase().includes(text) || label(target).toLowerCase().includes(text);
  }), [query, targets]);
  return <section className="mt-10"><Card className="overflow-hidden"><div className="border-b border-line p-5"><label className="text-xs uppercase tracking-[.14em] text-slate-500">Find a TIC, TOI, classification, or NASA disposition</label><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. 402026209, KP, TOI 232" className="mt-3 w-full rounded-lg border border-line bg-ink px-3 py-2.5 text-sm text-white outline-none placeholder:text-slate-500 focus:border-cyan/60" /><p className="mt-3 text-sm text-mist">Showing {filtered.length.toLocaleString()} of {targets.length.toLocaleString()} analysed TICs.</p></div><div className="max-h-[65vh] overflow-auto"><table className="w-full min-w-[720px] text-left text-sm"><thead className="sticky top-0 bg-panel text-xs uppercase tracking-[.12em] text-slate-500"><tr><th className="px-5 py-3">TIC</th><th className="px-5 py-3">TOI</th><th className="px-5 py-3">ExoSignal result</th><th className="px-5 py-3">Score</th><th className="px-5 py-3">NASA disposition</th></tr></thead><tbody>{filtered.map((item) => { const target = item as Target; return <tr key={`${target.tic_id}-${target.toi}`} className="border-t border-line/70 text-mist transition hover:bg-white/[.03]"><td className="px-5 py-3 font-mono text-cyan"><Link href={`/targets/${target.tic_id}`}>TIC {target.tic_id}</Link></td><td className="px-5 py-3">{formatNumber(target.toi, 2)}</td><td className="px-5 py-3 text-white">{label(target)}</td><td className="px-5 py-3">{percent(target.planet_like_benchmark_score, 1)}</td><td className="px-5 py-3">{target.actual_catalogue_disposition}</td></tr>; })}</tbody></table></div></Card></section>;
}
