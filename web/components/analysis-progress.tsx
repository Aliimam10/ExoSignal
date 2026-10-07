"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { CheckCircle2, Circle, LoaderCircle } from "lucide-react";
import { Card } from "@/components/ui/card";

type Job = { ticId: string; state: "running" | "completed" | "failed"; stage?: string; error?: string };
const stages = ["Looking up the NASA TOI record", "Selecting published numerical measurements", "Applying the frozen benchmark model", "Preparing the benchmark assessment"];

export function AnalysisProgress({ ticId, initialJob }: { ticId: string; initialJob?: Job }) {
  const [job, setJob] = useState<Job>(initialJob || { ticId, state: "running", stage: stages[0] });
  useEffect(() => { const poll = async () => { const response = await fetch(`/api/analyse?tic=${ticId}`, { cache: "no-store" }); setJob(await response.json()); }; poll(); const interval = setInterval(poll, 2500); return () => clearInterval(interval); }, [ticId]);
  const current = Math.max(0, stages.indexOf(job.stage || stages[0]));
  if (job.state === "completed") return <Card className="p-7"><CheckCircle2 className="text-emerald-300" size={28} /><h1 className="mt-4 text-2xl font-semibold">Benchmark assessment prepared for TIC {ticId}</h1><p className="mt-2 text-mist">Published NASA TOI measurements have been scored against the frozen benchmark model.</p><Link href={`/targets/${ticId}`} className="mt-6 inline-block rounded-lg bg-cyan px-4 py-2.5 text-sm font-medium text-slate-950">Open assessment</Link></Card>;
  if (job.state === "failed") return <Card className="p-7"><h1 className="text-2xl font-semibold">No catalogue assessment available</h1><p className="mt-3 leading-7 text-mist">{job.error || "The local assessment was not completed."}</p><p className="mt-5 text-sm leading-6 text-slate-400">A catalogue-only classifier can assess only TICs with published NASA TOI transit-candidate measurements. It cannot decide whether an arbitrary star hosts a planet.</p></Card>;
  return <Card className="p-7"><div className="flex items-center gap-3"><LoaderCircle className="animate-spin text-cyan" size={23} /><div><p className="font-mono text-xs uppercase tracking-[.16em] text-cyan">Catalogue assessment in progress</p><h1 className="mt-1 text-2xl font-semibold">Preparing TIC {ticId}</h1></div></div><p className="mt-5 max-w-xl leading-7 text-mist">This uses published NASA TOI measurements only. No light curve is downloaded and no synthetic data are generated.</p><ol className="mt-7 space-y-3">{stages.map((stage, index) => <li key={stage} className={`flex items-center gap-3 text-sm ${index <= current ? "text-white" : "text-slate-500"}`}>{index < current ? <CheckCircle2 size={17} className="text-emerald-300" /> : index === current ? <LoaderCircle size={17} className="animate-spin text-cyan" /> : <Circle size={17} />}{stage}</li>)}</ol></Card>;
}
