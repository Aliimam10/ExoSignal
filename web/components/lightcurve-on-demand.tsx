"use client";

import { useEffect, useState } from "react";
import { LoaderCircle } from "lucide-react";
import { artifactUrl } from "@/lib/artifacts";
import { Button } from "@/components/ui/button";

type Job = { state: "idle" | "running" | "completed" | "failed"; error?: string };

export function LightCurveOnDemand({ ticId }: { ticId: string }) {
  const [job, setJob] = useState<Job>();
  useEffect(() => {
    fetch(`/api/lightcurve?tic=${ticId}`, { cache: "no-store" }).then((response) => response.json()).then(setJob).catch(() => undefined);
  }, [ticId]);
  useEffect(() => {
    if (job?.state !== "running") return;
    const interval = setInterval(() => fetch(`/api/lightcurve?tic=${ticId}`, { cache: "no-store" }).then((response) => response.json()).then(setJob).catch(() => undefined), 3000);
    return () => clearInterval(interval);
  }, [job?.state, ticId]);
  async function start() {
    const response = await fetch("/api/lightcurve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ticId }) });
    setJob(await response.json());
  }
  if (job?.state === "completed") return <div><img className="mt-2 w-full rounded-lg border border-line" src={artifactUrl(`targets/${ticId}/candidates/c01_phase_folded.png`)} alt={`Phase-folded transit candidate light curve for TIC ${ticId}`} /><p className="mt-3 text-xs text-slate-400">Phase-folded strongest transit-like candidate: repeated orbital cycles are aligned so a possible flux dip is visible near the centre. This visualisation was not used by the catalogue classifier.</p></div>;
  if (job?.state === "running") return <p className="inline-flex items-center gap-2 text-sm text-cyan"><LoaderCircle className="animate-spin" size={16} /> Retrieving and processing public TESS data; multi-sector targets can take a few minutes…</p>;
  return <div><Button onClick={start}>Retrieve TESS light curve</Button>{job?.state === "failed" && <p className="mt-3 text-sm text-rose-300">{job.error}</p>}</div>;
}
