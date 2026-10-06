import Link from "next/link";
import { notFound } from "next/navigation";
import { getTarget } from "@/lib/outputs";
import { TargetReport } from "@/components/target-report";
import { Card } from "@/components/ui/card";

export default async function TargetPage({ params, searchParams }: { params: Promise<{ tic: string }>; searchParams: Promise<{ source?: string }> }) {
  const { tic } = await params; const { source } = await searchParams; const target = await getTarget(tic, source);
  if (!/^\d+$/.test(tic)) notFound();
  if (!target) return <main className="mx-auto max-w-3xl px-5 py-20"><Card className="p-8"><p className="font-mono text-xs uppercase tracking-[.18em] text-cyan">No local dossier</p><h1 className="mt-3 text-3xl font-semibold">TIC {tic} has not been analysed here yet.</h1><p className="mt-4 leading-7 text-mist">This frontend reads ExoSignal&apos;s persisted scientific outputs. Run the Python analysis for this TIC, then refresh this page; no result is fabricated in the interface.</p><pre className="mt-6 overflow-x-auto rounded-lg border border-line bg-black/20 p-4 text-sm text-cyan">exosignal analyze "TIC {tic}"</pre><Link href="/" className="mt-6 inline-block text-cyan hover:text-white">Return to target search →</Link></Card></main>;
  return <TargetReport target={target} />;
}
