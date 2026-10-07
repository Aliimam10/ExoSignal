import Link from "next/link";
import { notFound } from "next/navigation";
import { getCatalogueAssessment } from "@/lib/outputs";
import { CatalogueAssessment } from "@/components/catalogue-assessment";
import { Card } from "@/components/ui/card";

export default async function TargetPage({ params }: { params: Promise<{ tic: string }> }) {
  const { tic } = await params; const assessment = await getCatalogueAssessment(tic);
  if (!/^\d+$/.test(tic)) notFound();
  if (!assessment) return <main className="mx-auto max-w-3xl px-5 py-20"><Card className="p-8"><p className="font-mono text-xs uppercase tracking-[.18em] text-cyan">No published TOI assessment</p><h1 className="mt-3 text-3xl font-semibold">TIC {tic} has no locally prepared NASA TOI assessment.</h1><p className="mt-4 leading-7 text-mist">A catalogue-only classifier can only score targets with published NASA TOI candidate measurements. It cannot determine whether an arbitrary star hosts an exoplanet.</p><Link href="/" className="mt-6 inline-block text-cyan hover:text-white">Return to TIC assessment →</Link></Card></main>;
  return <CatalogueAssessment assessment={assessment} />;
}
