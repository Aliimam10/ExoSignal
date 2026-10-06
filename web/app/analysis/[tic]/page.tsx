import { AnalysisProgress } from "@/components/analysis-progress";
import { currentJob } from "@/lib/analysis-jobs";

export default async function AnalysisPage({ params }: { params: Promise<{ tic: string }> }) {
  const { tic } = await params;
  const ticId = tic.replace(/\D/g, "");
  return <main className="mx-auto max-w-3xl px-5 py-20"><AnalysisProgress ticId={ticId} initialJob={currentJob(ticId)} /></main>;
}
