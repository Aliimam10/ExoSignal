import { CheckCircle2, Database, ShieldCheck } from "lucide-react";
import { getValidation } from "@/lib/outputs";
import { artifactUrl } from "@/lib/artifacts";
import { formatNumber } from "@/lib/utils";
import { Card } from "@/components/ui/card";

type Metrics = Record<string, unknown>;

function Metric({ label, value }: { label: string; value: unknown }) {
  return <div className="rounded-lg border border-line/70 bg-white/[.025] px-3 py-3"><p className="text-[10px] uppercase tracking-[.12em] text-slate-500">{label}</p><p className="mt-1 font-mono text-lg text-white">{formatNumber(value, 3)}</p></div>;
}

export default async function ValidationPage() {
  const { report, benchmark, schema } = await getValidation();
  const selectedModel = String(report.selected_ranking_model || "calibrated_random_forest");
  const selectedMetrics = report[`${selectedModel}_test`] as Metrics;
  const splitCounts = report.split_counts as Record<string, Metrics>;
  const test = splitCounts.test || {};
  const testTotal = Object.values(test).reduce<number>((total, count) => total + Number(count), 0);
  const features = (schema.feature_columns as string[]) || [];
  const matrix = (selectedMetrics.confusion_matrix as number[][]) || [];
  const bands = report.validation_strong_class_thresholds as Metrics;

  return <main className="mx-auto max-w-7xl px-5 pb-20 pt-12 lg:px-8">
    <section className="max-w-4xl"><p className="font-mono text-xs uppercase tracking-[.18em] text-cyan">Benchmark validation</p><h1 className="mt-2 text-4xl font-semibold tracking-tight">A fixed NASA catalogue benchmark, with labels held back.</h1><p className="mt-4 text-lg leading-8 text-mist">This is a classification evaluation of published TOI measurements. The disposition is hidden during fitting and revealed only for the untouched TIC-level test set. It is not an independent light-curve recovery experiment.</p></section>

    <section className="mt-10 grid gap-4 md:grid-cols-4">{[["Trusted source TICs", benchmark.targets], ["Planet-like labels", benchmark.positive_targets], ["FP / EB labels", benchmark.negative_targets], ["Untouched test TICs", testTotal]].map(([label, value]) => <Card key={String(label)} className="p-5"><p className="text-xs uppercase tracking-[.13em] text-slate-500">{String(label)}</p><p className="mt-2 text-3xl font-semibold">{String(value)}</p></Card>)}</section>

    <section className="mt-14 grid gap-7 lg:grid-cols-[.82fr_1.18fr]"><div><p className="font-mono text-xs uppercase tracking-[.18em] text-cyan">Final untouched evaluation</p><h2 className="mt-2 text-2xl font-semibold">{selectedModel.replaceAll("_", " ")}</h2><p className="mt-3 leading-7 text-mist">The final model was selected from validation performance, calibrated only on validation TICs, then evaluated once on {testTotal} held-out TICs. The observed accuracy is {formatNumber(selectedMetrics.accuracy, 3)}—a real measured result, not a target percentage.</p><div className="mt-5 rounded-xl border border-cyan/20 bg-cyan/5 p-4 text-sm leading-6 text-cyan">Features are NASA&apos;s numerical columns only. IDs, names, TOI disposition, dates, and catalogue-status fields are excluded from the model.</div></div><div className="grid gap-4 sm:grid-cols-2">{[["Precision", selectedMetrics.precision], ["Recall", selectedMetrics.recall], ["PR-AUC", selectedMetrics.pr_auc], ["ROC-AUC", selectedMetrics.roc_auc], ["Accuracy", selectedMetrics.accuracy], ["Brier score", selectedMetrics.brier_score]].map(([label, value]) => <Metric key={String(label)} label={String(label)} value={value} />)}</div></section>

    <section className="mt-10"><Card className="p-4"><img className="w-full rounded-lg border border-line" src={artifactUrl("catalogue_benchmark_final/models/held_out_model_evaluation.png")} alt="Held-out confusion matrix and calibration plot" /><p className="px-2 pt-3 text-sm leading-6 text-mist">Held-out confusion matrix and calibration curve. The test labels remained untouched until this final report.</p></Card></section>

    <section className="mt-14 grid gap-4 md:grid-cols-3"><Card className="p-6"><Database className="text-cyan" size={21} /><h2 className="mt-5 text-lg font-semibold">Final source benchmark</h2><p className="mt-2 text-sm leading-6 text-mist">{String(benchmark.positive_targets)} CP/KP planet-like and {String(benchmark.negative_targets)} FP/FA false-positive or false-alarm TICs. The {String(benchmark.excluded_prior_test_tics)} already-viewed pilot-test TICs were excluded.</p></Card><Card className="p-6"><ShieldCheck className="text-cyan" size={21} /><h2 className="mt-5 text-lg font-semibold">Validation-derived UI bands</h2><p className="mt-2 text-sm leading-6 text-mist">Strong FP / EB-like: score ≤ {formatNumber(bands.strong_fp_eb_max_score, 3)}. Strongly planet-like: score ≥ {formatNumber(bands.strong_planet_like_min_score, 3)}. Scores in between are uncertain.</p></Card><Card className="p-6"><CheckCircle2 className="text-cyan" size={21} /><h2 className="mt-5 text-lg font-semibold">Confusion matrix</h2><p className="mt-2 text-sm leading-6 text-mist">Metrics use the fixed 0.50 binary threshold only. True negatives / false positives: {matrix[0] ? `${matrix[0][0]} / ${matrix[0][1]}` : "not available"}. False negatives / true positives: {matrix[1] ? `${matrix[1][0]} / ${matrix[1][1]}` : "not available"}.</p></Card></section>

    <section className="mt-14"><p className="font-mono text-xs uppercase tracking-[.18em] text-cyan">Model inputs</p><h2 className="mt-2 text-2xl font-semibold">Original NASA numerical column names</h2><div className="mt-5 flex flex-wrap gap-2">{features.map((feature) => <span key={feature} className="rounded-full border border-line bg-panel px-3 py-1.5 font-mono text-xs text-mist">{feature}</span>)}</div><p className="mt-5 max-w-4xl text-sm leading-6 text-slate-400">A score means &ldquo;more consistent with this benchmark,&rdquo; not that the star is confirmed to host an exoplanet. The assessor only works for TICs that have a published NASA TOI candidate row, because an arbitrary star does not have the published transit measurements required by this catalogue-only model.</p></section>
  </main>;
}
