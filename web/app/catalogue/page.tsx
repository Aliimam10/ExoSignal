import { getCatalogueTargets, getValidation } from "@/lib/outputs";
import { CatalogueList } from "@/components/catalogue-list";

export default async function CataloguePage() {
  const [targets, validation] = await Promise.all([getCatalogueTargets(), getValidation()]);
  return <main className="mx-auto max-w-7xl px-5 pb-20 pt-12 lg:px-8"><section className="max-w-4xl"><p className="font-mono text-xs uppercase tracking-[.18em] text-cyan">Analysed TIC catalogue</p><h1 className="mt-2 text-4xl font-semibold tracking-tight">All {targets.length.toLocaleString()} final benchmark TICs.</h1><p className="mt-4 text-lg leading-8 text-mist">Choose a TIC to inspect its model result, published NASA disposition, complete NASA TOI row, and optional on-demand TESS light curve. The disposition is shown for comparison; it was not a model feature.</p><p className="mt-4 text-sm text-slate-400">Source population: {String(validation.benchmark.positive_targets)} CP/KP and {String(validation.benchmark.negative_targets)} FP/FA TICs.</p></section><CatalogueList targets={targets} /></main>;
}
