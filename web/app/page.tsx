import Link from "next/link";
import { ArrowRight, Database, type LucideIcon, ScanLine, ShieldCheck, Sparkles } from "lucide-react";
import { SearchBox } from "@/components/search-box";
import { Card } from "@/components/ui/card";

const stages: Array<[LucideIcon, string, string]> = [[Database, "TESS data", "Sector-aware SPOC PDCSAP light curves"], [ScanLine, "Transit search", "Iterative BLS with protected detrending"], [ShieldCheck, "Astrophysical vetting", "Event, pixel, crowding, and EB diagnostics"], [Sparkles, "ML ranking", "Held-out evaluated candidate prioritisation"]];

export default function SearchPage() {
  return <main className="mx-auto max-w-7xl px-5 pb-20 pt-16 lg:px-8 lg:pt-24">
    <section className="grid gap-12 lg:grid-cols-[1.15fr_.85fr] lg:items-center">
      <div className="animate-enter">
        <p className="mb-5 font-mono text-xs font-medium uppercase tracking-[.22em] text-cyan">TESS transit intelligence</p>
        <h1 className="max-w-3xl text-5xl font-semibold leading-[1.02] tracking-[-.045em] text-white sm:text-6xl">A clear signal path from starlight to scrutiny.</h1>
        <p className="mt-7 max-w-2xl text-lg leading-8 text-mist">ExoSignal turns real TESS observations into inspectable transit candidates, with the evidence needed to question every result.</p>
        <div className="mt-9 max-w-2xl"><SearchBox /><p className="mt-3 text-sm text-slate-400">Opens a locally persisted analysis. Try the included TIC 402026209.</p></div>
        <div className="mt-7 flex flex-wrap gap-3"><Link href="/discovery" className="inline-flex items-center gap-2 rounded-lg border border-line bg-white/[.035] px-4 py-2.5 text-sm text-slate-100 transition hover:bg-white/[.08]">Explore blind discovery <ArrowRight size={16} /></Link><Link href="/validation" className="inline-flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm text-cyan transition hover:text-white">Inspect validation evidence <ArrowRight size={16} /></Link></div>
      </div>
      <Card className="relative overflow-hidden p-7 animate-enter [animation-delay:120ms]">
        <div className="absolute right-0 top-0 size-48 rounded-full bg-violet/10 blur-3xl" />
        <p className="text-sm font-medium text-mist">Scientific report, not a black box</p>
        <div className="mt-7 space-y-6">{[["Sectors", "Process each observation independently"], ["Candidates", "Keep raw numerical evidence visible"], ["Ranking", "Never confuse a score with confirmation"]].map(([label, text], index) => <div key={label} className="flex gap-4"><span className="grid size-8 shrink-0 place-items-center rounded-full border border-cyan/20 text-xs text-cyan">0{index + 1}</span><div><h2 className="font-medium text-white">{label}</h2><p className="mt-1 text-sm leading-6 text-slate-400">{text}</p></div></div>)}</div>
      </Card>
    </section>
    <section className="mt-20"><p className="font-mono text-xs uppercase tracking-[.2em] text-slate-400">Evidence pipeline</p><div className="mt-5 grid gap-4 md:grid-cols-4">{stages.map(([Icon, title, description], index) => <Card key={title} className="p-5 transition hover:-translate-y-1 hover:border-cyan/35"><span className="mb-7 inline-flex size-9 items-center justify-center rounded-lg bg-cyan/10 text-cyan"><Icon size={19} /></span><p className="text-xs text-slate-500">0{index + 1}</p><h2 className="mt-1 font-semibold text-white">{title}</h2><p className="mt-2 text-sm leading-6 text-mist">{description}</p></Card>)}</div></section>
  </main>;
}
