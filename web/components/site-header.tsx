import Link from "next/link";
import { Orbit } from "lucide-react";

const links = [["Summary", "/"], ["TIC catalogue", "/catalogue"], ["Benchmark validation", "/validation"], ["Methodology", "/methodology"]];

export function SiteHeader() {
  return <header className="sticky top-0 z-40 border-b border-line/70 bg-ink/85 backdrop-blur-xl">
    <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5 lg:px-8">
      <Link href="/" className="flex items-center gap-2.5 font-semibold tracking-tight text-white"><span className="grid size-8 place-items-center rounded-lg border border-cyan/30 bg-cyan/10 text-cyan"><Orbit size={18} /></span>ExoSignal</Link>
      <nav className="flex items-center gap-1" aria-label="Primary navigation">
        {links.map(([label, href]) => <Link key={href} href={href} className="rounded-md px-3 py-2 text-sm text-mist transition hover:bg-white/[.05] hover:text-white">{label}</Link>)}
      </nav>
    </div>
  </header>;
}
