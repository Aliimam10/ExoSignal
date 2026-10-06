"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";

export function SearchBox() {
  const [tic, setTic] = useState("402026209"); const [error, setError] = useState(""); const [submitting, setSubmitting] = useState(false); const router = useRouter();
  async function submit(event: FormEvent) { event.preventDefault(); const id = tic.replace(/\D/g, ""); if (!id) { setError("Enter a numeric TIC identifier."); return; } setSubmitting(true); setError(""); try { const response = await fetch("/api/analyse", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ticId: id }) }); if (!response.ok) throw new Error((await response.json()).error || "Unable to start analysis."); router.push(`/analysis/${id}`); } catch (reason) { setError(reason instanceof Error ? reason.message : "Unable to start analysis."); setSubmitting(false); } }
  return <form onSubmit={submit} className="rounded-2xl border border-cyan/20 bg-[#0e2037]/90 p-2 shadow-panel sm:flex sm:items-center">
    <label className="flex flex-1 items-center gap-3 px-3 py-2"><Search className="text-cyan" size={18} /><span className="sr-only">TIC identifier</span><input disabled={submitting} value={tic} onChange={(event) => setTic(event.target.value)} placeholder="TIC 402026209" className="w-full bg-transparent text-base text-white outline-none placeholder:text-slate-500" /></label>
    <Button disabled={submitting} type="submit" className="w-full sm:w-auto">{submitting ? "Starting analysis…" : "Analyse target"}</Button>{error && <p className="px-3 text-sm text-rose-300">{error}</p>}
  </form>;
}
