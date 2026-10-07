import { existsSync, statSync } from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { outputsRoot } from "@/lib/outputs";

type State = "running" | "completed" | "failed";
export type LightCurveJob = { ticId: string; state: State; error?: string };

const jobs = new Map<string, LightCurveJob>();

function output(ticId: string) { return path.join(outputsRoot, "targets", ticId, "processed_lightcurve.png"); }
function targetDirectory(ticId: string) { return path.join(outputsRoot, "targets", ticId); }

export function currentLightCurveJob(ticId: string): LightCurveJob | undefined {
  const job = jobs.get(ticId);
  if (!job && existsSync(output(ticId))) return { ticId, state: "completed" };
  // A server restart clears the in-memory job map but must not invite a
  // duplicate download while a recently started Python process still works.
  if (!job && existsSync(targetDirectory(ticId))) {
    const ageMs = Date.now() - statSync(targetDirectory(ticId)).mtimeMs;
    if (ageMs < 30 * 60 * 1000) return { ticId, state: "running" };
  }
  return job;
}

export function startLightCurveJob(ticId: string): LightCurveJob {
  const existing = currentLightCurveJob(ticId);
  if (existing?.state === "running" || existing?.state === "completed") return existing;
  const job: LightCurveJob = { ticId, state: "running" };
  jobs.set(ticId, job);
  const executable = process.env.EXOSIGNAL_PYTHON || "python";
  // The on-demand view needs a phase-folded BLS candidate, not the slower
  // optional TLS refinement. TLS remains available in the scientific CLI.
  const handle = spawn(executable, ["-m", "exosignal.cli", "analyze", `TIC ${ticId}`, "--skip-tls"], { cwd: path.resolve(process.cwd(), ".."), stdio: "ignore", detached: false });
  handle.once("error", (error) => { job.state = "failed"; job.error = `Could not start the local TESS retrieval: ${error.message}`; });
  handle.once("close", (code) => {
    if (code === 0 && existsSync(output(ticId))) job.state = "completed";
    else { job.state = "failed"; job.error = "TESS light-curve retrieval did not complete. The catalogue assessment remains available."; }
  });
  return job;
}
