import { existsSync } from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { outputsRoot } from "@/lib/outputs";

type JobState = "running" | "completed" | "failed";
export type AnalysisJob = { ticId: string; state: JobState; stage: string; startedAt: string; error?: string };

const jobs = new Map<string, AnalysisJob>();
const stages = ["Retrieving TESS observations", "Preprocessing sectors", "Searching for periodic transits", "Running astrophysical vetting", "Preparing dossier"];

function targetDirectory(ticId: string) { return path.join(outputsRoot, "targets", ticId); }
function stageFromOutputs(ticId: string) {
  const root = targetDirectory(ticId); const candidates = path.join(root, "candidates", "candidates.json");
  if (existsSync(path.join(root, "candidates", "c01_vetting.json"))) return stages[4];
  if (existsSync(candidates)) return stages[3];
  if (existsSync(path.join(root, "processed_lightcurve.csv"))) return stages[2];
  if (existsSync(root)) return stages[1];
  return stages[0];
}

export function currentJob(ticId: string): AnalysisJob | undefined {
  const job = jobs.get(ticId);
  if (job?.state === "running") job.stage = stageFromOutputs(ticId);
  if (!job && existsSync(path.join(targetDirectory(ticId), "candidates", "candidates.json"))) return { ticId, state: "completed", stage: stages[4], startedAt: "Persisted local result" };
  return job;
}

export function startJob(ticId: string): AnalysisJob {
  const existing = currentJob(ticId);
  if (existing?.state === "running" || existing?.state === "completed") return existing;
  const job: AnalysisJob = { ticId, state: "running", stage: stageFromOutputs(ticId), startedAt: new Date().toISOString() };
  jobs.set(ticId, job);
  const executable = process.env.EXOSIGNAL_PYTHON || "python";
  const processHandle = spawn(executable, ["-m", "exosignal.cli", "analyze", `TIC ${ticId}`], { cwd: path.resolve(process.cwd(), ".."), stdio: "ignore", detached: false });
  processHandle.once("error", (error) => { job.state = "failed"; job.error = `Could not start the local ExoSignal command: ${error.message}`; });
  processHandle.once("close", (code) => {
    if (code === 0 && existsSync(path.join(targetDirectory(ticId), "candidates", "candidates.json"))) { job.state = "completed"; job.stage = stages[4]; }
    else { job.state = "failed"; job.error = "The local analysis did not finish successfully. Check the terminal output and saved target directory."; }
  });
  return job;
}

export { stages };
