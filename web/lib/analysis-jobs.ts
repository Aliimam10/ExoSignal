import { existsSync } from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { catalogueBenchmarkRoot } from "@/lib/outputs";

type JobState = "running" | "completed" | "failed";
export type AnalysisJob = { ticId: string; state: JobState; stage: string; startedAt: string; error?: string };

const jobs = new Map<string, AnalysisJob>();
const stages = ["Looking up the NASA TOI record", "Selecting published numerical measurements", "Applying the frozen benchmark model", "Preparing the benchmark assessment"];

function targetDirectory(ticId: string) { return path.join(catalogueBenchmarkRoot, "assessments", ticId); }
function stageFromOutputs(ticId: string) {
  const root = targetDirectory(ticId);
  if (existsSync(path.join(root, "catalogue_assessment.json"))) return stages[3];
  if (existsSync(root)) return stages[2];
  return stages[0];
}

export function currentJob(ticId: string): AnalysisJob | undefined {
  const job = jobs.get(ticId);
  if (job?.state === "running") job.stage = stageFromOutputs(ticId);
  if (!job && existsSync(path.join(targetDirectory(ticId), "catalogue_assessment.json"))) return { ticId, state: "completed", stage: stages[3], startedAt: "Persisted local result" };
  return job;
}

export function startJob(ticId: string): AnalysisJob {
  const existing = currentJob(ticId);
  if (existing?.state === "running" || existing?.state === "completed") return existing;
  const job: AnalysisJob = { ticId, state: "running", stage: stageFromOutputs(ticId), startedAt: new Date().toISOString() };
  jobs.set(ticId, job);
  const executable = process.env.EXOSIGNAL_PYTHON || "python";
  const processHandle = spawn(executable, ["-m", "exosignal.cli", "catalogue-assess", `TIC ${ticId}`], { cwd: path.resolve(process.cwd(), ".."), stdio: "ignore", detached: false });
  processHandle.once("error", (error) => { job.state = "failed"; job.error = `Could not start the local ExoSignal command: ${error.message}`; });
  processHandle.once("close", (code) => {
    if (code === 0 && existsSync(path.join(targetDirectory(ticId), "catalogue_assessment.json"))) { job.state = "completed"; job.stage = stages[3]; }
    else { job.state = "failed"; job.error = "No assessable NASA TOI record was found, or the local benchmark assessment did not finish successfully."; }
  });
  return job;
}

export { stages };
