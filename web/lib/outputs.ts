import { access, constants, readFile } from "node:fs/promises";
import path from "node:path";

export type JsonMap = Record<string, unknown>;

const repositoryRoot = path.resolve(process.cwd(), "..");
export const outputsRoot = path.join(repositoryRoot, "outputs");
export const catalogueBenchmarkRoot = path.join(outputsRoot, "catalogue_benchmark_final");

async function exists(file: string) {
  try { await access(file, constants.R_OK); return true; } catch { return false; }
}

async function json(file: string): Promise<JsonMap> {
  return JSON.parse(await readFile(file, "utf8")) as JsonMap;
}

export async function getValidation() {
  const root = catalogueBenchmarkRoot;
  return {
    report: await json(path.join(root, "models", "model_report.json")),
    benchmark: await json(path.join(root, "catalogue_benchmark_report.json")),
    schema: await json(path.join(root, "feature_schema.json")),
  };
}

export async function getCatalogueAssessment(ticId: string): Promise<JsonMap | null> {
  if (!/^\d+$/.test(ticId)) return null;
  const file = path.join(catalogueBenchmarkRoot, "assessments", ticId, "catalogue_assessment.json");
  if (await exists(file)) return json(file);
  const index = await json(path.join(catalogueBenchmarkRoot, "catalogue_target_index.json"));
  const candidates = ((index.targets as JsonMap[]) || []).filter((candidate) => String(candidate.tic_id) === ticId);
  if (!candidates.length) return null;
  return {
    tic_id: Number(ticId), source: index.source, selected_model: index.selected_model,
    strong_classification_thresholds: index.strong_classification_thresholds,
    interpretation: index.interpretation, candidates,
  };
}

export async function getCatalogueTargets(): Promise<JsonMap[]> {
  const index = await json(path.join(catalogueBenchmarkRoot, "catalogue_target_index.json"));
  return ((index.targets as JsonMap[]) || []).map((target) => ({
    tic_id: target.tic_id, toi: target.toi, benchmark_class: target.benchmark_class,
    planet_like_benchmark_score: target.planet_like_benchmark_score,
    actual_catalogue_disposition: target.actual_catalogue_disposition,
  }));
}
