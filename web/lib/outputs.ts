import { readFile, access } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";

export type JsonMap = Record<string, unknown>;
export type Candidate = JsonMap & { candidate_id: string; period_days: number; depth: number; duration_hours: number; observed_transits: number; bls_snr: number };
export type Event = Record<string, string>;
export type TargetData = { ticId: string; root: string; metadata: JsonMap; sectors: JsonMap; candidates: Candidate[]; dossiers: Record<string, JsonMap>; events: Record<string, Event[]> };

const repositoryRoot = path.resolve(process.cwd(), "..");
export const outputsRoot = path.join(repositoryRoot, "outputs");

async function exists(file: string) { try { await access(file, constants.R_OK); return true; } catch { return false; } }
async function json(file: string): Promise<JsonMap> { return JSON.parse(await readFile(file, "utf8")) as JsonMap; }

function csvRows(text: string): string[][] {
  return text.trim().split(/\r?\n/).filter(Boolean).map((line) => {
    const fields: string[] = []; let cell = ""; let quoted = false;
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      if (char === '"' && line[i + 1] === '"') { cell += char; i += 1; }
      else if (char === '"') quoted = !quoted;
      else if (char === "," && !quoted) { fields.push(cell); cell = ""; }
      else cell += char;
    }
    fields.push(cell); return fields;
  });
}

export async function csv(file: string): Promise<Event[]> {
  const rows = csvRows(await readFile(file, "utf8"));
  const [header, ...body] = rows;
  return body.map((row) => Object.fromEntries(header.map((key, index) => [key, row[index] ?? ""])));
}

function targetRoot(ticId: string, source?: string) {
  if (!/^\d+$/.test(ticId)) throw new Error("A TIC identifier must contain digits only.");
  const discovery = path.join(outputsRoot, "discovery", "sector_002", "targets", ticId);
  return source === "discovery" ? discovery : path.join(outputsRoot, "targets", ticId);
}

export async function getTarget(ticId: string, source?: string): Promise<TargetData | null> {
  const root = targetRoot(ticId, source);
  if (!await exists(path.join(root, "candidates", "candidates.json"))) return null;
  let candidates = JSON.parse(await readFile(path.join(root, "candidates", "candidates.json"), "utf8")) as Candidate[];
  if (source === "discovery") {
    const ranked = await csv(path.join(outputsRoot, "discovery", "sector_002", "pre_crossmatch_ranking.csv"));
    candidates = candidates.map((candidate) => ({ ...candidate, ranking_score: ranked.find((row) => row.tic_id === ticId && row.candidate_id === candidate.candidate_id)?.planet_like_probability }));
  }
  const dossiers: Record<string, JsonMap> = {}; const events: Record<string, Event[]> = {};
  for (const candidate of candidates) {
    const stem = candidate.candidate_id.toLowerCase();
    const dossier = path.join(root, "candidates", `${stem}_vetting.json`);
    const eventFile = path.join(root, "candidates", `${stem}_event_metrics.csv`);
    if (await exists(dossier)) dossiers[candidate.candidate_id] = await json(dossier);
    if (await exists(eventFile)) events[candidate.candidate_id] = await csv(eventFile);
  }
  return { ticId, root: path.relative(outputsRoot, root), metadata: await json(path.join(root, "target_metadata.json")), sectors: await json(path.join(root, "sectors.json")), candidates, dossiers, events };
}

export async function getDiscovery() {
  const root = path.join(outputsRoot, "discovery", "sector_002");
  const ranking = await csv(path.join(root, "pre_crossmatch_ranking.csv"));
  const vetting = await Promise.all(ranking.map(async (row) => {
    try {
      const dossier = await json(path.join(root, "targets", row.tic_id, "candidates", `${row.candidate_id.toLowerCase()}_vetting.json`));
      return [`${row.tic_id}-${row.candidate_id}`, String((dossier.overall as JsonMap | undefined)?.status || "NOT AVAILABLE")] as const;
    } catch { return [`${row.tic_id}-${row.candidate_id}`, "NOT AVAILABLE"] as const; }
  }));
  const decoratedRanking: Event[] = ranking.map((row): Event => ({ ...row, vetting_status: new Map(vetting).get(`${row.tic_id}-${row.candidate_id}`) || "NOT AVAILABLE" }));
  return {
    parameters: await json(path.join(root, "discovery_parameters.json")),
    manifest: await csv(path.join(root, "target_manifest.csv")),
    ranking: decoratedRanking,
    reveal: await csv(path.join(root, "catalogue_reveal.csv")),
  };
}

export async function getValidation() {
  const modelRoot = path.join(outputsRoot, "models");
  return {
    report: await json(path.join(modelRoot, "model_report.json")),
    benchmark: await json(path.join(outputsRoot, "benchmark_200", "benchmark_attrition.json")),
    injection: await csv(path.join(outputsRoot, "targets", "98796344", "injection_recovery.csv")),
  };
}
