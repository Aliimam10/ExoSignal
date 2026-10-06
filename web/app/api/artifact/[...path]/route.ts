import { readFile } from "node:fs/promises";
import path from "node:path";
import { outputsRoot } from "@/lib/outputs";

const mime: Record<string, string> = { ".png": "image/png", ".csv": "text/csv; charset=utf-8", ".json": "application/json; charset=utf-8" };

export async function GET(_: Request, context: { params: Promise<{ path: string[] }> }) {
  const { path: pieces } = await context.params;
  const candidate = path.resolve(outputsRoot, ...pieces);
  if (!candidate.startsWith(`${outputsRoot}${path.sep}`)) return new Response("Not found", { status: 404 });
  try { return new Response(await readFile(candidate), { headers: { "Content-Type": mime[path.extname(candidate)] || "application/octet-stream", "Cache-Control": "private, max-age=300" } }); }
  catch { return new Response("Not found", { status: 404 }); }
}
