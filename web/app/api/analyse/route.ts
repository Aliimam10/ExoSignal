import { NextRequest } from "next/server";
import { currentJob, startJob } from "@/lib/analysis-jobs";

function ticId(value: unknown) { const result = String(value || "").replace(/\D/g, ""); return result || null; }

export async function POST(request: NextRequest) {
  const id = ticId((await request.json().catch(() => ({}))).ticId);
  if (!id) return Response.json({ error: "A numeric TIC identifier is required." }, { status: 400 });
  return Response.json(startJob(id));
}

export function GET(request: NextRequest) {
  const id = ticId(request.nextUrl.searchParams.get("tic"));
  if (!id) return Response.json({ error: "A numeric TIC identifier is required." }, { status: 400 });
  return Response.json(currentJob(id) || { ticId: id, state: "failed", error: "No local analysis job is active for this TIC." });
}
