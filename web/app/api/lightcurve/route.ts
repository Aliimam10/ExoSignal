import { NextRequest } from "next/server";
import { currentLightCurveJob, startLightCurveJob } from "@/lib/lightcurve-jobs";

function ticId(value: unknown) { const result = String(value || "").replace(/\D/g, ""); return result || null; }

export async function POST(request: NextRequest) {
  const id = ticId((await request.json().catch(() => ({}))).ticId);
  if (!id) return Response.json({ error: "A numeric TIC identifier is required." }, { status: 400 });
  return Response.json(startLightCurveJob(id));
}

export function GET(request: NextRequest) {
  const id = ticId(request.nextUrl.searchParams.get("tic"));
  if (!id) return Response.json({ error: "A numeric TIC identifier is required." }, { status: 400 });
  return Response.json(currentLightCurveJob(id) || { ticId: id, state: "idle" });
}
