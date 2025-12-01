import { NextResponse } from "next/server";

const FASTAPI_BASE =
  process.env["NEXT_PUBLIC_FASTAPI_BASE_URL"] ?? "http://localhost:8000";

export async function GET() {
  const response = await fetch(`${FASTAPI_BASE}/bayes-graph`);
  if (!response.ok) {
    return NextResponse.json({ error: "Failed to load bayes graph" }, { status: 500 });
  }

  const arrayBuffer = await response.arrayBuffer();
  return new NextResponse(Buffer.from(arrayBuffer), {
    status: 200,
    headers: {
      "Content-Type": "image/png",
      "Cache-Control": "no-store",
    },
  });
}
