import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();

  const fastapiUrl = process.env.FASTAPI_URL;
  if (!fastapiUrl) {
    return NextResponse.json({ error: "FASTAPI_URL not configured" }, { status: 500 });
  }

  const res = await fetch(`${fastapiUrl}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
