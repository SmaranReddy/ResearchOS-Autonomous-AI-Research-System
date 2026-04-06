import { NextRequest } from "next/server";
import type { QueryRequest } from "@/types/chat";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  let body: QueryRequest;
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body" }), { status: 400 });
  }

  if (!body.query?.trim()) {
    return new Response(JSON.stringify({ error: "query is required" }), { status: 400 });
  }

  try {
    const upstream = await fetch(`${BACKEND_URL}/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: body.query.trim(),
        history: body.history ?? null,
      }),
      // @ts-expect-error — Node 18 fetch supports duplex
      duplex: "half",
      signal: AbortSignal.timeout(180_000),
    });

    if (!upstream.ok || !upstream.body) {
      const text = await upstream.text().catch(() => "");
      return new Response(
        `data: ${JSON.stringify({ type: "error", message: `Backend ${upstream.status}: ${text}` })}\n\n`,
        { status: 200, headers: { "Content-Type": "text/event-stream" } }
      );
    }

    // Forward the SSE stream directly — no buffering
    return new Response(upstream.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return new Response(
      `data: ${JSON.stringify({ type: "error", message })}\n\n`,
      { status: 200, headers: { "Content-Type": "text/event-stream" } }
    );
  }
}
