import { createUIMessageStreamResponse } from "ai";

const FASTAPI_URL =
  process.env["FASTAPI_CHAT_URL"] ?? "http://localhost:8000/chat";
const FASTAPI_MODELS = new Set([
  "Gemini-2.5-Pro (Baseline)",
  "Gemini-2.5-Pro + BN",
]);

export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages, system, tools, model } = await req.json();

  if (!FASTAPI_MODELS.has(model)) {
    return new Response(
      JSON.stringify({ error: `Unknown model selection: ${model}` }),
      { status: 400 },
    );
  }

  try {
    const response = await fetch(FASTAPI_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ messages, system, tools, model }),
    });

    if (!response.ok) {
      return new Response(
        JSON.stringify({
          error: "FastAPI backend request failed",
        }),
        { status: response.status },
      );
    }

    const data = await response.json();
    const text =
      typeof data?.reply === "string"
        ? data.reply
        : "FastAPI response missing reply field.";

    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue({ type: "start" });
        controller.enqueue({ type: "start-step" });
        controller.enqueue({ type: "text-start", id: "text-1" });
        controller.enqueue({
          type: "text-delta",
          id: "text-1",
          delta: text,
        });
        controller.enqueue({ type: "text-end", id: "text-1" });
        controller.enqueue({ type: "finish-step" });
        controller.enqueue({ type: "finish" });
        controller.close();
      },
    });

    return createUIMessageStreamResponse({ stream });
  } catch (error) {
    console.error(error);
    return new Response(
      JSON.stringify({ error: "FastAPI backend unreachable" }),
      { status: 500 },
    );
  }
}
