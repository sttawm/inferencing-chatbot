import { openai } from "@ai-sdk/openai";
import { frontendTools } from "@assistant-ui/react-ai-sdk";
import { convertToModelMessages, streamText } from "ai";

const FASTAPI_MODEL = "my-fastapi";
const FASTAPI_URL =
  process.env["FASTAPI_CHAT_URL"] ?? "http://localhost:8000/chat";

const OPENAI_MODELS = new Set(["gpt-4o", "gpt-4o-mini"]);
const DEFAULT_OPENAI_MODEL = "gpt-4o";

export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages, system, tools, model } = await req.json();

  if (model === FASTAPI_MODEL) {
    try {
      const response = await fetch(FASTAPI_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ messages, system, tools}),
      });

      if (!response.ok) {
        return new Response(
          JSON.stringify({
            error: "FastAPI backend request failed",
          }),
          { status: response.status },
        );
      }

      return response;
    } catch (error) {
      console.error(error);
      return new Response(
        JSON.stringify({ error: "FastAPI backend unreachable" }),
        { status: 500 },
      );
    }
  }

  const resolvedModel =
    typeof model === "string" && OPENAI_MODELS.has(model)
      ? model
      : DEFAULT_OPENAI_MODEL;

  const result = streamText({
    model: openai(resolvedModel),
    messages: convertToModelMessages(messages),
    // forward system prompt and tools from the frontend
    system,
    tools: {
      ...frontendTools(tools),
    },
  });

  return result.toUIMessageStreamResponse();
}
