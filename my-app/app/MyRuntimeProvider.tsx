"use client";

import {
  AssistantCloud,
  AssistantRuntimeProvider,
} from "@assistant-ui/react";
import {
  AssistantChatTransport,
  useChatRuntime,
} from "@assistant-ui/react-ai-sdk";
import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
} from "react";

const cloud = new AssistantCloud({
  baseUrl: process.env["NEXT_PUBLIC_ASSISTANT_BASE_URL"]!,
  anonymous: true,
});

export const DEFAULT_MODEL = "gpt-4o";

type ModelContextValue = {
  model: string;
  setModel: (value: string) => void;
};

const ModelContext = createContext<ModelContextValue | null>(null);

export const useModelSelection = () => {
  const ctx = useContext(ModelContext);
  if (!ctx) {
    throw new Error("useModelSelection must be used within MyRuntimeProvider");
  }
  return ctx;
};

export function MyRuntimeProvider({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const [model, setModelRaw] = useState(DEFAULT_MODEL);
  const modelRef = useRef(model);
  modelRef.current = model;

  const setModel = useCallback((next: string) => {
    setModelRaw(next || DEFAULT_MODEL);
  }, []);

  const transport = useMemo(
    () =>
      new AssistantChatTransport({
        api: "/api/chat",
        prepareSendMessagesRequest: async (options) => {
          options.body = {
            ...(options.body ?? {}),
            model: modelRef.current,
          };
        },
      }),
    [],
  );

  const runtime = useChatRuntime({
    cloud,
    transport,
  });

  const contextValue = useMemo(
    () => ({
      model,
      setModel,
    }),
    [model, setModel],
  );

  return (
    <ModelContext.Provider value={contextValue}>
      <AssistantRuntimeProvider runtime={runtime}>
        {children}
      </AssistantRuntimeProvider>
    </ModelContext.Provider>
  );
}
