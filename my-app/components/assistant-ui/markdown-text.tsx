"use client";

import "@assistant-ui/react-markdown/styles/dot.css";

import {
  type CodeHeaderProps,
  MarkdownTextPrimitive,
  unstable_memoizeMarkdownComponents as memoizeMarkdownComponents,
  useIsMarkdownCodeBlock,
} from "@assistant-ui/react-markdown";
import { useMessagePartText } from "@assistant-ui/react";
import remarkGfm from "remark-gfm";
import { type FC, memo, useMemo, useState } from "react";
import { CheckIcon, CopyIcon } from "lucide-react";

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { cn } from "@/lib/utils";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";

const MarkdownTextImpl = () => {
  const messagePart = useMessagePartText();

  const bnSections = useMemo(() => extractBnSections(messagePart.text), [
    messagePart.text,
  ]);

  const mainText = bnSections?.assistant ?? messagePart.text;

  return (
    <div className="relative">
      {bnSections && (
        <BnSidePanels
          updates={bnSections.updates}
          probabilities={bnSections.probabilities}
          deltas={bnSections.deltas}
          inferenceTiming={bnSections.inferenceTiming}
        />
      )}

      <MarkdownTextPrimitive
        remarkPlugins={[remarkGfm]}
        className="aui-md"
        components={defaultComponents}
        preprocess={() => mainText}
      />
    </div>
  );
};

export const MarkdownText = memo(MarkdownTextImpl);

const CodeHeader: FC<CodeHeaderProps> = ({ language, code }) => {
  const { isCopied, copyToClipboard } = useCopyToClipboard();
  const onCopy = () => {
    if (!code || isCopied) return;
    copyToClipboard(code);
  };

  return (
    <div className="aui-code-header-root mt-4 flex items-center justify-between gap-4 rounded-t-lg bg-muted-foreground/15 px-4 py-2 text-sm font-semibold text-foreground dark:bg-muted-foreground/20">
      <span className="aui-code-header-language lowercase [&>span]:text-xs">
        {language}
      </span>
      <TooltipIconButton tooltip="Copy" onClick={onCopy}>
        {!isCopied && <CopyIcon />}
        {isCopied && <CheckIcon />}
      </TooltipIconButton>
    </div>
  );
};

const useCopyToClipboard = ({
  copiedDuration = 3000,
}: {
  copiedDuration?: number;
} = {}) => {
  const [isCopied, setIsCopied] = useState<boolean>(false);

  const copyToClipboard = (value: string) => {
    if (!value) return;

    navigator.clipboard.writeText(value).then(() => {
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), copiedDuration);
    });
  };

  return { isCopied, copyToClipboard };
};

const defaultComponents = memoizeMarkdownComponents({
  h1: ({ className, ...props }) => (
    <h1
      className={cn(
        "aui-md-h1 mb-8 scroll-m-20 text-4xl font-extrabold tracking-tight last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h2: ({ className, ...props }) => (
    <h2
      className={cn(
        "aui-md-h2 mt-8 mb-4 scroll-m-20 text-3xl font-semibold tracking-tight first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h3: ({ className, ...props }) => (
    <h3
      className={cn(
        "aui-md-h3 mt-6 mb-4 scroll-m-20 text-2xl font-semibold tracking-tight first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h4: ({ className, ...props }) => (
    <h4
      className={cn(
        "aui-md-h4 mt-6 mb-4 scroll-m-20 text-xl font-semibold tracking-tight first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h5: ({ className, ...props }) => (
    <h5
      className={cn(
        "aui-md-h5 my-4 text-lg font-semibold first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h6: ({ className, ...props }) => (
    <h6
      className={cn(
        "aui-md-h6 my-4 font-semibold first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  p: ({ className, ...props }) => (
    <p
      className={cn(
        "aui-md-p mt-5 mb-5 leading-7 first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  a: ({ className, ...props }) => (
    <a
      className={cn(
        "aui-md-a font-medium text-primary underline underline-offset-4",
        className,
      )}
      {...props}
    />
  ),
  blockquote: ({ className, ...props }) => (
    <blockquote
      className={cn("aui-md-blockquote border-l-2 pl-6 italic", className)}
      {...props}
    />
  ),
  ul: ({ className, ...props }) => (
    <ul
      className={cn("aui-md-ul my-5 ml-6 list-disc [&>li]:mt-2", className)}
      {...props}
    />
  ),
  ol: ({ className, ...props }) => (
    <ol
      className={cn("aui-md-ol my-5 ml-6 list-decimal [&>li]:mt-2", className)}
      {...props}
    />
  ),
  hr: ({ className, ...props }) => (
    <hr className={cn("aui-md-hr my-5 border-b", className)} {...props} />
  ),
  table: ({ className, ...props }) => (
    <table
      className={cn(
        "aui-md-table my-5 w-full border-separate border-spacing-0 overflow-y-auto",
        className,
      )}
      {...props}
    />
  ),
  th: ({ className, ...props }) => (
    <th
      className={cn(
        "aui-md-th bg-muted px-4 py-2 text-left font-bold first:rounded-tl-lg last:rounded-tr-lg [&[align=center]]:text-center [&[align=right]]:text-right",
        className,
      )}
      {...props}
    />
  ),
  td: ({ className, ...props }) => (
    <td
      className={cn(
        "aui-md-td border-b border-l px-4 py-2 text-left last:border-r [&[align=center]]:text-center [&[align=right]]:text-right",
        className,
      )}
      {...props}
    />
  ),
  tr: ({ className, ...props }) => (
    <tr
      className={cn(
        "aui-md-tr m-0 border-b p-0 first:border-t [&:last-child>td:first-child]:rounded-bl-lg [&:last-child>td:last-child]:rounded-br-lg",
        className,
      )}
      {...props}
    />
  ),
  sup: ({ className, ...props }) => (
    <sup
      className={cn("aui-md-sup [&>a]:text-xs [&>a]:no-underline", className)}
      {...props}
    />
  ),
  pre: ({ className, ...props }) => (
    <pre
      className={cn(
        "aui-md-pre overflow-x-auto !rounded-t-none rounded-b-lg bg-black p-4 text-white",
        className,
      )}
      {...props}
    />
  ),
  code: function Code({ className, ...props }) {
    const isCodeBlock = useIsMarkdownCodeBlock();
    return (
      <code
        className={cn(
          !isCodeBlock &&
            "aui-md-inline-code rounded border bg-muted font-semibold",
          className,
        )}
        {...props}
      />
    );
  },
  CodeHeader,
});

type BnSections = {
  updates: string;
  probabilities: string;
  deltas: string;
  inferenceTiming: string;
  assistant: string;
};

const extractBnSections = (text: string): BnSections | null => {
  const match = text.match(
    /Bayesian network updates:\s*([\s\S]*?)\n{2,}Updated probabilities:\s*([\s\S]*?)\n{2,}Probability deltas:\s*([\s\S]*?)\n{2,}Inference timing:\s*([\s\S]*?)\n{2,}Assistant response:\s*([\s\S]*)/i,
  );

  if (!match) return null;

  const [, updates, probabilities, deltas, inferenceTiming, assistant] = match;
  return {
    updates: updates.trim(),
    probabilities: probabilities.trim(),
    deltas: deltas.trim(),
    inferenceTiming: inferenceTiming.trim(),
    assistant: assistant.trim(),
  };
};

const BnSidePanels: FC<{
  updates: string;
  probabilities: string;
  deltas: string;
  inferenceTiming: string;
}> = ({ updates, probabilities, deltas, inferenceTiming }) => {
  const [updatesOpen, setUpdatesOpen] = useState(false);
  const [probabilitiesOpen, setProbabilitiesOpen] = useState(false);
  const [deltasOpen, setDeltasOpen] = useState(false);

  if (!updates && !probabilities && !deltas) return null;

  return (
    <div className="aui-bn-side-buttons mb-3 flex flex-col items-end gap-2 md:absolute md:-right-40 md:top-0 md:mb-0">
      {updates && (
        <Sheet open={updatesOpen} onOpenChange={setUpdatesOpen}>
          <SheetTrigger asChild>
            <Button size="sm" variant="outline" className="shadow-sm">
              BN updates
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="sm:w-[420px]">
            <SheetHeader>
              <SheetTitle>Bayesian network updates</SheetTitle>
              <SheetDescription>Evidence provided to the BN.</SheetDescription>
            </SheetHeader>
            <div className="mt-4 max-h-[70vh] overflow-y-auto rounded-lg bg-muted/40 p-3 text-sm">
              <pre className="whitespace-pre-wrap break-words text-foreground">
                {updates || "None"}
              </pre>
            </div>
          </SheetContent>
        </Sheet>
      )}

      {probabilities && (
        <Sheet open={probabilitiesOpen} onOpenChange={setProbabilitiesOpen}>
          <SheetTrigger asChild>
            <Button size="sm" variant="outline" className="shadow-sm">
              Probabilities
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="sm:w-[420px]">
            <SheetHeader>
              <SheetTitle>Updated probabilities</SheetTitle>
              <SheetDescription>
                Full posterior over all variables.
              </SheetDescription>
            </SheetHeader>
            <div className="mt-4 max-h-[70vh] overflow-y-auto pr-1">
              <MarkdownTextPrimitive
                remarkPlugins={[remarkGfm]}
                className="aui-md"
                components={defaultComponents}
                preprocess={() => probabilities}
              />
            </div>
          </SheetContent>
        </Sheet>
      )}

      {deltas && (
        <Sheet open={deltasOpen} onOpenChange={setDeltasOpen}>
          <SheetTrigger asChild>
            <Button size="sm" variant="outline" className="shadow-sm">
              Probability deltas
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="sm:w-[420px]">
            <SheetHeader>
              <SheetTitle>Probability deltas</SheetTitle>
              <SheetDescription>
                Difference vs. baseline (no evidence).
              </SheetDescription>
            </SheetHeader>
            <div className="mt-4 max-h-[70vh] overflow-y-auto pr-1">
              <MarkdownTextPrimitive
                remarkPlugins={[remarkGfm]}
                className="aui-md"
                components={defaultComponents}
                preprocess={() => deltas}
              />
            </div>
          </SheetContent>
        </Sheet>
      )}

      {inferenceTiming && (
        <div className="aui-bn-inference-info w-full max-w-[230px] rounded-lg border bg-background/90 px-3 py-2 text-xs shadow-sm">
          <p className="font-semibold text-muted-foreground">Inference timing</p>
          <pre className="mt-1 whitespace-pre-wrap text-foreground">
            {inferenceTiming}
          </pre>
        </div>
      )}
    </div>
  );
};
