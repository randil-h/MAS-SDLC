"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { createRun, getRun, RunDetails, PipelineStep } from "@/lib/api";

type ActiveTab = "requirements" | "code" | "tests" | "review";

type ModelOption = {
  value: string;
  label: string;
  tier: "light" | "medium" | "heavy";
  note: string;
};

const MODEL_OPTIONS: ModelOption[] = [
  { value: "tinyllama",    label: "TinyLlama 1.1B",  tier: "light",  note: "~600 MB — fastest, great for dev" },
  { value: "qwen2:0.5b",  label: "Qwen2 0.5B",      tier: "light",  note: "~350 MB — ultra-light" },
  { value: "gemma:2b",    label: "Gemma 2B",         tier: "light",  note: "~1.4 GB — lightweight" },
  { value: "phi3:mini",   label: "Phi-3 Mini 3.8B",  tier: "light",  note: "~2.2 GB — smart & fast" },
  { value: "mistral:7b",  label: "Mistral 7B",       tier: "medium", note: "~4.1 GB — balanced" },
  { value: "qwen:7b",     label: "Qwen 7B",          tier: "medium", note: "~4.5 GB — balanced" },
  { value: "llama3:8b",   label: "Llama 3 8B",       tier: "heavy",  note: "~4.7 GB — recommended" },
  { value: "codellama:7b",label: "Code Llama 7B",    tier: "heavy",  note: "~3.8 GB — code-focused" },
  { value: "__custom__",  label: "Custom model...",  tier: "light",  note: "Enter a model name manually" },
];

const TIER_STYLES: Record<ModelOption["tier"], string> = {
  light:  "text-green-600 border-green-400 bg-green-50",
  medium: "text-amber-600 border-amber-400 bg-amber-50",
  heavy:  "text-stone-500 border-stone-300 bg-stone-50",
};

const STEP_STATUS_STYLES: Record<PipelineStep["status"], string> = {
  pending:   "bg-stone-200",
  running:   "bg-stone-800 shadow-[0_0_0_4px_rgba(17,17,17,0.1)]",
  completed: "bg-green-500",
  failed:    "bg-red-500",
};

const STEP_LABEL_STYLES: Record<PipelineStep["status"], string> = {
  pending:   "text-stone-400",
  running:   "text-stone-500",
  completed: "text-green-600",
  failed:    "text-red-600",
};

const RUN_STATUS_STYLES: Record<string, string> = {
  queued:    "text-stone-400",
  running:   "text-stone-700",
  completed: "text-green-600",
  failed:    "text-red-600",
};

const TABS: { key: ActiveTab; label: string }[] = [
  { key: "requirements", label: "Requirements" },
  { key: "code",         label: "Generated Code" },
  { key: "tests",        label: "Test Results" },
  { key: "review",       label: "Review Report" },
];

const DEFAULT_PROMPT =
  "Build a password reset module with secure token generation and expiry validation.";

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

/* ── Small reusable pieces ───────────────────────────────────────────────── */

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-bold tracking-[0.16em] uppercase text-stone-400 mb-5">
      {children}
    </p>
  );
}

function FieldLabel({ htmlFor, children }: { htmlFor?: string; children: React.ReactNode }) {
  return (
    <label
      htmlFor={htmlFor}
      className="block text-[11px] font-semibold tracking-[0.07em] uppercase text-stone-500 mb-2"
    >
      {children}
    </label>
  );
}

function inputClass() {
  return "w-full bg-stone-50 border border-stone-200 rounded-lg text-sm text-stone-900 placeholder:text-stone-400 px-3.5 py-2.5 focus:outline-none focus:border-stone-400 transition-colors font-sans appearance-none";
}

/* ── Main component ──────────────────────────────────────────────────────── */

export function Dashboard() {
  const [prompt, setPrompt]             = useState(DEFAULT_PROMPT);
  const [selectedModel, setSelectedModel] = useState("phi3:mini");
  const [customModel, setCustomModel]   = useState("");
  const [ollamaUrl, setOllamaUrl]       = useState("http://localhost:11434");
  const [numCtx, setNumCtx]             = useState(2048);
  const [runId, setRunId]               = useState<string | null>(null);
  const [run, setRun]                   = useState<RunDetails | null>(null);
  const [tab, setTab]                   = useState<ActiveTab>("requirements");
  const [reviewViewMode, setReviewViewMode] = useState<"text" | "code">("text");
  const [copied, setCopied]             = useState(false);
  const [submitError, setSubmitError]   = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const resolvedModel =
    selectedModel === "__custom__" ? customModel.trim() : selectedModel;

  const selectedOption = MODEL_OPTIONS.find((m) => m.value === selectedModel);

  const copyReviewMarkdown = async () => {
    const content = run?.result?.review_report ?? "";
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  /* polling — exponential back-off, stops after 5 consecutive hard errors */
  useEffect(() => {
    if (!runId) return;
    let active = true;
    let consecutiveErrors = 0;
    let timeoutId: ReturnType<typeof setTimeout>;

    const BASE_INTERVAL_MS  = 1500;
    const MAX_INTERVAL_MS   = 12000;
    const MAX_ERRORS        = 5;

    const poll = async () => {
      try {
        const d = await getRun(runId);
        if (!active) return;
        consecutiveErrors = 0;
        setRun(d);

        // Stop polling once the run reaches a terminal state
        if (d.status === "completed" || d.status === "failed") return;

        timeoutId = setTimeout(poll, BASE_INTERVAL_MS);
      } catch (e) {
        if (!active) return;
        consecutiveErrors += 1;

        if (consecutiveErrors >= MAX_ERRORS) {
          setSubmitError(
            "Lost connection to the API server. Please check that uvicorn is still running, " +
            "then refresh the page."
          );
          return; // stop polling entirely
        }

        // exponential back-off: 1.5 s → 3 s → 6 s → 12 s
        const delay = Math.min(
          BASE_INTERVAL_MS * Math.pow(2, consecutiveErrors - 1),
          MAX_INTERVAL_MS
        );
        timeoutId = setTimeout(poll, delay);
      }
    };

    poll();
    return () => {
      active = false;
      clearTimeout(timeoutId);
    };
  }, [runId]);

  const isActive = isSubmitting || run?.status === "queued" || run?.status === "running";

  const progressPct = run?.progress_percent ?? 0;

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSubmitError("");
    setIsSubmitting(true);
    setRun(null);
    try {
      const s = await createRun({ user_prompt: prompt.trim(), model_name: resolvedModel, ollama_base_url: ollamaUrl.trim(), num_ctx: numCtx });
      setRunId(s.run_id);
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const canSubmit = !isActive && !!prompt.trim() && (selectedModel !== "__custom__" || !!customModel.trim());

  /* ── Render ── */
  return (
    <div className="max-w-screen-xl mx-auto px-6 pb-20">

      {/* Header */}
      <header className="py-14 border-b border-stone-100 mb-12">
        <p className="text-[10px] font-bold tracking-[0.2em] uppercase text-stone-400 mb-4">
          MAS SDLC — Multi-Agent Pipeline
        </p>
        <h1 className="text-6xl sm:text-7xl lg:text-7xl font-semibold tracking-tighter leading-none text-stone-900">
          AI Software<br />Delivery Team
        </h1>
        <p className="mt-5 max-w-xl text-stone-500 text-[15px] leading-relaxed">
          Four specialised agents work sequentially to turn a plain-English
          feature request into requirements, working code, tests, and a review —
          all running locally via selectable models.
        </p>
      </header>

      {/* Workspace grid */}
      <div className="grid grid-cols-1 lg:grid-cols-[480px_1fr] gap-5">

        {/* ── Left: form ── */}
        <section className="bg-white  rounded-xl p-7">
          <SectionLabel>New Pipeline Run</SectionLabel>

          <form onSubmit={onSubmit} className="space-y-5">
            {/* Prompt */}
            <div>
              <FieldLabel htmlFor="prompt">Feature Request</FieldLabel>
              <textarea
                id="prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                minLength={5}
                required
                rows={6}
                placeholder="Describe the feature you want to build..."
                className={inputClass() + " resize-y min-h-[120px]"}
              />
            </div>

            {/* Model + URL */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <FieldLabel htmlFor="model">Model</FieldLabel>
                <select
                  id="model"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className={inputClass()}
                >
                  {MODEL_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
                {selectedOption && selectedOption.value !== "__custom__" && (
                  <div className="flex items-center gap-2 mt-2">
                    <span className={`text-[9px] font-bold tracking-widest uppercase px-1.5 py-0.5 rounded border ${TIER_STYLES[selectedOption.tier]}`}>
                      {selectedOption.tier}
                    </span>
                    <span className="text-[11px] text-stone-400">{selectedOption.note}</span>
                  </div>
                )}
              </div>

              <div>
                <FieldLabel htmlFor="ollama-url">Ollama URL</FieldLabel>
                <input
                  id="ollama-url"
                  value={ollamaUrl}
                  onChange={(e) => setOllamaUrl(e.target.value)}
                  required
                  className={inputClass()}
                />
              </div>
            </div>

            {/* Context window */}
            <div>
              <FieldLabel htmlFor="num-ctx">Context Window (tokens)</FieldLabel>
              <select
                id="num-ctx"
                value={numCtx}
                onChange={(e) => setNumCtx(Number(e.target.value))}
                className={inputClass()}
              >
                <option value={512}>512 — minimal memory</option>
                <option value={1024}>1024 — light</option>
                <option value={2048}>2048 — default (recommended)</option>
                <option value={4096}>4096 — larger responses</option>
                <option value={8192}>8192 — needs lots of RAM</option>
              </select>
              <p className="text-[11px] text-stone-400 mt-1.5">
                Smaller = less RAM/VRAM pressure. Raise only if outputs are getting cut off.
              </p>
            </div>

            {/* Custom model input */}
            {selectedModel === "__custom__" && (
              <div>
                <FieldLabel htmlFor="custom-model">Custom Model Name</FieldLabel>
                <input
                  id="custom-model"
                  value={customModel}
                  onChange={(e) => setCustomModel(e.target.value)}
                  placeholder="e.g. llama3:70b"
                  required
                  className={inputClass()}
                />
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={!canSubmit}
              className={`w-full py-3 rounded-lg text-sm font-bold tracking-wide transition-all ${
                isActive
                  ? "bg-stone-100 text-stone-400 border border-stone-200 cursor-not-allowed"
                  : "bg-stone-900 text-white hover:bg-stone-700 active:scale-[0.98] cursor-pointer"
              } disabled:opacity-40 disabled:cursor-not-allowed`}
            >
              {isActive ? "Pipeline running..." : "Run SDLC Pipeline"}
            </button>
          </form>

          {submitError && (
            <div className="mt-4 px-4 py-3 bg-red-50 rounded-lg text-red-600 text-sm">
              {submitError}
            </div>
          )}
        </section>

        {/* ── Right: live status ── */}
        <section className="bg-white rounded-xl p-7">
          <SectionLabel>Live Status</SectionLabel>

          {!run ? (
            <p className="text-stone-400 text-sm">No active run. Start a pipeline to watch progress here.</p>
          ) : (
            <>
              {/* KPI row */}
              <div className="grid grid-cols-3 gap-3 mb-6">
                {[
                  { label: "Status",       value: run.status.toUpperCase(), extra: RUN_STATUS_STYLES[run.status] },
                  { label: "Progress",     value: `${progressPct}%`,         extra: "text-stone-900" },
                  { label: "Active Agent", value: run.current_step_label ?? "—", extra: "text-stone-900 text-xs" },
                ].map(({ label, value, extra }) => (
                  <div key={label} className="border border-stone-100 rounded-lg p-4">
                    <p className="text-[10px] font-bold tracking-[0.14em] uppercase text-stone-400 mb-1.5">{label}</p>
                    <p className={`font-bold text-sm leading-tight ${extra}`}>{value}</p>
                  </div>
                ))}
              </div>

              {/* Progress bar */}
              <div className="h-[2px] bg-stone-100 rounded-full overflow-hidden mb-7">
                <div
                  className="h-full bg-stone-900 rounded-full transition-all duration-500"
                  style={{ width: `${progressPct}%` }}
                />
              </div>

              {/* Timeline */}
              <ol className="space-y-0">
                {run.steps.map((step, i) => (
                  <li
                    key={step.key}
                    className={`flex items-start gap-4 py-3.5 ${i < run.steps.length - 1 ? "border-b border-stone-50" : ""}`}
                  >
                    <span className={`mt-1 w-2.5 h-2.5 rounded-full shrink-0 ${STEP_STATUS_STYLES[step.status]}`} />
                    <div>
                      <p className={`text-sm font-semibold ${step.status === "pending" ? "text-stone-400" : "text-stone-900"}`}>
                        {step.label}
                      </p>
                      <p className={`text-[11px] font-semibold tracking-widest uppercase mt-0.5 ${STEP_LABEL_STYLES[step.status]}`}>
                        {step.status}
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
            </>
          )}
        </section>

        {/* ── Full-width: outputs ── */}
        <section className="lg:col-span-2 bg-white rounded-xl p-7">
          <SectionLabel>Pipeline Outputs</SectionLabel>

          {/* Tab bar */}
          <div className="flex border-b border-stone-100 mb-6 -mx-1">
            {TABS.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`px-4 py-2.5 text-sm font-semibold tracking-wide border-b-2 transition-colors ${
                  tab === key
                    ? "border-stone-900 text-stone-900"
                    : "border-transparent text-stone-400 hover:text-stone-600"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Review tab controls */}
          {tab === "review" && run?.result && (
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-0.5 p-0.5 bg-stone-100 rounded-lg">
                <button
                  onClick={() => setReviewViewMode("text")}
                  title="Preview"
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                    reviewViewMode === "text"
                      ? "bg-white shadow-sm text-stone-900"
                      : "text-stone-400 hover:text-stone-600"
                  }`}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                  Preview
                </button>
                <button
                  onClick={() => setReviewViewMode("code")}
                  title="Raw markdown"
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                    reviewViewMode === "code"
                      ? "bg-white shadow-sm text-stone-900"
                      : "text-stone-400 hover:text-stone-600"
                  }`}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="16 18 22 12 16 6" />
                    <polyline points="8 6 2 12 8 18" />
                  </svg>
                  Raw
                </button>
              </div>
              {run.result.review_report && (
                <button
                  onClick={copyReviewMarkdown}
                  title="Copy markdown"
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-stone-500 hover:text-stone-900 hover:bg-stone-100 transition-all"
                >
                  {copied ? (
                    <>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                      Copied!
                    </>
                  ) : (
                    <>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                      </svg>
                      Copy
                    </>
                  )}
                </button>
              )}
            </div>
          )}

          {!run?.result ? (
            <p className="text-stone-400 text-sm py-4">Outputs appear here once the pipeline completes.</p>
          ) : tab === "review" && reviewViewMode === "text" ? (
            <div className="bg-stone-50 border border-stone-100 rounded-lg p-5 max-h-[500px] overflow-auto">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({...props}) => <h1 className="text-2xl font-bold border-b border-stone-200 pb-2 mb-4 mt-2 text-stone-900" {...props} />,
                  h2: ({...props}) => <h2 className="text-xl font-semibold border-b border-stone-200 pb-1.5 mb-3 mt-5 text-stone-900" {...props} />,
                  h3: ({...props}) => <h3 className="text-base font-semibold mb-2 mt-4 text-stone-900" {...props} />,
                  h4: ({...props}) => <h4 className="text-sm font-semibold mb-1.5 mt-3 text-stone-800" {...props} />,
                  p: ({...props}) => <p className="mb-3 text-[13px] text-stone-700 leading-relaxed" {...props} />,
                  ul: ({...props}) => <ul className="list-disc pl-5 mb-3 space-y-0.5 text-[13px] text-stone-700" {...props} />,
                  ol: ({...props}) => <ol className="list-decimal pl-5 mb-3 space-y-0.5 text-[13px] text-stone-700" {...props} />,
                  li: ({...props}) => <li className="leading-relaxed" {...props} />,
                  blockquote: ({...props}) => <blockquote className="border-l-4 border-stone-300 pl-4 py-0.5 italic text-stone-500 my-3" {...props} />,
                  pre: ({...props}) => <pre className="bg-white border border-stone-200 rounded-md p-3.5 overflow-auto text-[12px] leading-6 font-mono mb-3" {...props} />,
                  code: (props) => {
                    const { className, children } = props;
                    const isBlock = /language-/.test(className ?? "");
                    return isBlock
                      ? <code className={`font-mono text-[12px] text-stone-700 ${className ?? ""}`}>{children}</code>
                      : <code className="bg-stone-100 text-stone-800 px-1 py-0.5 rounded text-[12px] font-mono">{children}</code>;
                  },
                  a: ({...props}) => <a className="text-blue-600 hover:underline" {...props} />,
                  hr: () => <hr className="border-stone-200 my-4" />,
                  strong: ({...props}) => <strong className="font-semibold text-stone-900" {...props} />,
                  table: ({...props}) => <div className="overflow-auto mb-3"><table className="min-w-full border-collapse border border-stone-200 text-[13px]" {...props} /></div>,
                  thead: ({...props}) => <thead className="bg-stone-100" {...props} />,
                  th: ({...props}) => <th className="border border-stone-200 px-3 py-1.5 text-left font-semibold text-stone-700 text-[12px]" {...props} />,
                  td: ({...props}) => <td className="border border-stone-200 px-3 py-1.5 text-stone-700" {...props} />,
                  tr: ({...props}) => <tr className="even:bg-stone-50" {...props} />,
                }}
              >
                {run.result.review_report ?? "No report generated."}
              </ReactMarkdown>
            </div>
          ) : (
            <pre className="bg-stone-50 border border-stone-100 rounded-lg p-5 max-h-[500px] overflow-auto text-[12.5px] leading-7 text-stone-700 font-mono whitespace-pre-wrap break-words">
              {tab === "requirements" && prettyJson(run.result.requirements)}
              {tab === "code"         && (run.result.generated_code      ?? "No code generated.")}
              {tab === "tests"        && prettyJson(run.result.test_results)}
              {tab === "review"       && (run.result.review_report        ?? "No report generated.")}
            </pre>
          )}

          {(run?.errors?.length ?? 0) > 0 && (
            <div className="mt-5 border border-red-200 rounded-lg p-5">
              <p className="text-[10px] font-bold tracking-[0.16em] uppercase text-red-500 mb-3">
                Errors & Warnings
              </p>
              <ul className="list-disc pl-4 space-y-1.5">
                {run!.errors.map((msg, i) => (
                  <li key={i} className="text-red-700 text-sm">{msg}</li>
                ))}
              </ul>
            </div>
          )}
        </section>

      </div>
    </div>
  );
}
