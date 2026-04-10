export type RunStatus = "queued" | "running" | "completed" | "failed";

export type PipelineStep = {
  key: string;
  label: string;
  status: "pending" | "running" | "completed" | "failed";
  started_at: string | null;
  completed_at: string | null;
};

export type RunDetails = {
  run_id: string;
  status: RunStatus;
  progress_percent: number;
  current_step_key: string | null;
  current_step_label: string | null;
  started_at: string;
  completed_at: string | null;
  steps: PipelineStep[];
  errors: string[];
  result: {
    requirements: Record<string, unknown> | null;
    generated_code: string | null;
    test_results: Record<string, unknown> | null;
    review_report: string | null;
    errors: string[];
    log_path: string;
  } | null;
  log_path: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export async function createRun(payload: {
  user_prompt: string;
  model_name: string;
  ollama_base_url: string;
  num_ctx?: number;
}): Promise<{ run_id: string }> {
  const response = await fetch(`${API_BASE}/api/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to create run: ${errorText}`);
  }

  return response.json();
}

export async function getRun(runId: string): Promise<RunDetails> {
  const response = await fetch(`${API_BASE}/api/runs/${runId}`, {
    cache: "no-store"
  });
  if (!response.ok) {
    throw new Error("Unable to fetch run state.");
  }
  return response.json();
}
