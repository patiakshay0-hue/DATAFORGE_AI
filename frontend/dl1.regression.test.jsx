/**
 * Regression test for the Deep Learning 2.0 results panel.
 *
 * The bug this pins down: the poll marked a run done by calling setJob, and
 * `job.status` was a dependency of that same effect, so React tore the effect
 * down in the commit the response triggered and abandoned the in-flight result
 * request. Training completed, the progress panel unmounted, and nothing
 * replaced it — results simply never appeared.
 *
 * The test drives the real component against a mocked API that behaves the way
 * the backend does: a few "running" polls, then "done", then the payload.
 */
import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";

// Mocked before the component is imported, so it binds to these.
const status = vi.fn();
const result = vi.fn();

vi.mock("./src/api", () => ({
  API: "http://test",
  api: { get: (url) => status(url) },
  apiLong: { get: (url) => result(url), post: vi.fn() },
  errorMessage: (e, fallback) => e?.response?.data?.detail || fallback,
  isNetworkError: () => false,
  wakeBackend: vi.fn(),
}));

// jsdom implements neither of these; the theme provider and recharts need them.
global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};
window.matchMedia = (query) => ({
  matches: false,
  media: query,
  onchange: null,
  addListener() {},
  removeListener() {},
  addEventListener() {},
  removeEventListener() {},
  dispatchEvent: () => false,
});

const { default: DeepLearning1View } = await import(
  "./src/components/DeepLearning1View.jsx"
);
const { ThemeProvider } = await import("./src/ThemeContext.jsx");

const JOB = "abc123";
const running = (progress, stage) => ({
  data: { job_id: JOB, status: "running", stage, progress, message: `${stage}…`, error: null,
          filename: "customers.csv", target_column: "" },
});
const done = {
  data: { job_id: JOB, status: "done", stage: "ready", progress: 100,
          message: "Found 2 pattern(s)", error: null, filename: "customers.csv", target_column: "" },
};
const PAYLOAD = {
  data: {
    ...done.data,
    profile: { rows: 15000, rows_total: 15000, sampled: false, columns_total: 12,
               columns_used: 12, columns_dropped: 0, numeric: 8, categorical: 4,
               missing_total: 0, features_used: [], features_dropped: [], column_spec: [] },
    config: { config: { latent_dim: 5, hidden_layers: [32, 16], epochs: 60, learning_rate: 0.001,
                        batch_size: 128, activation: "relu", optimizer: "adam", dropout: 0.1,
                        loss_function: "mse", early_stopping: {}, rationale: ["because"] } },
    training: { engine: "scikit-learn", architecture: ["Input · 12 features"], n_params: 4321,
                training_time: "106.09s", epochs_run: 60, epochs_requested: 60, stopped_early: false,
                best_epoch: 60, final_loss: 0.12, ae_error: 0.12, pca_error: 0.13,
                nonlinear_gain: 0.08, history: [{ epoch: 1, train_loss: 0.9, val_loss: 0.8 }],
                feature_error: [{ feature: "income", reconstruction_error: 0.2, error_pct: 30.0 }] },
    patterns: [
      { id: "information_value", type: "information_value", title: "'income' carries the most unique information",
        description: "d", confidence: 0.8, columns: ["income"], visualization: "bar",
        recommendation: "keep", data: { items: [] } },
      { id: "anomalies", type: "anomalies", title: "142 rows do not fit the learned structure",
        description: "d", confidence: 0.6, columns: ["balance"], visualization: "scatter",
        recommendation: "review", data: { count: 142, share: 0.01, drivers: [], indices: [] } },
    ],
    selected_patterns: [], preferred_pattern: null,
  },
};

/** Click "Use customers.csv", the way a user starts a run. */
const startRun = async () => {
  const { apiLong } = await import("./src/api");
  apiLong.post.mockResolvedValue({ data: running(5, "preprocess").data });
  const btn = await screen.findByText(/Use customers\.csv/i);
  await act(async () => {
    fireEvent.click(btn.closest("button"));
  });
};

const mount = () =>
  render(
    <ThemeProvider>
      <DeepLearning1View data={{ filename: "customers.csv" }} />
    </ThemeProvider>,
  );

describe("Deep Learning 2.0 results", () => {
  beforeEach(() => {
    status.mockReset();
    result.mockReset();
    result.mockResolvedValue(PAYLOAD);
  });

  it("renders the results panel once the run reports done", async () => {
    status
      .mockResolvedValueOnce(running(25, "train"))
      .mockResolvedValueOnce(running(75, "discover"))
      .mockResolvedValue(done);

    mount();
    await startRun();

    await waitFor(() => expect(screen.getByText(/columns →|dimensions/i)).toBeTruthy(),
                  { timeout: 15000 });
    expect(result).toHaveBeenCalledWith(`/dl1/result/${JOB}`);
    expect(screen.getByText(/carries the most unique information/)).toBeTruthy();
    expect(screen.getByText(/do not fit the learned structure/)).toBeTruthy();
  }, 30000);

  it("still renders results when the payload is slow to arrive", async () => {
    status.mockResolvedValue(done);
    result.mockImplementation(
      () => new Promise((r) => setTimeout(() => r(PAYLOAD), 2500)),
    );

    mount();
    await startRun();

    await waitFor(() => expect(screen.getByText(/carries the most unique information/)).toBeTruthy(),
                  { timeout: 20000 });
  }, 30000);
});
