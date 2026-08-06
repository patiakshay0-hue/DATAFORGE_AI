import React, { useState, useEffect, useMemo } from "react";
import axios from "axios";
import ReactApexChart from "react-apexcharts";
import {
  Cpu,
  Sparkles,
  Loader2,
  Target,
  Layers,
  Gauge,
  BarChart3,
  Network,
  TrendingUp,
  CheckCircle2,
  ArrowRight,
  AlertCircle,
  Download,
  Zap,
  RotateCcw,
  FileText,
  Eye,
  Brain,
  PieChart,
  ScatterChart as ScatterIcon,
} from "lucide-react";
import { useTheme } from "../ThemeContext";

const API = import.meta.env.VITE_API_URL;

const SmartConfigView = ({ data }) => {
  const { isDark } = useTheme();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedPattern, setSelectedPattern] = useState(null);
  const [showReport, setShowReport] = useState(false);

  const columns = data?.schema?.map((c) => c.name) || [];

  useEffect(() => {
    if (data) {
      fetchAutoConfig();
    }
  }, [data]);

  const fetchAutoConfig = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(`${API}/deep/auto-config`, {
        target_column: null,
      });
      setResult(res.data);
      setSelectedPattern(0);
    } catch (e) {
      setError(e.response?.data?.detail || "Failed to analyze data");
    } finally {
      setLoading(false);
    }
  };

  const patterns = result?.patterns;
  const cfg = result?.config;
  const profile = result?.data_profile;

  const card = `rounded-xl p-5 border ${isDark ? "border-slate-800" : "border-slate-200 shadow-sm"}`;
  const chartTheme = isDark
    ? { mode: "dark", palette: "palette2" }
    : { mode: "light", palette: "palette2" };

  if (showReport) {
    return (
      <ReportView
        data={data}
        result={result}
        selectedPattern={selectedPattern}
        patterns={patterns}
        cfg={cfg}
        profile={profile}
        isDark={isDark}
        onBack={() => setShowReport(false)}
      />
    );
  }

  return (
    <div className="space-y-6">
      <div
        className="relative overflow-hidden rounded-2xl p-6"
        style={{
          background: "var(--df-card)",
          border: "1px solid var(--df-border)",
        }}
      >
        <div
          className="absolute inset-0 opacity-10"
          style={{
            background:
              "radial-gradient(circle at 50% 0%, #14b8a6, transparent 60%)",
          }}
        />
        <div className="relative flex items-center gap-4">
          <div className="p-3 bg-teal-500/10 border border-teal-500/20 rounded-xl shrink-0">
            <Brain size={22} className="text-teal-400" />
          </div>
          <div className="flex-1 min-w-0">
            <h3
              className="text-lg font-black"
              style={{ color: "var(--df-t1)" }}
            >
              Smart Configuration Studio
            </h3>
            <p className="text-xs mt-0.5" style={{ color: "var(--df-t3)" }}>
              Auto-select optimal hyperparameters and discover hidden patterns
              in your data
            </p>
          </div>
        </div>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="text-center space-y-4">
            <div className="relative w-16 h-16 mx-auto">
              <div
                className="absolute inset-0 rounded-full border-2"
                style={{ borderColor: "var(--df-border)" }}
              />
              <div className="absolute inset-0 rounded-full border-2 border-teal-400 border-t-transparent animate-spin" />
              <Brain
                size={20}
                className="absolute inset-0 m-auto text-teal-400"
              />
            </div>
            <p
              className="text-sm font-medium"
              style={{ color: "var(--df-t2)" }}
            >
              Analyzing data patterns...
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
          <AlertCircle size={16} className="shrink-0" /> {error}
        </div>
      )}

      {result && !loading && (
        <>
          <div className={card} style={{ background: "var(--df-card)" }}>
            <div className="flex items-center gap-2 mb-4">
              <Sparkles size={16} className="text-teal-400" />
              <h4
                className="font-bold text-sm"
                style={{ color: "var(--df-t1)" }}
              >
                Auto-Optimized Configuration
              </h4>
              <span className="text-[9px] font-black uppercase tracking-widest text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-1.5 py-0.5 rounded-full ml-2">
                AI Recommended
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
              <ConfigCard
                label="Hidden Layers"
                value={cfg?.hidden_layers?.join(" → ")}
                icon={Layers}
                color="text-teal-400"
              />
              <ConfigCard
                label="Epochs"
                value={cfg?.epochs}
                icon={TrendingUp}
                color="text-teal-400"
              />
              <ConfigCard
                label="Learning Rate"
                value={cfg?.learning_rate}
                icon={Gauge}
                color="text-amber-400"
              />
              <ConfigCard
                label="Batch Size"
                value={cfg?.batch_size}
                icon={Cpu}
                color="text-emerald-400"
              />
              <ConfigCard
                label="Dropout"
                value={cfg?.dropout}
                icon={Zap}
                color="text-rose-400"
              />
            </div>
            <div
              className="mt-4 pt-4 flex items-center gap-4 flex-wrap text-xs"
              style={{
                borderTop: "1px solid var(--df-border)",
                color: "var(--df-t3)",
              }}
            >
              <span>
                <strong>{profile?.n_samples}</strong> samples
              </span>
              <span>
                <strong>{profile?.n_features}</strong> features
              </span>
              <span className="capitalize">
                <strong>{profile?.task}</strong> task
              </span>
              {profile?.n_classes && (
                <span>
                  <strong>{profile?.n_classes}</strong> classes
                </span>
              )}
            </div>
          </div>

          <div className={card} style={{ background: "var(--df-card)" }}>
            <div className="flex items-center gap-2 mb-4">
              <Eye size={16} className="text-teal-400" />
              <h4
                className="font-bold text-sm"
                style={{ color: "var(--df-t1)" }}
              >
                Hidden Patterns in Data
              </h4>
              <span className="text-[9px] font-black uppercase tracking-widest text-teal-400 bg-teal-500/10 border border-teal-500/20 px-1.5 py-0.5 rounded-full ml-2">
                Select Best Fit
              </span>
            </div>
            <p className="text-xs mb-4" style={{ color: "var(--df-t3)" }}>
              Each card shows a different hidden pattern with a chart. Click the
              one that best represents your data.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <PatternChartCard
                title="PCA Clusters"
                selected={selectedPattern === 0}
                onClick={() => setSelectedPattern(0)}
                isDark={isDark}
                chartType="scatter"
                series={[
                  {
                    name: "Data points",
                    data: (patterns?.pca?.points || []).map((p) => ({
                      x: p.x,
                      y: p.y,
                    })),
                  },
                ]}
                desc={
                  patterns?.pca?.explained_variance_ratio?.[0] != null
                    ? `PC1 explains ${(patterns.pca.explained_variance_ratio[0] * 100).toFixed(1)}% variance — data has clear directional spread`
                    : "Analyzing principal components"
                }
                color="#14b8a6"
              />
              <PatternChartCard
                title={`Natural Groups (k=${patterns?.clusters?.n_clusters || "?"})`}
                selected={selectedPattern === 1}
                onClick={() => setSelectedPattern(1)}
                isDark={isDark}
                chartType="bar"
                series={[
                  {
                    name: "Count",
                    data: (() => {
                      const counts = {};
                      (patterns?.clusters?.labels || []).forEach((l) => {
                        counts[l] = (counts[l] || 0) + 1;
                      });
                      return Object.values(counts);
                    })(),
                  },
                ]}
                categories={(() => {
                  const counts = {};
                  (patterns?.clusters?.labels || []).forEach((l) => {
                    counts[l] = (counts[l] || 0) + 1;
                  });
                  return Object.keys(counts).map((k) => `C${+k + 1}`);
                })()}
                desc={
                  patterns?.clusters?.silhouette_score
                    ? `Silhouette ${patterns.clusters.silhouette_score} — ${patterns.clusters.silhouette_score > 0.3 ? "well-separated clusters" : "moderate overlap"}`
                    : "Cluster analysis complete"
                }
                color="#10b981"
              />
              <PatternChartCard
                title="Feature Dominance"
                selected={selectedPattern === 2}
                onClick={() => setSelectedPattern(2)}
                isDark={isDark}
                chartType="bar"
                series={[
                  {
                    name: "Variance",
                    data: (patterns?.feature_rank || [])
                      .slice(0, 6)
                      .map((f) => f.variance),
                  },
                ]}
                categories={(patterns?.feature_rank || [])
                  .slice(0, 6)
                  .map((f) => f.feature)}
                desc={
                  patterns?.feature_rank?.[0]
                    ? `"${patterns.feature_rank[0].feature}" dominates — drives most variation`
                    : "Ranking features by variance"
                }
                color="#f59e0b"
              />
              <PatternChartCard
                title="PCA Loadings"
                selected={selectedPattern === 3}
                onClick={() => setSelectedPattern(3)}
                isDark={isDark}
                chartType="bar"
                series={[
                  {
                    name: "PC1",
                    data: (patterns?.pca?.loadings || [])
                      .slice(0, 6)
                      .map((l) => l.pc1),
                  },
                  {
                    name: "PC2",
                    data: (patterns?.pca?.loadings || [])
                      .slice(0, 6)
                      .map((l) => l.pc2),
                  },
                ]}
                categories={(patterns?.pca?.loadings || [])
                  .slice(0, 6)
                  .map((l) => l.feature)}
                desc={(() => {
                  const top = patterns?.pca?.loadings
                    ?.slice()
                    .sort((a, b) => Math.abs(b.pc1) - Math.abs(a.pc1))[0];
                  return top
                    ? `"${top.feature}" (PC1: ${top.pc1?.toFixed(2)}) — top contributor to data spread`
                    : "Feature contributions to principal components";
                })()}
                color="#f43f5e"
              />
            </div>
          </div>

          <div className={card} style={{ background: "var(--df-card)" }}>
            <div className="flex items-center gap-2 mb-3">
              <Target size={16} className="text-teal-400" />
              <h4
                className="font-bold text-sm"
                style={{ color: "var(--df-t1)" }}
              >
                Select Target for Report
              </h4>
            </div>
            <p className="text-xs mb-3" style={{ color: "var(--df-t3)" }}>
              Choose what you want to predict — the report will recommend which
              columns to use
            </p>
            <div className="flex flex-wrap gap-2.5">
              {columns.map((col) => (
                <button
                  key={col}
                  onClick={() => setSelectedPattern(columns.indexOf(col))}
                  className="flex items-center gap-2 px-3.5 py-2.5 rounded-xl border text-sm transition-all"
                  style={{
                    background:
                      selectedPattern === columns.indexOf(col)
                        ? isDark
                          ? "rgba(20,184,166,0.12)"
                          : "rgba(20,184,166,0.06)"
                        : "var(--df-input-bg)",
                    borderColor:
                      selectedPattern === columns.indexOf(col)
                        ? "rgba(20,184,166,0.5)"
                        : "var(--df-border)",
                  }}
                >
                  {col}
                  {selectedPattern === columns.indexOf(col) && (
                    <CheckCircle2 size={14} className="text-teal-400" />
                  )}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-3 pt-2">
            <button
              onClick={() => setShowReport(true)}
              disabled={selectedPattern == null}
              className="flex items-center gap-2.5 px-7 py-3 rounded-xl font-bold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed disabled:scale-100"
              style={{
                background: "linear-gradient(135deg, #14b8a6, #2dd4bf)",
              }}
            >
              <FileText size={15} /> Generate Summarized Report
            </button>
            <button
              onClick={fetchAutoConfig}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-colors"
              style={{
                background: isDark ? "rgba(30,41,59,0.8)" : "#f1f5f9",
                border: "1px solid var(--df-border)",
                color: "var(--df-t2)",
              }}
            >
              <RotateCcw size={14} /> Re-analyze
            </button>
          </div>
        </>
      )}
    </div>
  );
};

const PatternChartCard = ({
  title,
  selected,
  onClick,
  isDark,
  chartType,
  series,
  categories,
  desc,
  color,
}) => {
  const border = selected ? color : isDark ? "#1e293b" : "#e2e8f0";

  const chartOptions = {
    chart: {
      type: chartType,
      height: 140,
      sparkline: { enabled: true },
      toolbar: { show: false },
      background: "transparent",
    },
    theme: { mode: isDark ? "dark" : "light" },
    colors:
      chartType === "bar" && series?.length > 1 ? [color, "#2dd4bf"] : [color],
    ...(categories ? { xaxis: { categories } } : {}),
    ...(chartType === "scatter"
      ? {
          xaxis: {
            labels: { show: false },
            axisBorder: { show: false },
            axisTicks: { show: false },
          },
          yaxis: {
            labels: { show: false },
            axisBorder: { show: false },
            axisTicks: { show: false },
          },
        }
      : {}),
    stroke: { width: chartType === "scatter" ? 0 : 0 },
    fill: { opacity: 0.8 },
    plotOptions:
      chartType === "bar"
        ? { bar: { columnWidth: "60%", borderRadius: 2 } }
        : {},
    grid: { show: false },
    tooltip: { enabled: false },
  };

  return (
    <button
      onClick={onClick}
      className="w-full text-left rounded-xl border transition-all overflow-hidden"
      style={{
        background: selected ? `${color}08` : "var(--df-input-bg)",
        borderColor: selected ? color : "var(--df-border)",
      }}
    >
      <div className="h-[100px]">
        <ReactApexChart
          options={chartOptions}
          series={series}
          type={chartType === "scatter" ? "scatter" : "bar"}
          height={100}
        />
      </div>
      <div className="px-4 pb-4">
        <div className="flex items-center gap-2 mb-1">
          <span
            className="font-semibold text-sm"
            style={{ color: "var(--df-t1)" }}
          >
            {title}
          </span>
          {selected && (
            <CheckCircle2 size={14} className="text-emerald-400 shrink-0" />
          )}
        </div>
        <p
          className="text-xs leading-relaxed"
          style={{ color: "var(--df-t3)" }}
        >
          {desc}
        </p>
      </div>
    </button>
  );
};

const ConfigCard = ({ label, value, icon: Icon, color }) => (
  <div
    className="rounded-xl p-4"
    style={{
      background: "var(--df-input-bg)",
      border: "1px solid var(--df-border)",
    }}
  >
    <div className="flex items-center gap-2 mb-2">
      <Icon size={13} className={color} />
      <span
        className="text-[10px] font-semibold uppercase tracking-wider"
        style={{ color: "var(--df-t3)" }}
      >
        {label}
      </span>
    </div>
    <p
      className="font-bold text-sm font-mono"
      style={{ color: "var(--df-t1)" }}
    >
      {typeof value === "number" && value < 0.01 ? value.toFixed(4) : value}
    </p>
  </div>
);

// ── Report View ──────────────────────────────────────────────────────────────
const ReportView = ({
  data,
  result,
  selectedPattern,
  patterns,
  cfg,
  profile,
  isDark,
  onBack,
}) => {
  const columns = data?.schema?.map((c) => c.name) || [];
  const target = columns[selectedPattern];
  const featureRank = patterns?.feature_rank || [];
  const pcaLoadings = patterns?.pca?.loadings || [];
  const task = profile?.task || "auto";

  const recommendedCols = useMemo(() => {
    const highVar = featureRank
      .slice(0, Math.min(5, featureRank.length))
      .map((f) => f.feature);
    const highPca = pcaLoadings
      .filter((l) => Math.abs(l.pc1) > 0.1 || Math.abs(l.pc2) > 0.1)
      .slice(0, 5)
      .map((l) => l.feature);
    const combined = [...new Set([...highVar, ...highPca])];
    return combined.length > 0
      ? combined
      : columns.filter((c) => c !== target).slice(0, 8);
  }, [featureRank, pcaLoadings, columns, target]);

  const reportData = useMemo(() => {
    const schema = data?.schema || [];
    const colDetails = {};
    columns.forEach((col) => {
      const s = schema.find((c) => c.name === col);
      colDetails[col] = s
        ? { type: s.type || "unknown", nullable: s.nullable }
        : { type: "unknown", nullable: false };
    });
    return colDetails;
  }, [data, columns]);

  const apexTheme = { mode: isDark ? "dark" : "light", palette: "palette2" };

  const pcaChart = {
    series: [
      {
        name: "Data",
        data: (patterns?.pca?.points || []).map((p) => ({ x: p.x, y: p.y })),
      },
    ],
    options: {
      chart: {
        type: "scatter",
        height: 220,
        toolbar: { show: false },
        background: "transparent",
      },
      theme: apexTheme,
      colors: ["#14b8a6"],
      xaxis: {
        labels: { show: false },
        axisBorder: { show: false },
        axisTicks: { show: false },
      },
      yaxis: {
        labels: { show: false },
        axisBorder: { show: false },
        axisTicks: { show: false },
      },
      grid: { show: false },
      tooltip: { enabled: false },
    },
  };

  const clusterCounts = {};
  (patterns?.clusters?.labels || []).forEach((l) => {
    clusterCounts[l] = (clusterCounts[l] || 0) + 1;
  });
  const clusterChart = {
    series: [{ name: "Count", data: Object.values(clusterCounts) }],
    options: {
      chart: {
        type: "bar",
        height: 200,
        toolbar: { show: false },
        background: "transparent",
      },
      theme: apexTheme,
      colors: ["#10b981"],
      xaxis: {
        categories: Object.keys(clusterCounts).map((k) => `Cluster ${+k + 1}`),
      },
      grid: { show: false },
      plotOptions: { bar: { columnWidth: "50%", borderRadius: 3 } },
      tooltip: { enabled: false },
    },
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div
        className="relative overflow-hidden rounded-2xl p-6"
        style={{
          background: isDark
            ? "linear-gradient(135deg, rgba(20,184,166,0.1) 0%, var(--df-card) 50%, rgba(45,212,191,0.1) 100%)"
            : "linear-gradient(135deg, #f0f9ff 0%, #ffffff 50%, #eef2ff 100%)",
          border: "1px solid rgba(20,184,166,0.25)",
        }}
      >
        <div className="flex items-center gap-4 flex-wrap">
          <div className="p-4 bg-teal-500/15 border border-teal-500/25 rounded-2xl shrink-0">
            <FileText size={28} className="text-teal-400" />
          </div>
          <div className="flex-1 min-w-0">
            <p
              className="text-xs uppercase tracking-widest font-semibold"
              style={{ color: "var(--df-t3)" }}
            >
              Summarized Report
            </p>
            <h3
              className="text-2xl font-black mt-0.5"
              style={{ color: "var(--df-t1)" }}
            >
              Data Configuration Summary
            </h3>
            <p className="text-sm mt-0.5" style={{ color: "var(--df-t2)" }}>
              Target:{" "}
              <span className="text-teal-400 font-semibold">{target}</span> ·{" "}
              {profile?.n_samples} rows · {profile?.n_features} features ·{" "}
              <span className="capitalize">{task}</span>
            </p>
          </div>
        </div>
      </div>

      <div
        className="rounded-2xl p-6 border"
        style={{
          background: "var(--df-card)",
          borderColor: "var(--df-border)",
        }}
      >
        <h4
          className="font-bold text-sm flex items-center gap-2 mb-4"
          style={{ color: "var(--df-t1)" }}
        >
          <Sparkles size={15} className="text-teal-400" /> Recommended
          Hyperparameters
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <ReportMetric
            label="Hidden Layers"
            value={cfg?.hidden_layers?.join(" → ")}
          />
          <ReportMetric label="Epochs" value={cfg?.epochs} />
          <ReportMetric label="Learning Rate" value={cfg?.learning_rate} />
          <ReportMetric label="Batch Size" value={cfg?.batch_size} />
          <ReportMetric label="Dropout" value={cfg?.dropout} />
        </div>
      </div>

      <div
        className="rounded-2xl p-6 border"
        style={{
          background: "var(--df-card)",
          borderColor: "var(--df-border)",
        }}
      >
        <h4
          className="font-bold text-sm flex items-center gap-2 mb-4"
          style={{ color: "var(--df-t1)" }}
        >
          <CheckCircle2 size={15} className="text-emerald-400" /> Recommended
          Columns to Use
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {recommendedCols.map((col) => (
            <div
              key={col}
              className="flex items-center gap-2.5 rounded-xl px-4 py-3"
              style={{
                background: "var(--df-input-bg)",
                border: "1px solid var(--df-border)",
              }}
            >
              <div className="w-2 h-2 rounded-full bg-emerald-400 shrink-0" />
              <div>
                <span
                  className="text-sm font-medium"
                  style={{ color: "var(--df-t1)" }}
                >
                  {col}
                </span>
                <span
                  className="text-[10px] ml-2 uppercase"
                  style={{ color: "var(--df-t3)" }}
                >
                  {reportData[col]?.type}
                </span>
              </div>
            </div>
          ))}
        </div>
        <p className="text-xs mt-3" style={{ color: "var(--df-t3)" }}>
          Selected based on variance, PCA loadings, and relevance to target{" "}
          <strong className="text-teal-400">{target}</strong>
        </p>
      </div>

      <div
        className="rounded-2xl p-6 border"
        style={{
          background: "var(--df-card)",
          borderColor: "var(--df-border)",
        }}
      >
        <h4
          className="font-bold text-sm flex items-center gap-2 mb-4"
          style={{ color: "var(--df-t1)" }}
        >
          <BarChart3 size={15} className="text-teal-400" /> All Features &
          Column Details
        </h4>
        <div className="overflow-x-auto">
          <table className="text-xs w-full" style={{ color: "var(--df-t2)" }}>
            <thead>
              <tr
                className="border-b"
                style={{ borderColor: "var(--df-border)" }}
              >
                <th
                  className="text-left py-2 pr-3 font-semibold"
                  style={{ color: "var(--df-t3)" }}
                >
                  Column
                </th>
                <th
                  className="text-left py-2 pr-3 font-semibold"
                  style={{ color: "var(--df-t3)" }}
                >
                  Type
                </th>
                <th
                  className="text-left py-2 pr-3 font-semibold"
                  style={{ color: "var(--df-t3)" }}
                >
                  Nullable
                </th>
                <th
                  className="text-left py-2 font-semibold"
                  style={{ color: "var(--df-t3)" }}
                >
                  Recommendation
                </th>
              </tr>
            </thead>
            <tbody>
              {columns.map((col) => {
                const isTarget = col === target;
                const isRecommended = recommendedCols.includes(col);
                return (
                  <tr
                    key={col}
                    className="border-t"
                    style={{ borderColor: "var(--df-border)" }}
                  >
                    <td
                      className="py-2 pr-3 font-medium"
                      style={{ color: isTarget ? "#38bdf8" : "var(--df-t1)" }}
                    >
                      {col}{" "}
                      {isTarget && (
                        <span className="text-[8px] font-black text-teal-400 bg-teal-500/10 px-1 py-0.5 rounded-full ml-1">
                          TARGET
                        </span>
                      )}
                    </td>
                    <td className="py-2 pr-3">{reportData[col]?.type}</td>
                    <td className="py-2 pr-3">
                      {reportData[col]?.nullable ? "Yes" : "No"}
                    </td>
                    <td className="py-2">
                      {isTarget ? (
                        <span className="text-teal-400">Target variable</span>
                      ) : isRecommended ? (
                        <span className="text-emerald-400 font-semibold">
                          Use as feature
                        </span>
                      ) : (
                        <span style={{ color: "var(--df-t4)" }}>Optional</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div
        className="rounded-2xl p-6 border"
        style={{
          background: "var(--df-card)",
          borderColor: "var(--df-border)",
        }}
      >
        <h4
          className="font-bold text-sm flex items-center gap-2 mb-4"
          style={{ color: "var(--df-t1)" }}
        >
          <Eye size={15} className="text-amber-400" /> Hidden Patterns
          Discovered
        </h4>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div
            className="rounded-xl p-4"
            style={{
              background: "var(--df-input-bg)",
              border: "1px solid var(--df-border)",
            }}
          >
            <p
              className="text-xs font-semibold mb-2 flex items-center gap-1.5"
              style={{ color: "var(--df-t2)" }}
            >
              <ScatterIcon size={12} className="text-teal-400" /> PCA Projection
            </p>
            <ReactApexChart
              options={pcaChart.options}
              series={pcaChart.series}
              type="scatter"
              height={180}
            />
            <p className="text-[10px] mt-1" style={{ color: "var(--df-t3)" }}>
              PC1: {(patterns?.pca?.explained_variance_ratio?.[0] || 0) * 100}%
              · PC2: {(patterns?.pca?.explained_variance_ratio?.[1] || 0) * 100}
              %
            </p>
          </div>
          <div
            className="rounded-xl p-4"
            style={{
              background: "var(--df-input-bg)",
              border: "1px solid var(--df-border)",
            }}
          >
            <p
              className="text-xs font-semibold mb-2 flex items-center gap-1.5"
              style={{ color: "var(--df-t2)" }}
            >
              <PieChart size={12} className="text-emerald-400" /> Cluster
              Distribution (k={patterns?.clusters?.n_clusters || "?"})
            </p>
            <ReactApexChart
              options={clusterChart.options}
              series={clusterChart.series}
              type="bar"
              height={180}
            />
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3 pt-2 pb-8">
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-colors"
          style={{
            background: isDark ? "rgba(30,41,59,0.8)" : "#f1f5f9",
            border: "1px solid var(--df-border)",
            color: "var(--df-t2)",
          }}
        >
          <ArrowRight size={14} className="rotate-180" /> Back to Patterns
        </button>
      </div>
    </div>
  );
};

const ReportMetric = ({ label, value }) => (
  <div
    className="rounded-xl p-3"
    style={{
      background: "var(--df-input-bg)",
      border: "1px solid var(--df-border)",
    }}
  >
    <p
      className="text-[10px] font-semibold uppercase tracking-wider mb-1"
      style={{ color: "var(--df-t3)" }}
    >
      {label}
    </p>
    <p
      className="font-bold font-mono text-sm"
      style={{ color: "var(--df-t1)" }}
    >
      {value}
    </p>
  </div>
);

export default SmartConfigView;
