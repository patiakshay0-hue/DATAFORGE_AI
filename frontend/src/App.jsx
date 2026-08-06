import React, { useState, useEffect, useCallback } from "react";
import {
  FileUp,
  BarChart3,
  Brain,
  LayoutDashboard,
  Database,
  Lightbulb,
  ChevronRight,
  Network,
  Images,
  FileCog,
  Cpu,
  Shield,
  Zap,
  MessageSquare,
  Loader2,
  Download,
  Sun,
  Moon,
  Sparkles,
  Menu,
  X,
} from "lucide-react";
import { useTheme } from "./ThemeContext";
import FileUpload from "./components/FileUpload";
import EDAView from "./components/EDAView";
import DashboardView from "./components/DashboardView";
import MLView from "./components/MLView";
import SmartConfigView from "./components/SmartConfigView";
import DeepLearningView from "./components/DeepLearningView";
import DeepLearning1View from "./components/DeepLearning1View";
import ImageClassifierView from "./components/ImageClassifierView";
import ImportConvertView from "./components/ImportConvertView";
import PreviewView from "./components/PreviewView";
import InsightsView from "./components/InsightsView";
import ChatView from "./components/ChatView";

const TABS = [
  { id: "convert", label: "Import & Convert", icon: FileCog, alwaysOn: true },
  { id: "upload", label: "Upload Data", icon: FileUp, alwaysOn: true },
  { id: "preview", label: "Data Preview", icon: Database },
  { id: "eda", label: "Automated EDA", icon: BarChart3 },
  { id: "insights", label: "AI Insights", icon: Lightbulb },
  { id: "ml", label: "ML Models", icon: Brain },
  { id: "deep", label: "Deep Learning", icon: Brain, alwaysOn: true },
  { id: "deep1", label: "Deep Learning 2.0", icon: Sparkles, alwaysOn: true },
  { id: "vision", label: "Image Classifier", icon: Images, alwaysOn: true },
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "chat", label: "Chat with Data", icon: MessageSquare, pro: true },
];

const App = () => {
  const { isDark, toggleTheme } = useTheme();
  const [activeTab, setActiveTab] = useState("upload");
  const [data, setData] = useState(null);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL}/`).catch(() => {});
  }, []);

  const handleUploadSuccess = (response) => {
    setData(response);
    setActiveTab("preview");
    setSidebarOpen(false);
  };

  const navigateTab = useCallback((id) => {
    setActiveTab(id);
    setSidebarOpen(false);
  }, []);

  useEffect(() => {
    if (sidebarOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => { document.body.style.overflow = ""; };
  }, [sidebarOpen]);

  const handleExport = async () => {
    if (!data || exporting) return;
    setExporting(true);
    setExportError(null);
    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/export`);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Export failed");
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${data.filename.replace(/\.[^.]+$/, "")}_report.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setExportError(err.message);
      setTimeout(() => setExportError(null), 5000);
    } finally {
      setExporting(false);
    }
  };

  const activeLabel = TABS.find((t) => t.id === activeTab)?.label || "";

  const SidebarContent = () => (
    <>
      <div className="px-6 py-7" style={{ borderBottom: "1px solid var(--df-border)" }}>
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: "linear-gradient(135deg, #14b8a6, #2dd4bf)" }}
          >
            <Cpu size={18} className="text-white" />
          </div>
          <div>
            <h1
              className="font-bold text-base leading-none"
              style={{ color: "var(--df-t1)", fontFamily: "var(--font-display)" }}
            >
              DataForge AI
            </h1>
            <p className="text-[10px] mt-0.5 uppercase tracking-widest" style={{ color: "var(--df-t3)" }}>
              Analytics Platform
            </p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-5 space-y-1 overflow-y-auto" aria-label="Main navigation">
        <p className="text-[10px] font-bold uppercase tracking-widest px-3 mb-3" style={{ color: "var(--df-t4)" }}>
          Navigation
        </p>
        {TABS.map((tab) => {
          const disabled = !tab.alwaysOn && !data;
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => navigateTab(tab.id)}
              disabled={disabled}
              role="tab"
              aria-selected={active}
              aria-disabled={disabled}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium"
              style={
                active
                  ? {
                      background: isDark
                        ? "linear-gradient(90deg, rgba(20,184,166,0.12) 0%, rgba(45,212,191,0.06) 100%)"
                        : "linear-gradient(90deg, rgba(20,184,166,0.08) 0%, rgba(45,212,191,0.04) 100%)",
                      borderLeft: "2px solid var(--df-primary)",
                      color: "var(--df-t1)",
                    }
                  : disabled
                    ? {
                        borderLeft: "2px solid transparent",
                        color: "var(--df-t4)",
                        cursor: "not-allowed",
                        opacity: 0.5,
                      }
                    : {
                        borderLeft: "2px solid transparent",
                        color: "var(--df-t2)",
                      }
              }
            >
              <tab.icon size={16} style={{ color: active ? "var(--df-primary)" : "inherit" }} />
              <span className="flex-1 text-left">{tab.label}</span>
              {tab.pro && (
                <span className="text-[8px] font-black uppercase tracking-widest px-1.5 py-0.5 rounded-full"
                  style={{ color: "var(--df-primary)", background: "rgba(20,184,166,0.1)", border: "1px solid rgba(20,184,166,0.2)" }}>
                  Pro
                </span>
              )}
              {active && <ChevronRight size={14} style={{ color: "var(--df-primary)", opacity: 0.6 }} />}
            </button>
          );
        })}
      </nav>

      <div className="px-5 py-5 space-y-3" style={{ borderTop: "1px solid var(--df-border)" }}>
        {data ? (
          <div className="rounded-xl px-4 py-3" style={{ background: "rgba(20,184,166,0.08)", border: "1px solid rgba(20,184,166,0.2)" }}>
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: "var(--df-primary)" }} />
              <span className="text-xs font-semibold" style={{ color: "var(--df-primary)" }}>Dataset Loaded</span>
            </div>
            <p className="text-xs truncate" style={{ color: "var(--df-t2)" }}>{data.filename}</p>
            <p className="text-[10px] mt-0.5" style={{ color: "var(--df-t3)" }}>
              {data.eda?.rows} rows · {data.eda?.columns} cols
            </p>
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ background: isDark ? "rgba(30,41,59,0.7)" : "#f1f5f9" }}>
              <Shield size={13} style={{ color: "var(--df-t3)" }} />
            </div>
            <div>
              <p className="text-xs font-medium" style={{ color: "var(--df-t2)" }}>No data loaded</p>
              <p className="text-[10px]" style={{ color: "var(--df-t3)" }}>Upload to begin</p>
            </div>
          </div>
        )}

        <button
          onClick={toggleTheme}
          className="w-full flex items-center justify-between px-3 py-2 rounded-xl"
          style={{
            background: isDark ? "rgba(30,41,59,0.5)" : "#f1f5f9",
            border: "1px solid var(--df-border)",
          }}
          aria-label={`Switch to ${isDark ? "light" : "dark"} mode`}
        >
          <span className="flex items-center gap-2 text-xs font-medium" style={{ color: "var(--df-t2)" }}>
            {isDark ? <Moon size={13} style={{ color: "var(--df-primary)" }} /> : <Sun size={13} style={{ color: "#f59e0b" }} />}
            {isDark ? "Dark Mode" : "Light Mode"}
          </span>
          <div className="relative w-9 h-5 rounded-full"
            style={{ background: isDark ? "rgba(20,184,166,0.25)" : "#ccfbf1" }}>
            <div className="absolute top-0.5 w-4 h-4 rounded-full shadow-sm"
              style={{
                left: isDark ? "2px" : "18px",
                background: isDark ? "#0f172a" : "var(--df-primary)",
              }} />
          </div>
        </button>
      </div>
    </>
  );

  return (
    <div className="min-h-screen flex" style={{ background: "var(--df-bg)" }}>
      <a href="#main-content" className="skip-link">Skip to main content</a>

      {/* ── Desktop Sidebar ──────────────────────────────────────────────── */}
      <aside
        className="w-64 shrink-0 flex-col max-lg:hidden"
        style={{
          background: "var(--df-sidebar)",
          borderRight: "1px solid var(--df-border)",
          display: "flex",
        }}
        role="complementary"
        aria-label="Application sidebar"
      >
        <SidebarContent />
      </aside>

      {/* ── Mobile Overlay ─────────────────────────────────────────────── */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 lg:hidden"
          style={{ background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)" }}
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* ── Mobile Drawer ──────────────────────────────────────────────── */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-72 flex flex-col lg:hidden ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
        style={{
          background: "var(--df-sidebar-solid)",
          borderRight: "1px solid var(--df-border)",
          transition: "transform 300ms cubic-bezier(0.16, 1, 0.3, 1)",
        }}
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
      >
        <div className="flex items-center justify-end px-4 pt-4">
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-2 rounded-lg"
            style={{ color: "var(--df-t2)" }}
            aria-label="Close navigation menu"
          >
            <X size={20} />
          </button>
        </div>
        <SidebarContent />
      </aside>

      {/* ── Main Content ────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-h-screen overflow-hidden">
        <header
          className="shrink-0 flex items-center justify-between px-4 lg:px-8 py-3 lg:py-4"
          style={{
            background: "var(--df-header)",
            backdropFilter: "blur(12px)",
            borderBottom: "1px solid var(--df-border)",
          }}
          role="banner"
        >
          <div className="flex items-center gap-3">
            <button
              className="lg:hidden p-2 rounded-lg"
              style={{ color: "var(--df-t2)", background: "var(--df-input-bg)" }}
              onClick={() => setSidebarOpen(true)}
              aria-label="Open navigation menu"
            >
              <Menu size={20} />
            </button>
            <div>
              <h2 className="font-semibold text-lg" style={{ color: "var(--df-t1)", fontFamily: "var(--font-display)" }}>
                {activeLabel}
              </h2>
              <p className="text-xs mt-0.5" style={{ color: "var(--df-t3)" }}>
                {data ? `Analyzing: ${data.filename}` : "Upload a dataset to begin"}
              </p>
            </div>
          </div>

          {data && (
            <div className="flex items-center gap-2 lg:gap-3">
              <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs"
                style={{
                  background: isDark ? "rgba(30,41,59,0.8)" : "#f1f5f9",
                  border: "1px solid var(--df-border)",
                  color: "var(--df-t2)",
                }}>
                <Zap size={12} style={{ color: "var(--df-primary)" }} />
                {data.eda?.rows?.toLocaleString()} records
              </div>
              <div className="relative">
                <button
                  onClick={handleExport}
                  disabled={exporting}
                  className="flex items-center gap-2 px-4 py-2 disabled:opacity-60 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium"
                  style={{ background: "var(--df-primary)" }}
                >
                  {exporting ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                  <span className="hidden sm:inline">{exporting ? "Generating…" : "Export Report"}</span>
                </button>
                {exportError && (
                  <div className="absolute right-0 top-full mt-2 w-64 bg-red-500/10 border border-red-500/30 text-red-400 text-xs rounded-xl px-3 py-2 z-50">
                    {exportError}
                  </div>
                )}
              </div>
            </div>
          )}
        </header>

        <main id="main-content" className="flex-1 overflow-auto p-4 lg:p-8" role="main" tabIndex={-1}>
          {activeTab === "convert" && <ImportConvertView onDataLoaded={handleUploadSuccess} onNavigate={navigateTab} />}
          {activeTab === "upload" && <FileUpload onUploadSuccess={handleUploadSuccess} />}
          {activeTab === "preview" && data && <PreviewView data={data} />}
          {activeTab === "eda" && data && <EDAView data={data.eda} onDataUpdated={setData} />}
          {activeTab === "insights" && data && <InsightsView onNavigate={navigateTab} />}
          {activeTab === "ml" && data && <MLView data={data} />}
          {activeTab === "smart" && data && <SmartConfigView data={data} />}
          {activeTab === "deep" && <DeepLearningView data={data} />}
          {activeTab === "deep1" && <DeepLearning1View data={data} />}
          {activeTab === "vision" && <ImageClassifierView />}
          {activeTab === "dashboard" && data && <DashboardView data={data.eda} />}
          {activeTab === "chat" && data && <ChatView data={data} />}
        </main>
      </div>
    </div>
  );
};

export default App;
