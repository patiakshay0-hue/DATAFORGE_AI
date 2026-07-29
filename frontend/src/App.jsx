import React, { useState, useEffect } from 'react'
import {
  FileUp, BarChart3, Brain, LayoutDashboard,
  Database, Lightbulb, ChevronRight, Network, Images, FileCog,
  Cpu, Shield, Zap, MessageSquare, Loader2, Download,
  Sun, Moon
} from 'lucide-react'
import { useTheme } from './ThemeContext'
import FileUpload          from './components/FileUpload'
import EDAView             from './components/EDAView'
import DashboardView       from './components/DashboardView'
import MLView              from './components/MLView'
import DeepLearningView    from './components/DeepLearningView'
import ImageClassifierView from './components/ImageClassifierView'
import ImportConvertView   from './components/ImportConvertView'
import PreviewView         from './components/PreviewView'
import InsightsView        from './components/InsightsView'
import ChatView            from './components/ChatView'

const TABS = [
  { id: 'convert',   label: 'Import & Convert', icon: FileCog,         alwaysOn: true },
  { id: 'upload',    label: 'Upload Data',      icon: FileUp,          alwaysOn: true },
  { id: 'preview',   label: 'Data Preview',     icon: Database },
  { id: 'eda',       label: 'Automated EDA',    icon: BarChart3 },
  { id: 'insights',  label: 'AI Insights',      icon: Lightbulb },
  { id: 'ml',        label: 'ML Models',        icon: Brain },
  { id: 'deep',      label: 'Deep Learning',    icon: Network },
  { id: 'vision',    label: 'Image Classifier', icon: Images, alwaysOn: true },
  { id: 'dashboard', label: 'Dashboard',        icon: LayoutDashboard },
  { id: 'chat',      label: 'Chat with Data',   icon: MessageSquare, pro: true },
]

const App = () => {
  const { isDark, toggleTheme } = useTheme()
  const [activeTab,   setActiveTab]   = useState('upload')
  const [data,        setData]        = useState(null)
  const [exporting,   setExporting]   = useState(false)
  const [exportError, setExportError] = useState(null)
  const [backendReady, setBackendReady] = useState(true)

  // Warm up the backend on load. Render's free tier spins down after inactivity;
  // pinging it now means the server is awake by the time the user uploads,
  // instead of the first request hanging for ~60s on a cold start.
  useEffect(() => {
    let cancelled = false
    const coldTimer = setTimeout(() => { if (!cancelled) setBackendReady(false) }, 2500)
    fetch(`${import.meta.env.VITE_API_URL}/`, { cache: 'no-store' })
      .catch(() => {})
      .finally(() => { clearTimeout(coldTimer); if (!cancelled) setBackendReady(true) })
    return () => { cancelled = true; clearTimeout(coldTimer) }
  }, [])

  const handleUploadSuccess = (response) => {
    setData(response)
    setActiveTab('preview')
  }

  const handleExport = async () => {
    if (!data || exporting) return
    setExporting(true)
    setExportError(null)
    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/export`)
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Export failed')
      }
      const blob = await res.blob()
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href     = url
      a.download = `${data.filename.replace(/\.[^.]+$/, '')}_report.pdf`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      setExportError(err.message)
      setTimeout(() => setExportError(null), 5000)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="min-h-screen flex" style={{ background: 'var(--df-bg)' }}>

      {/* ── Sidebar ───────────────────────────────────────────────────────── */}
      <aside
        className="w-64 shrink-0 flex flex-col"
        style={{ background: 'var(--df-sidebar)', borderRight: '1px solid var(--df-border)' }}
      >
        {/* Logo */}
        <div className="px-6 py-7" style={{ borderBottom: '1px solid var(--df-border)' }}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{ background: 'linear-gradient(135deg, #0ea5e9, #6366f1)' }}>
              <Cpu size={18} className="text-white" />
            </div>
            <div>
              <h1 className="font-bold text-base leading-none" style={{ color: 'var(--df-t1)' }}>
                DataForge AI
              </h1>
              <p className="text-[10px] mt-0.5 uppercase tracking-widest" style={{ color: 'var(--df-t3)' }}>
                Analytics Platform
              </p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-5 space-y-1 overflow-y-auto">
          <p className="text-[10px] font-bold uppercase tracking-widest px-3 mb-3"
            style={{ color: 'var(--df-t4)' }}>
            Navigation
          </p>
          {TABS.map((tab) => {
            const disabled = !tab.alwaysOn && !data
            const active   = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => !disabled && setActiveTab(tab.id)}
                disabled={disabled}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium group"
                style={
                  active ? {
                    background: isDark
                      ? 'linear-gradient(90deg, rgba(14,165,233,0.15) 0%, rgba(99,102,241,0.08) 100%)'
                      : 'linear-gradient(90deg, rgba(14,165,233,0.1) 0%, rgba(99,102,241,0.05) 100%)',
                    borderLeft: '2px solid #0ea5e9',
                    color: 'var(--df-t1)',
                  } : disabled ? {
                    borderLeft: '2px solid transparent',
                    color: 'var(--df-t4)',
                    cursor: 'not-allowed',
                    opacity: 0.5,
                  } : {
                    borderLeft: '2px solid transparent',
                    color: 'var(--df-t2)',
                  }
                }
              >
                <tab.icon size={16} style={{ color: active ? '#38bdf8' : 'inherit' }} />
                <span className="flex-1 text-left">{tab.label}</span>
                {tab.pro && (
                  <span className="text-[8px] font-black uppercase tracking-widest text-violet-400 bg-violet-500/10 border border-violet-500/20 px-1.5 py-0.5 rounded-full">
                    Pro
                  </span>
                )}
                {active && <ChevronRight size={14} className="text-sky-500 opacity-60" />}
              </button>
            )
          })}
        </nav>

        {/* Status + Theme toggle */}
        <div className="px-5 py-5 space-y-3" style={{ borderTop: '1px solid var(--df-border)' }}>
          {data ? (
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl px-4 py-3">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-emerald-400 text-xs font-semibold">Dataset Loaded</span>
              </div>
              <p className="text-xs truncate" style={{ color: 'var(--df-t2)' }}>{data.filename}</p>
              <p className="text-[10px] mt-0.5" style={{ color: 'var(--df-t3)' }}>
                {data.eda?.rows} rows · {data.eda?.columns} cols
              </p>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-lg flex items-center justify-center"
                style={{ background: isDark ? 'rgba(30,41,59,0.7)' : '#f1f5f9' }}>
                <Shield size={13} style={{ color: 'var(--df-t3)' }} />
              </div>
              <div>
                <p className="text-xs font-medium" style={{ color: 'var(--df-t2)' }}>No data loaded</p>
                <p className="text-[10px]" style={{ color: 'var(--df-t3)' }}>Upload to begin</p>
              </div>
            </div>
          )}

          {/* Theme toggle pill */}
          <button
            onClick={toggleTheme}
            className="w-full flex items-center justify-between px-3 py-2 rounded-xl"
            style={{
              background: isDark ? 'rgba(30,41,59,0.5)' : '#f1f5f9',
              border: '1px solid var(--df-border)',
            }}
          >
            <span className="flex items-center gap-2 text-xs font-medium" style={{ color: 'var(--df-t2)' }}>
              {isDark
                ? <Moon size={13} style={{ color: '#38bdf8' }} />
                : <Sun  size={13} style={{ color: '#f59e0b' }} />
              }
              {isDark ? 'Dark Mode' : 'Light Mode'}
            </span>
            {/* Animated pill */}
            <div
              className="relative w-9 h-5 rounded-full"
              style={{ background: isDark ? 'rgba(14,165,233,0.25)' : '#bae6fd' }}
            >
              <div
                className="absolute top-0.5 w-4 h-4 rounded-full shadow-sm"
                style={{
                  left: isDark ? '2px' : '18px',
                  background: isDark ? '#0f172a' : '#0ea5e9',
                }}
              />
            </div>
          </button>
        </div>
      </aside>

      {/* ── Main Content ──────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-h-screen overflow-hidden">

        {/* Top bar */}
        <header
          className="shrink-0 flex items-center justify-between px-8 py-4"
          style={{
            background: 'var(--df-header)',
            backdropFilter: 'blur(12px)',
            borderBottom: '1px solid var(--df-border)',
          }}
        >
          <div>
            <h2 className="font-semibold text-lg" style={{ color: 'var(--df-t1)' }}>
              {TABS.find(t => t.id === activeTab)?.label}
            </h2>
            <p className="text-xs mt-0.5" style={{ color: 'var(--df-t3)' }}>
              {data ? `Analyzing: ${data.filename}` : 'Upload a dataset to begin'}
            </p>
          </div>

          {data && (
            <div className="flex items-center gap-3">
              <div
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs"
                style={{
                  background: isDark ? 'rgba(30,41,59,0.8)' : '#f1f5f9',
                  border: '1px solid var(--df-border)',
                  color: 'var(--df-t2)',
                }}
              >
                <Zap size={12} className="text-amber-400" />
                {data.eda?.rows?.toLocaleString()} records processed
              </div>

              <div className="relative">
                <button
                  onClick={handleExport}
                  disabled={exporting}
                  className="flex items-center gap-2 px-4 py-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-60 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium"
                >
                  {exporting ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                  {exporting ? 'Generating…' : 'Export Report'}
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

        {/* Cold-start notice — the free-tier backend can take ~1 min to wake */}
        {!backendReady && (
          <div className="shrink-0 flex items-center gap-2.5 px-8 py-2.5 text-xs"
            style={{ background: 'rgba(245,158,11,0.1)', borderBottom: '1px solid rgba(245,158,11,0.2)', color: '#f59e0b' }}>
            <Loader2 size={13} className="animate-spin shrink-0" />
            Waking up the server (free hosting sleeps when idle) — the first request can take up to a minute. You can start uploading; it'll go through once it's awake.
          </div>
        )}

        {/* Page content */}
        <main className="flex-1 overflow-auto p-8">
          {activeTab === 'convert'   && <ImportConvertView onDataLoaded={handleUploadSuccess} onNavigate={setActiveTab} />}
          {activeTab === 'upload'    && <FileUpload onUploadSuccess={handleUploadSuccess} />}
          {activeTab === 'preview'   && data && <PreviewView data={data} />}
          {activeTab === 'eda'       && data && <EDAView data={data.eda} onDataUpdated={setData} />}
          {activeTab === 'insights'  && data && <InsightsView onNavigate={setActiveTab} />}
          {activeTab === 'ml'        && data && <MLView data={data} />}
          {activeTab === 'deep'      && data && <DeepLearningView data={data} />}
          {activeTab === 'vision'    && <ImageClassifierView />}
          {activeTab === 'dashboard' && data && <DashboardView data={data.eda} />}
          {activeTab === 'chat'      && data && <ChatView data={data} />}
        </main>
      </div>
    </div>
  )
}

export default App
