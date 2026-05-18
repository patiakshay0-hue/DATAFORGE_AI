import React, { useState } from 'react'
import {
  FileUp, BarChart3, Brain, LayoutDashboard,
  FileText, Database, Lightbulb, ChevronRight,
  Cpu, Shield, Zap, MessageSquare, Loader2, Download
} from 'lucide-react'
import FileUpload    from './components/FileUpload'
import EDAView       from './components/EDAView'
import DashboardView from './components/DashboardView'
import MLView        from './components/MLView'
import PreviewView   from './components/PreviewView'
import InsightsView  from './components/InsightsView'
import ChatView      from './components/ChatView'

const TABS = [
  { id: 'upload',    label: 'Upload Data',   icon: FileUp,        alwaysOn: true },
  { id: 'preview',   label: 'Data Preview',  icon: Database },
  { id: 'eda',       label: 'Automated EDA', icon: BarChart3 },
  { id: 'insights',  label: 'AI Insights',   icon: Lightbulb },
  { id: 'ml',        label: 'ML Models',     icon: Brain },
  { id: 'dashboard', label: 'Dashboard',     icon: LayoutDashboard },
  { id: 'chat',      label: 'Chat with Data',icon: MessageSquare,  pro: true },
]

const App = () => {
  const [activeTab,    setActiveTab]    = useState('upload')
  const [data,         setData]         = useState(null)
  const [exporting,    setExporting]    = useState(false)
  const [exportError,  setExportError]  = useState(null)

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
    <div className="min-h-screen flex" style={{ background: '#080d18' }}>

      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside className="w-64 shrink-0 flex flex-col border-r border-slate-800/80"
        style={{ background: 'linear-gradient(180deg, #0d1523 0%, #080d18 100%)' }}>

        {/* Logo */}
        <div className="px-6 py-7 border-b border-slate-800/60">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{ background: 'linear-gradient(135deg, #0ea5e9, #6366f1)' }}>
              <Cpu size={18} className="text-white" />
            </div>
            <div>
              <h1 className="text-white font-bold text-base leading-none">DataForge AI</h1>
              <p className="text-slate-500 text-[10px] mt-0.5 uppercase tracking-widest">Analytics Platform</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-5 space-y-1 overflow-y-auto">
          <p className="text-slate-600 text-[10px] font-bold uppercase tracking-widest px-3 mb-3">Navigation</p>
          {TABS.map((tab) => {
            const disabled = !tab.alwaysOn && !data
            const active   = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => !disabled && setActiveTab(tab.id)}
                disabled={disabled}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group ${
                  active   ? 'text-white'
                  : disabled ? 'text-slate-700 cursor-not-allowed'
                             : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
                }`}
                style={active
                  ? { background: 'linear-gradient(90deg, rgba(14,165,233,0.15) 0%, rgba(99,102,241,0.08) 100%)', borderLeft: '2px solid #0ea5e9' }
                  : { borderLeft: '2px solid transparent' }}
              >
                <tab.icon size={16} className={active ? 'text-sky-400' : ''} />
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

        {/* Status */}
        <div className="px-5 py-5 border-t border-slate-800/60">
          {data ? (
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl px-4 py-3">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-emerald-400 text-xs font-semibold">Dataset Loaded</span>
              </div>
              <p className="text-slate-400 text-[11px] truncate">{data.filename}</p>
              <p className="text-slate-600 text-[10px] mt-0.5">
                {data.eda?.rows} rows · {data.eda?.columns} cols
              </p>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-lg bg-slate-800 flex items-center justify-center">
                <Shield size={13} className="text-slate-500" />
              </div>
              <div>
                <p className="text-slate-400 text-xs font-medium">Production Mode</p>
                <p className="text-slate-600 text-[10px]">No data loaded</p>
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* ── Main Content ─────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-h-screen overflow-hidden">

        {/* Top bar */}
        <header className="shrink-0 flex items-center justify-between px-8 py-4 border-b border-slate-800/60"
          style={{ background: 'rgba(8,13,24,0.8)', backdropFilter: 'blur(12px)' }}>
          <div>
            <h2 className="text-white font-semibold text-lg">
              {TABS.find(t => t.id === activeTab)?.label}
            </h2>
            <p className="text-slate-500 text-xs mt-0.5">
              {data ? `Analyzing: ${data.filename}` : 'Upload a dataset to begin'}
            </p>
          </div>

          {data && (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/80 border border-slate-700/60 rounded-lg text-slate-400 text-xs">
                <Zap size={12} className="text-amber-400" />
                {data.eda?.rows?.toLocaleString()} records processed
              </div>

              {/* Export button */}
              <div className="relative">
                <button
                  onClick={handleExport}
                  disabled={exporting}
                  className="flex items-center gap-2 px-4 py-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-60 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition-colors"
                >
                  {exporting
                    ? <Loader2 size={14} className="animate-spin" />
                    : <Download size={14} />}
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

        {/* Page content */}
        <main className="flex-1 overflow-auto p-8">
          {activeTab === 'upload'    && <FileUpload onUploadSuccess={handleUploadSuccess} />}
          {activeTab === 'preview'   && data && <PreviewView data={data} />}
          {activeTab === 'eda'       && data && <EDAView data={data.eda} />}
          {activeTab === 'insights'  && data && <InsightsView />}
          {activeTab === 'ml'        && data && <MLView data={data} />}
          {activeTab === 'dashboard' && data && <DashboardView data={data.eda} />}
          {activeTab === 'chat'      && data && <ChatView data={data} />}
        </main>
      </div>
    </div>
  )
}

export default App
