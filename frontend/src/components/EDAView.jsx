import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts'
import {
  BarChart3, PieChart as PieIcon, Table, Wand2, Sparkles, Loader2,
  CheckCircle2, AlertCircle, ChevronDown, ChevronUp
} from 'lucide-react'
import { useTheme } from '../ThemeContext'

const COLORS = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1']

const EDAView = ({ data, onDataUpdated }) => {
  const { isDark } = useTheme()
  const [activeStatTab, setActiveStatTab] = useState('summary')

  if (!data) return null
  const { charts = {}, summary = {}, correlation = {}, missing = {}, rows = 0, columns = 0 } = data

  const chartEntries = Object.entries(charts)
  const barCharts  = chartEntries.filter(([, v]) => v.type === 'bar')
  const pieCharts  = chartEntries.filter(([, v]) => v.type === 'pie')
  const summaryRows   = Object.entries(summary).slice(0, 10)
  const missingEntries = Object.entries(missing).filter(([, v]) => v > 0)

  const gridColor  = isDark ? '#1e293b' : '#e2e8f0'
  const axisColor  = isDark ? '#475569' : '#94a3b8'
  const ttBg       = isDark ? '#1e293b' : '#ffffff'
  const ttBorder   = isDark ? '#334155' : '#e2e8f0'

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div style={{ background: ttBg, border: `1px solid ${ttBorder}` }}
        className="rounded-xl px-4 py-3 shadow-xl text-sm">
        <p style={{ color: 'var(--df-t2)' }} className="mb-1">{label}</p>
        <p style={{ color: 'var(--df-t1)' }} className="font-bold">{payload[0]?.value}</p>
      </div>
    )
  }

  const card = `rounded-2xl p-6 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`
  const cardStyle = { background: 'var(--df-card)' }

  return (
    <div className="space-y-8">
      {/* Overview strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Total Rows',       value: rows?.toLocaleString() ?? 0,       color: 'text-sky-400' },
          { label: 'Total Columns',    value: columns,                            color: 'text-violet-400' },
          { label: 'Numeric Features', value: barCharts.length,                   color: 'text-emerald-400' },
          { label: 'Missing Cells',    value: Object.values(missing).reduce((a,b)=>a+b,0),
            color: missingEntries.length > 0 ? 'text-amber-400' : 'text-emerald-400' },
        ].map((s) => (
          <div key={s.label} className={`rounded-xl p-5 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`}
            style={cardStyle}>
            <p className="text-xs uppercase tracking-widest font-semibold" style={{ color: 'var(--df-t3)' }}>{s.label}</p>
            <p className={`text-3xl font-black mt-1 ${s.color}`}>{s.value}</p>
          </div>
        ))}
      </div>

      {/* Missing-value cleaning */}
      <CleaningPanel isDark={isDark} missing={missing} rows={rows} onDataUpdated={onDataUpdated} />

      {/* Stats table */}
      <div className={`rounded-2xl overflow-hidden border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`}
        style={cardStyle}>
        <div className="flex items-center justify-between px-6 py-5" style={{ borderBottom: '1px solid var(--df-border)' }}>
          <div className="flex items-center gap-3">
            <Table size={18} className="text-sky-400" />
            <h3 className="font-bold" style={{ color: 'var(--df-t1)' }}>Statistical Summary</h3>
          </div>
          <div className="flex text-xs">
            {['summary', 'missing'].map(t => (
              <button key={t} onClick={() => setActiveStatTab(t)}
                className={`px-4 py-1.5 rounded-lg capitalize font-medium ${
                  activeStatTab === t
                    ? 'bg-sky-500/20 text-sky-400'
                    : 'hover:text-sky-400'
                }`}
                style={{ color: activeStatTab === t ? undefined : 'var(--df-t3)' }}>
                {t}
              </button>
            ))}
          </div>
        </div>
        <div className="overflow-x-auto">
          {activeStatTab === 'summary' ? (
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--df-border)', background: isDark ? 'rgba(13,21,35,0.8)' : '#f8fafc' }}>
                  {['Feature','Count','Mean','Std','Min','Median','Max'].map(h => (
                    <th key={h} className="px-5 py-3 text-left text-[11px] uppercase tracking-wider font-semibold"
                      style={{ color: 'var(--df-t3)' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {summaryRows.map(([col, s]) => (
                  <tr key={col} className="transition-colors"
                    style={{ borderBottom: '1px solid var(--df-border)' }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--df-row-hover)'}
                    onMouseLeave={e => e.currentTarget.style.background = ''}>
                    <td className="px-5 py-3.5 font-semibold" style={{ color: 'var(--df-t1)' }}>{col}</td>
                    <td className="px-5 py-3.5" style={{ color: 'var(--df-t2)' }}>{s.count ?? '—'}</td>
                    <td className="px-5 py-3.5" style={{ color: 'var(--df-t2)' }}>{typeof s.mean === 'number' ? s.mean.toFixed(2) : '—'}</td>
                    <td className="px-5 py-3.5" style={{ color: 'var(--df-t2)' }}>{typeof s.std  === 'number' ? s.std.toFixed(2)  : '—'}</td>
                    <td className="px-5 py-3.5" style={{ color: 'var(--df-t3)' }}>{s.min ?? '—'}</td>
                    <td className="px-5 py-3.5" style={{ color: 'var(--df-t3)' }}>{s['50%'] ?? '—'}</td>
                    <td className="px-5 py-3.5" style={{ color: 'var(--df-t3)' }}>{s.max ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--df-border)', background: isDark ? 'rgba(13,21,35,0.8)' : '#f8fafc' }}>
                  {['Column','Missing Count','% Missing'].map(h => (
                    <th key={h} className="px-5 py-3 text-left text-[11px] uppercase tracking-wider font-semibold"
                      style={{ color: 'var(--df-t3)' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(missing).map(([col, count]) => (
                  <tr key={col} style={{ borderBottom: '1px solid var(--df-border)' }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--df-row-hover)'}
                    onMouseLeave={e => e.currentTarget.style.background = ''}>
                    <td className="px-5 py-3.5 font-semibold" style={{ color: 'var(--df-t1)' }}>{col}</td>
                    <td className="px-5 py-3.5">
                      <span className={`font-bold ${count > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>{count}</span>
                    </td>
                    <td className="px-5 py-3.5" style={{ color: 'var(--df-t3)' }}>
                      {rows > 0 ? `${((count / rows) * 100).toFixed(1)}%` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Distribution charts */}
      {barCharts.length > 0 && (
        <div>
          <h3 className="font-bold text-lg mb-4 flex items-center gap-2" style={{ color: 'var(--df-t1)' }}>
            <BarChart3 size={18} className="text-sky-400" /> Distributions
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
            {barCharts.map(([key, cfg]) => (
              <div key={key} className={card} style={cardStyle}>
                <p className="text-[10px] font-bold uppercase tracking-widest mb-0.5" style={{ color: 'var(--df-t3)' }}>Distribution</p>
                <h4 className="font-bold mb-5" style={{ color: 'var(--df-t1)' }}>{key.replace('dist_', '')}</h4>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={cfg.data} margin={{ top: 0, right: 0, left: -25, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                      <XAxis dataKey="bin" stroke={axisColor} fontSize={9} tickLine={false} />
                      <YAxis stroke={axisColor} fontSize={9} tickLine={false} axisLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="count" fill="#0ea5e9" radius={[3, 3, 0, 0]}>
                        {cfg.data.map((_, i) => (
                          <Cell key={i} fill={COLORS[i % COLORS.length]} fillOpacity={0.85} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Pie / categorical charts */}
      {pieCharts.length > 0 && (
        <div>
          <h3 className="font-bold text-lg mb-4 flex items-center gap-2" style={{ color: 'var(--df-t1)' }}>
            <PieIcon size={18} className="text-violet-400" /> Categorical Breakdown
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
            {pieCharts.map(([key, cfg]) => (
              <div key={key} className={card} style={cardStyle}>
                <p className="text-[10px] font-bold uppercase tracking-widest mb-0.5" style={{ color: 'var(--df-t3)' }}>Category</p>
                <h4 className="font-bold mb-5" style={{ color: 'var(--df-t1)' }}>{key.replace('count_', '')}</h4>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={cfg.data} cx="50%" cy="50%" innerRadius={45} outerRadius={65}
                        paddingAngle={3} dataKey="value">
                        {cfg.data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                      </Pie>
                      <Tooltip content={<CustomTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="grid grid-cols-2 gap-1 mt-2">
                  {cfg.data.slice(0, 6).map((d, i) => (
                    <div key={i} className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--df-t3)' }}>
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                      <span className="truncate">{d.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Missing-value cleaning panel ──────────────────────────────────────────────
const API = import.meta.env.VITE_API_URL

const METHOD_LABELS = {
  median: 'Fill with median', mean: 'Fill with mean', zero: 'Fill with 0',
  mode: 'Fill with most frequent', constant: 'Fill with value…',
  ffill: 'Forward fill', bfill: 'Backward fill',
  drop_rows: 'Drop rows with missing', drop_column: 'Drop this column',
}

const CleaningPanel = ({ isDark, missing, rows, onDataUpdated }) => {
  const [report,   setReport]   = useState(null)
  const [strat,    setStrat]    = useState({})       // column -> { method, value }
  const [open,     setOpen]     = useState(false)
  const [applying, setApplying] = useState(false)
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)

  const totalMissing = Object.values(missing).reduce((a, b) => a + b, 0)

  // (re)load the report whenever the underlying data (missing counts) changes
  useEffect(() => {
    if (totalMissing === 0) { setReport(null); return }
    axios.get(`${API}/clean/missing-report`).then(res => {
      setReport(res.data)
      const init = {}
      res.data.columns.forEach(c => { init[c.column] = { method: c.suggested, value: 'Unknown' } })
      setStrat(init)
    }).catch(() => setError('Could not load the missing-value report.'))
  }, [totalMissing])

  const apply = async (strategies) => {
    setApplying(true); setError(null); setResult(null)
    try {
      const res = await axios.post(`${API}/clean/apply`, { strategies })
      setResult(res.data.clean_summary)
      onDataUpdated?.(res.data)   // refresh the whole dataset across the app
    } catch (e) {
      setError(e.response?.data?.detail || 'Cleaning failed.')
    } finally { setApplying(false) }
  }

  const applyPerColumn = () =>
    apply(report.columns.map(c => ({ column: c.column, method: strat[c.column]?.method, value: strat[c.column]?.value })))

  const autoClean = () =>
    apply(report.columns.map(c => ({ column: c.column, method: c.suggested })))

  const card = `rounded-2xl border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`

  // All clean → success confirmation
  if (totalMissing === 0) {
    return (
      <div className={`${card} p-5 flex items-center gap-3`} style={{ background: 'var(--df-card)' }}>
        <CheckCircle2 size={18} className="text-emerald-400 shrink-0" />
        <div>
          <p className="font-semibold text-sm" style={{ color: 'var(--df-t1)' }}>No missing values</p>
          <p className="text-xs" style={{ color: 'var(--df-t3)' }}>
            {result ? `Cleaned — ${result.cells_filled} cells filled, ${result.rows_removed} rows removed${result.columns_dropped.length ? `, ${result.columns_dropped.length} column(s) dropped` : ''}.` : 'Your dataset is complete and ready to analyse.'}
          </p>
        </div>
      </div>
    )
  }

  const affected = report?.columns || []

  return (
    <div className={card} style={{ background: 'var(--df-card)' }}>
      {/* Banner */}
      <div className="flex items-center justify-between gap-4 px-6 py-5 flex-wrap">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 shrink-0">
            <Wand2 size={18} className="text-amber-400" />
          </div>
          <div>
            <h3 className="font-bold" style={{ color: 'var(--df-t1)' }}>Handle Missing Values</h3>
            <p className="text-xs" style={{ color: 'var(--df-t3)' }}>
              <span className="text-amber-400 font-semibold">{totalMissing.toLocaleString()}</span> missing cells across{' '}
              <span className="text-amber-400 font-semibold">{affected.length}</span> column(s)
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2.5">
          <button onClick={autoClean} disabled={applying || !report}
            className="flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
            style={{ background: 'linear-gradient(135deg, #10b981, #0ea5e9)' }}>
            {applying ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
            Auto-clean
          </button>
          <button onClick={() => setOpen(o => !o)}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl font-medium text-sm transition-colors"
            style={{ background: 'var(--df-input-bg)', border: '1px solid var(--df-border)', color: 'var(--df-t2)' }}>
            Customize {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>
      </div>

      {error && (
        <div className="mx-6 mb-4 flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
          <AlertCircle size={15} className="shrink-0" /> {error}
        </div>
      )}

      {/* Per-column controls */}
      {open && report && (
        <div className="px-6 pb-6" style={{ borderTop: '1px solid var(--df-border)' }}>
          <p className="text-xs mt-4 mb-3" style={{ color: 'var(--df-t3)' }}>
            Choose how to handle each column. Suggested defaults are pre-selected by column type.
          </p>
          <div className="space-y-2.5">
            {affected.map(c => {
              const cur = strat[c.column] || {}
              return (
                <div key={c.column} className="flex items-center gap-3 flex-wrap p-3 rounded-xl"
                  style={{ background: 'var(--df-input-bg)', border: '1px solid var(--df-border)' }}>
                  <div className="flex-1 min-w-[140px]">
                    <p className="font-semibold text-sm" style={{ color: 'var(--df-t1)' }}>{c.column}</p>
                    <p className="text-[11px]" style={{ color: 'var(--df-t3)' }}>
                      <span className={c.kind === 'numeric' ? 'text-sky-400' : 'text-violet-400'}>{c.kind}</span>
                      {' · '}{c.missing} missing ({c.pct}%)
                    </p>
                  </div>
                  <select value={cur.method || c.suggested}
                    onChange={e => setStrat(s => ({ ...s, [c.column]: { ...s[c.column], method: e.target.value } }))}
                    className="rounded-lg px-3 py-2 text-sm outline-none border"
                    style={{ background: 'var(--df-card)', borderColor: 'var(--df-input-border)', color: 'var(--df-t1)' }}>
                    {c.methods.map(m => <option key={m} value={m}>{METHOD_LABELS[m] || m}</option>)}
                  </select>
                  {cur.method === 'constant' && (
                    <input type="text" value={cur.value ?? ''} placeholder="value"
                      onChange={e => setStrat(s => ({ ...s, [c.column]: { ...s[c.column], value: e.target.value } }))}
                      className="rounded-lg px-3 py-2 text-sm outline-none border w-28"
                      style={{ background: 'var(--df-card)', borderColor: 'var(--df-input-border)', color: 'var(--df-t1)' }} />
                  )}
                </div>
              )
            })}
          </div>
          <button onClick={applyPerColumn} disabled={applying}
            className="mt-4 flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
            style={{ background: 'linear-gradient(135deg, #7c3aed, #6366f1)' }}>
            {applying ? <Loader2 size={14} className="animate-spin" /> : <Wand2 size={14} />}
            Apply choices
          </button>
        </div>
      )}
    </div>
  )
}

export default EDAView
