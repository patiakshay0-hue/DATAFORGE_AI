import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import {
  Sparkles, Play, Loader2, AlertCircle, RotateCcw, Layers, Cpu, Clock, Zap,
  CheckCircle2, FileDown, Upload, Star, Check, TrendingUp, Boxes,
} from 'lucide-react'
import {
  LineChart, Line, BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell,
} from 'recharts'
import { useTheme } from '../ThemeContext'

const API = import.meta.env.VITE_API_URL

// Mirrors the stage keys the backend reports, so progress stays in step with it.
const STAGES = [
  { key: 'preprocess', label: 'Analysing data' },
  { key: 'configure',  label: 'Choosing architecture' },
  { key: 'train',      label: 'Training network' },
  { key: 'discover',   label: 'Finding patterns' },
  { key: 'ready',      label: 'Ready' },
]

const TYPE_COLORS = {
  information_value: '#8b5cf6',
  compressibility:   '#0ea5e9',
  non_linear:        '#f59e0b',
  clusters:          '#10b981',
  segments:          '#14b8a6',
  anomalies:         '#ef4444',
  coupling:          '#ec4899',
  correlation:       '#6366f1',
  redundancy:        '#94a3b8',
}

const DeepLearning1View = ({ data }) => {
  const { isDark } = useTheme()
  const [job, setJob]         = useState(null)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)
  const [selected, setSelected]   = useState([])
  const [preferred, setPreferred] = useState(null)
  const [recommendation, setRecommendation] = useState(null)
  const [busy, setBusy]           = useState(false)
  const [downloading, setDownloading] = useState(false)
  const pollRef = useRef(null)

  // Poll the run while it is in flight; stop as soon as it settles.
  useEffect(() => {
    if (!job || job.status === 'done' || job.status === 'error') return
    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/dl1/status/${job.job_id}`)
        setJob(res.data)
        if (res.data.status === 'done') {
          const full = await axios.get(`${API}/dl1/result/${job.job_id}`)
          setResult(full.data)
        } else if (res.data.status === 'error') {
          setError(res.data.error || 'The run failed.')
        }
      } catch {
        setError('Lost contact with the server.')
        setJob(j => (j ? { ...j, status: 'error' } : j))
      }
    }, 700)
    return () => clearInterval(pollRef.current)
  }, [job?.job_id, job?.status])

  const start = async (file) => {
    setBusy(true); setError(null); setResult(null)
    setSelected([]); setPreferred(null); setRecommendation(null)
    try {
      const form = new FormData()
      if (file) form.append('file', file)
      const res = await axios.post(`${API}/dl1/run`, file ? form : undefined,
        file ? { headers: { 'Content-Type': 'multipart/form-data' } } : undefined)
      setJob(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Could not start the run.')
    } finally { setBusy(false) }
  }

  const toggle = (id) =>
    setSelected(s => (s.includes(id) ? s.filter(x => x !== id) : [...s, id]))

  const confirm = async () => {
    setBusy(true)
    try {
      const res = await axios.post(`${API}/dl1/select/${job.job_id}`,
        { pattern_ids: selected, preferred })
      setRecommendation(res.data.recommendation)
    } catch (e) {
      setError(e.response?.data?.detail || 'Could not save your selection.')
    } finally { setBusy(false) }
  }

  const download = async () => {
    setDownloading(true)
    try {
      const res = await axios.get(`${API}/dl1/report/${job.job_id}`, { responseType: 'blob' })
      const url = URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }))
      const a = document.createElement('a')
      a.href = url
      a.download = `${(job.filename || 'dataset').replace(/\.[^.]+$/, '')}_deep_learning_1.0.pdf`
      document.body.appendChild(a); a.click(); a.remove()
      URL.revokeObjectURL(url)
    } catch {
      setError('Could not download the report.')
    } finally { setDownloading(false) }
  }

  const reset = () => {
    clearInterval(pollRef.current)
    setJob(null); setResult(null); setError(null)
    setSelected([]); setPreferred(null); setRecommendation(null)
  }

  const card = `rounded-2xl p-6 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`
  const running = job && job.status !== 'done' && job.status !== 'error'

  return (
    <div className="space-y-6">
      <Header isDark={isDark} />

      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/25">
          <AlertCircle size={16} className="text-red-400 mt-0.5 shrink-0" />
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {!job && <StartPanel data={data} onStart={start} busy={busy} card={card} isDark={isDark} />}

      {running && <Progress job={job} card={card} isDark={isDark} />}

      {result && (
        <>
          <Summary result={result} card={card} isDark={isDark} />
          <LossChart result={result} card={card} isDark={isDark} />
          <FeatureInfo result={result} card={card} isDark={isDark} />
          <Patterns
            patterns={result.patterns} selected={selected} preferred={preferred}
            onToggle={toggle} onPrefer={setPreferred} isDark={isDark}
          />

          <div className="flex items-center justify-between gap-4 flex-wrap pt-2"
            style={{ borderTop: '1px solid var(--df-border)' }}>
            <p className="text-sm" style={{ color: 'var(--df-t3)' }}>
              {selected.length
                ? `${selected.length} pattern${selected.length > 1 ? 's' : ''} selected`
                : 'Select the patterns that matter to you'}
            </p>
            <div className="flex gap-3">
              <button onClick={reset}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium"
                style={{ background: isDark ? 'rgba(30,41,59,0.8)' : '#f1f5f9',
                         border: '1px solid var(--df-border)', color: 'var(--df-t2)' }}>
                <RotateCcw size={14} /> Start over
              </button>
              <button onClick={confirm} disabled={busy || !selected.length}
                className="flex items-center gap-2 px-6 py-2.5 rounded-xl font-bold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-40 disabled:scale-100"
                style={{ background: 'linear-gradient(135deg, #7c3aed, #6366f1)' }}>
                {busy ? <Loader2 size={14} className="animate-spin" /> : <Check size={14} />}
                Generate report
              </button>
            </div>
          </div>

          {recommendation && (
            <Report recommendation={recommendation} result={result} card={card}
              isDark={isDark} onDownload={download} downloading={downloading} />
          )}
        </>
      )}
    </div>
  )
}

// ── Header ────────────────────────────────────────────────────────────────────
const Header = ({ isDark }) => (
  <div className="relative overflow-hidden rounded-2xl p-6"
    style={{ background: 'var(--df-card)', border: '1px solid var(--df-border)' }}>
    <div className="absolute inset-0 opacity-10"
      style={{ background: 'radial-gradient(circle at 50% 0%, #8b5cf6, transparent 60%)' }} />
    <div className="relative flex items-center gap-4">
      <div className="p-3 bg-violet-500/10 border border-violet-500/20 rounded-xl shrink-0">
        <Sparkles size={22} className="text-violet-400" />
      </div>
      <div className="flex-1 min-w-0">
        <h3 className="text-lg font-black" style={{ color: 'var(--df-t1)' }}>Deep Learning 1.0</h3>
        <p className="text-xs mt-0.5" style={{ color: 'var(--df-t3)' }}>
          No target column needed — the network learns the data's own structure and reports what it found
        </p>
      </div>
    </div>
  </div>
)

// ── Start ─────────────────────────────────────────────────────────────────────
const StartPanel = ({ data, onStart, busy, card, isDark }) => {
  const loaded = data?.filename
  return (
    <div className={card} style={{ background: 'var(--df-card)' }}>
      <div className="text-center py-8">
        <div className="w-16 h-16 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center mx-auto mb-5">
          <Upload size={26} className="text-violet-400" />
        </div>
        <p className="text-xl font-bold mb-2" style={{ color: 'var(--df-t1)' }}>
          Add your data and train
        </p>
        <p className="text-sm mb-6 max-w-md mx-auto" style={{ color: 'var(--df-t3)' }}>
          An autoencoder learns to reconstruct your data through a narrow bottleneck.
          Whatever survives that squeeze is the real structure — no labels required.
        </p>

        <div className="flex items-center justify-center gap-3 flex-wrap">
          {loaded && (
            <button onClick={() => onStart(null)} disabled={busy}
              className="flex items-center gap-2 px-6 py-3 rounded-xl font-bold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
              style={{ background: 'linear-gradient(135deg, #7c3aed, #6366f1)' }}>
              {busy ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} fill="currentColor" />}
              Use {data.filename}
            </button>
          )}
          <label
            className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm cursor-pointer transition-all hover:scale-105"
            style={{ background: 'var(--df-input-bg)', border: '1px solid var(--df-border)', color: 'var(--df-t1)' }}>
            <Upload size={15} /> {loaded ? 'Upload a different file' : 'Upload a dataset'}
            <input type="file" accept=".csv,.xlsx,.xls,.json" className="hidden"
              onChange={e => e.target.files?.[0] && onStart(e.target.files[0])} />
          </label>
        </div>
        <p className="text-xs mt-4" style={{ color: 'var(--df-t4)' }}>
          CSV, XLSX or JSON · at least 10 rows
        </p>
      </div>
    </div>
  )
}

// ── Progress ──────────────────────────────────────────────────────────────────
const Progress = ({ job, card, isDark }) => {
  const activeIdx = STAGES.findIndex(s => s.key === job.stage)
  return (
    <div className={card} style={{ background: 'var(--df-card)' }}>
      <div className="max-w-lg mx-auto py-6 text-center">
        <div className="relative w-16 h-16 mx-auto mb-6">
          <div className="absolute inset-0 rounded-full border-2" style={{ borderColor: 'var(--df-border)' }} />
          <div className="absolute inset-0 rounded-full border-2 border-violet-400 border-t-transparent animate-spin" />
          <Sparkles size={20} className="absolute inset-0 m-auto text-violet-400" />
        </div>
        <p className="font-bold" style={{ color: 'var(--df-t1)' }}>{job.message || 'Working…'}</p>

        <div className="w-full h-2 rounded-full overflow-hidden mt-5"
          style={{ background: isDark ? '#1e293b' : '#e2e8f0' }}>
          <div className="h-full rounded-full transition-all duration-500"
            style={{ width: `${job.progress}%`, background: 'linear-gradient(90deg,#7c3aed,#6366f1)' }} />
        </div>
        <p className="text-xs mt-2 font-mono" style={{ color: 'var(--df-t3)' }}>{job.progress}%</p>

        <div className="mt-7 space-y-2.5 text-left">
          {STAGES.map((s, i) => {
            const done = activeIdx > i
            const active = activeIdx === i
            return (
              <div key={s.key} className="flex items-center gap-3">
                <div className="w-5 h-5 rounded-full flex items-center justify-center shrink-0"
                  style={{
                    background: done ? '#10b981' : active ? 'rgba(139,92,246,0.2)' : 'var(--df-input-bg)',
                    border: `1px solid ${done ? '#10b981' : active ? '#8b5cf6' : 'var(--df-border)'}`,
                  }}>
                  {done ? <Check size={11} className="text-white" />
                    : active ? <Loader2 size={11} className="animate-spin text-violet-400" /> : null}
                </div>
                <span className="text-sm"
                  style={{ color: done || active ? 'var(--df-t1)' : 'var(--df-t4)' }}>{s.label}</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// ── Summary ───────────────────────────────────────────────────────────────────
const Stat = ({ label, value, sub }) => (
  <div className="rounded-xl p-4" style={{ background: 'var(--df-input-bg)', border: '1px solid var(--df-border)' }}>
    <p className="text-xs mb-1" style={{ color: 'var(--df-t3)' }}>{label}</p>
    <p className="text-lg font-black" style={{ color: 'var(--df-t1)' }}>{value}</p>
    {sub && <p className="text-[11px] mt-0.5" style={{ color: 'var(--df-t4)' }}>{sub}</p>}
  </div>
)

const Summary = ({ result, card, isDark }) => {
  const cfg = result.config?.config || result.config || {}
  const t = result.training || {}
  const p = result.profile || {}
  const gain = t.nonlinear_gain

  return (
    <div className="space-y-5">
      <div className="relative overflow-hidden rounded-2xl p-6"
        style={{
          background: isDark
            ? 'linear-gradient(135deg, rgba(139,92,246,0.1) 0%, var(--df-card) 50%, rgba(99,102,241,0.1) 100%)'
            : 'linear-gradient(135deg, #f5f3ff 0%, #ffffff 50%, #eef2ff 100%)',
          border: '1px solid rgba(139,92,246,0.25)',
        }}>
        <div className="flex items-center gap-5 flex-wrap">
          <div className="p-4 bg-violet-500/15 border border-violet-500/25 rounded-2xl shrink-0">
            <Boxes size={26} className="text-violet-400" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs uppercase tracking-widest font-semibold" style={{ color: 'var(--df-t3)' }}>
              Network trained · no target used
            </p>
            <h3 className="text-2xl font-black mt-0.5" style={{ color: 'var(--df-t1)' }}>
              {p.columns_used} columns → {cfg.latent_dim} dimensions
            </h3>
            <p className="text-sm mt-1 flex items-center gap-3 flex-wrap" style={{ color: 'var(--df-t2)' }}>
              <span className="flex items-center gap-1"><Cpu size={12} /> {t.engine}</span>
              <span className="flex items-center gap-1"><Zap size={12} /> {t.n_params?.toLocaleString()} params</span>
              <span className="flex items-center gap-1"><Clock size={12} /> {t.training_time}</span>
              <span className="flex items-center gap-1">
                <TrendingUp size={12} /> {t.epochs_run} epochs{t.stopped_early ? ' (early stop)' : ''}
              </span>
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Stat label="Hidden layers" value={(cfg.hidden_layers || []).join(' → ') || '—'}
          sub={`then ${cfg.latent_dim} latent`} />
        <Stat label="Epochs" value={t.epochs_run} sub={`budget ${cfg.epochs}`} />
        <Stat label="Learning rate" value={cfg.learning_rate} sub={`batch ${cfg.batch_size}`} />
        {/* A negative gain means PCA won; showing it as a negative percentage reads
            as a bug, so report the finding in words. */}
        <Stat label="Beats linear by"
          value={typeof gain !== 'number' ? '—' : gain > 0 ? `${(gain * 100).toFixed(0)}%` : 'None'}
          sub={typeof gain === 'number' && gain <= 0 ? 'structure is linear' : 'vs PCA baseline'} />
      </div>

      {cfg.rationale?.length > 0 && (
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <h4 className="font-bold text-sm mb-3 flex items-center gap-2" style={{ color: 'var(--df-t1)' }}>
            <Layers size={14} className="text-violet-400" /> Why these settings
          </h4>
          <ul className="space-y-1.5">
            {cfg.rationale.map((r, i) => (
              <li key={i} className="text-xs flex gap-2" style={{ color: 'var(--df-t2)' }}>
                <span className="text-violet-400">•</span>{r}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// ── Loss curve ────────────────────────────────────────────────────────────────
const LossChart = ({ result, card, isDark }) => {
  const history = result.training?.history || []
  if (!history.length) return null
  const grid = isDark ? '#1e293b' : '#e2e8f0'
  const axis = isDark ? '#64748b' : '#94a3b8'
  return (
    <div className={card} style={{ background: 'var(--df-card)' }}>
      <h4 className="font-bold text-sm mb-4" style={{ color: 'var(--df-t1)' }}>Reconstruction Loss</h4>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={history} margin={{ top: 5, right: 8, left: -8, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false} />
          <XAxis dataKey="epoch" stroke={axis} fontSize={11} tickLine={false} />
          <YAxis stroke={axis} fontSize={11} tickLine={false} width={54} />
          <Tooltip contentStyle={{ background: isDark ? '#0f172a' : '#fff',
            border: '1px solid var(--df-border)', borderRadius: 12, fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line type="monotone" dataKey="train_loss" name="Train" stroke="#0ea5e9" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="val_loss" name="Validation" stroke="#f59e0b" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Per-feature information ───────────────────────────────────────────────────
const FeatureInfo = ({ result, card, isDark }) => {
  const items = result.training?.feature_error || []
  if (!items.length) return null
  const axis = isDark ? '#64748b' : '#94a3b8'
  return (
    <div className={card} style={{ background: 'var(--df-card)' }}>
      <h4 className="font-bold text-sm" style={{ color: 'var(--df-t1)' }}>Unique Information per Column</h4>
      <p className="text-[11px] mb-4" style={{ color: 'var(--df-t3)' }}>
        Higher means the column cannot be inferred from the others — it carries signal of its own
      </p>
      <ResponsiveContainer width="100%" height={Math.max(160, items.length * 30)}>
        <BarChart data={items} layout="vertical" margin={{ top: 4, right: 20, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#e2e8f0'} horizontal={false} />
          <XAxis type="number" stroke={axis} fontSize={11} tickLine={false} unit="%" />
          <YAxis type="category" dataKey="feature" stroke={axis} fontSize={11} tickLine={false} width={95} />
          <Tooltip contentStyle={{ background: isDark ? '#0f172a' : '#fff',
            border: '1px solid var(--df-border)', borderRadius: 12, fontSize: 12 }}
            formatter={v => [`${v}%`, 'Unique information']} />
          <Bar dataKey="error_pct" radius={[0, 4, 4, 0]} fill="#8b5cf6" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Pattern browser ───────────────────────────────────────────────────────────
const Patterns = ({ patterns, selected, preferred, onToggle, onPrefer, isDark }) => {
  if (!patterns?.length) {
    return (
      <div className="rounded-2xl p-6 text-center" style={{ background: 'var(--df-card)', border: '1px solid var(--df-border)' }}>
        <p className="text-sm" style={{ color: 'var(--df-t3)' }}>
          No strong patterns surfaced in this dataset.
        </p>
      </div>
    )
  }
  return (
    <div>
      <h4 className="font-bold text-sm mb-1" style={{ color: 'var(--df-t1)' }}>
        Hidden Patterns ({patterns.length})
      </h4>
      <p className="text-[11px] mb-4" style={{ color: 'var(--df-t3)' }}>
        Pick the ones that matter to you — your choice shapes the report and the recommended columns
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {patterns.map(p => {
          const on = selected.includes(p.id)
          const star = preferred === p.id
          const colour = TYPE_COLORS[p.type] || '#8b5cf6'
          return (
            <div key={p.id} onClick={() => onToggle(p.id)}
              className="rounded-2xl p-5 border cursor-pointer transition-all hover:scale-[1.01]"
              style={{
                background: on ? (isDark ? 'rgba(139,92,246,0.08)' : 'rgba(139,92,246,0.04)') : 'var(--df-card)',
                borderColor: on ? 'rgba(139,92,246,0.5)' : 'var(--df-border)',
              }}>
              <div className="flex items-start gap-3">
                <div className="w-5 h-5 rounded-md flex items-center justify-center shrink-0 mt-0.5"
                  style={{ background: on ? '#8b5cf6' : 'transparent',
                           border: `1.5px solid ${on ? '#8b5cf6' : 'var(--df-border)'}` }}>
                  {on && <Check size={12} className="text-white" />}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <p className="font-bold text-sm" style={{ color: 'var(--df-t1)' }}>{p.title}</p>
                    <button
                      onClick={e => { e.stopPropagation(); onPrefer(star ? null : p.id) }}
                      title="Mark as preferred"
                      className="shrink-0 p-1 rounded-md transition-colors">
                      <Star size={14} className={star ? 'text-amber-400' : 'opacity-30'}
                        fill={star ? 'currentColor' : 'none'} />
                    </button>
                  </div>

                  <div className="flex items-center gap-2 mt-1.5 mb-2 flex-wrap">
                    <span className="text-[9px] font-black uppercase tracking-wider px-2 py-0.5 rounded-full"
                      style={{ color: colour, background: `${colour}1a`, border: `1px solid ${colour}33` }}>
                      {p.type.replace(/_/g, ' ')}
                    </span>
                    <span className="text-[10px] font-mono" style={{ color: 'var(--df-t3)' }}>
                      {(p.confidence * 100).toFixed(0)}% confidence
                    </span>
                  </div>

                  <p className="text-xs leading-relaxed" style={{ color: 'var(--df-t2)' }}>{p.description}</p>

                  {p.columns?.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mt-2.5">
                      {p.columns.slice(0, 6).map(c => (
                        <span key={c} className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                          style={{ background: 'var(--df-input-bg)', color: 'var(--df-t2)',
                                   border: '1px solid var(--df-border)' }}>{c}</span>
                      ))}
                    </div>
                  )}

                  {p.recommendation && (
                    <p className="text-[11px] mt-2.5 italic" style={{ color: 'var(--df-t3)' }}>
                      {p.recommendation}
                    </p>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Report ────────────────────────────────────────────────────────────────────
const Report = ({ recommendation, result, card, isDark, onDownload, downloading }) => {
  const used = recommendation.features_used || []
  const ignored = recommendation.features_ignored || []
  const ranking = recommendation.ranking || []

  return (
    <div className={card} style={{ background: 'var(--df-card)' }}>
      <div className="flex items-start justify-between gap-4 flex-wrap mb-5">
        <div>
          <h4 className="font-bold text-base flex items-center gap-2" style={{ color: 'var(--df-t1)' }}>
            <CheckCircle2 size={16} className="text-emerald-400" /> Your Report
          </h4>
          <p className="text-xs mt-0.5" style={{ color: 'var(--df-t3)' }}>
            Based on {recommendation.selected_patterns?.length || 0} selected pattern(s)
          </p>
        </div>
        <button onClick={onDownload} disabled={downloading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
          style={{ background: 'linear-gradient(135deg, #059669, #10b981)' }}>
          {downloading ? <Loader2 size={14} className="animate-spin" /> : <FileDown size={14} />}
          {downloading ? 'Building…' : 'Download PDF'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
        <div className="rounded-xl p-4" style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.25)' }}>
          <p className="text-xs font-bold uppercase tracking-wider mb-2 text-emerald-400">
            Columns to use ({used.length})
          </p>
          <div className="flex flex-wrap gap-1.5">
            {used.length ? used.map(c => (
              <span key={c} className="text-[11px] font-mono px-2 py-0.5 rounded"
                style={{ background: 'var(--df-input-bg)', color: 'var(--df-t1)' }}>{c}</span>
            )) : <span className="text-xs" style={{ color: 'var(--df-t3)' }}>—</span>}
          </div>
        </div>
        <div className="rounded-xl p-4" style={{ background: 'rgba(148,163,184,0.08)', border: '1px solid var(--df-border)' }}>
          <p className="text-xs font-bold uppercase tracking-wider mb-2" style={{ color: 'var(--df-t3)' }}>
            Columns to ignore ({ignored.length})
          </p>
          <div className="flex flex-wrap gap-1.5">
            {ignored.length ? ignored.map(c => (
              <span key={c} className="text-[11px] font-mono px-2 py-0.5 rounded line-through"
                style={{ background: 'var(--df-input-bg)', color: 'var(--df-t3)' }}>{c}</span>
            )) : <span className="text-xs" style={{ color: 'var(--df-t3)' }}>none</span>}
          </div>
        </div>
      </div>

      <p className="text-xs font-bold uppercase tracking-wider mb-2" style={{ color: 'var(--df-t3)' }}>
        Feature ranking
      </p>
      <div className="space-y-1.5">
        {ranking.slice(0, 12).map(r => (
          <div key={r.feature} className="flex items-center gap-3">
            <span className="text-[10px] font-mono w-5 text-right shrink-0" style={{ color: 'var(--df-t4)' }}>{r.rank}</span>
            <span className="text-xs font-mono w-32 truncate shrink-0" style={{ color: 'var(--df-t1)' }}>{r.feature}</span>
            <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--df-input-bg)' }}>
              <div className="h-full rounded-full"
                style={{ width: `${Math.min(100, r.information_pct)}%`,
                         background: r.in_selected_pattern ? 'linear-gradient(90deg,#7c3aed,#6366f1)' : '#475569' }} />
            </div>
            <span className="text-[10px] font-mono w-12 text-right shrink-0" style={{ color: 'var(--df-t3)' }}>
              {r.information_pct.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      {recommendation.excluded_at_load?.length > 0 && (
        <div className="mt-5 pt-4" style={{ borderTop: '1px solid var(--df-border)' }}>
          <p className="text-xs font-bold uppercase tracking-wider mb-2" style={{ color: 'var(--df-t3)' }}>
            Excluded before modelling
          </p>
          {recommendation.excluded_at_load.map(e => (
            <p key={e.column} className="text-[11px]" style={{ color: 'var(--df-t3)' }}>
              <span className="font-mono" style={{ color: 'var(--df-t2)' }}>{e.column}</span> — {e.reason}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

export default DeepLearning1View
