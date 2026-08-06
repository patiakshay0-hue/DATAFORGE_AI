import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Network, Play, Loader2, Target, Sparkles, AlertCircle, RotateCcw,
  Layers, Gauge, Clock, TrendingUp, Cpu, Zap, BarChart3, Grid3x3,
  Wand2, CheckCircle2, ArrowRight
} from 'lucide-react'
import {
  LineChart, Line, BarChart, Bar, ScatterChart, Scatter, ReferenceLine,
  XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell
} from 'recharts'
import { useTheme } from '../ThemeContext'

const API = import.meta.env.VITE_API_URL
const PERCENT_KEYS = new Set(['accuracy', 'f1_score', 'precision', 'recall', 'r2_score'])

const HYPERS = [
  { key: 'epochs',        label: 'Epochs',        min: 5,      max: 200,  step: 5,      hint: 'Passes over the training data' },
  { key: 'learning_rate', label: 'Learning Rate', min: 0.0001, max: 0.01, step: 0.0001, hint: 'Step size for weight updates'   },
  { key: 'dropout',       label: 'Dropout',       min: 0,      max: 0.6,  step: 0.05,   hint: 'Regularization to curb overfit' },
  { key: 'batch_size',    label: 'Batch Size',    min: 8,      max: 256,  step: 8,      hint: 'Samples per gradient step'      },
]
const DEFAULT_CFG = { hidden_layers: [64, 32], epochs: 50, learning_rate: 0.001, dropout: 0.2, batch_size: 32 }
const LAYER_PRESETS = [
  { label: 'Shallow',  value: [32],          desc: '1 hidden layer'  },
  { label: 'Balanced', value: [64, 32],      desc: '2 hidden layers' },
  { label: 'Deep',     value: [128, 64, 32], desc: '3 hidden layers' },
]

const DeepLearningView = ({ data }) => {
  const { isDark } = useTheme()
  const [targetColumn,  setTargetColumn]  = useState('')
  const [config,        setConfig]        = useState(DEFAULT_CFG)
  const [suggestion,    setSuggestion]    = useState(null)
  const [recs,          setRecs]          = useState([])
  const [loadingSug,    setLoadingSug]    = useState(false)
  const [training,      setTraining]      = useState(false)
  const [trainStep,     setTrainStep]     = useState('')
  const [result,        setResult]        = useState(null)
  const [error,         setError]         = useState(null)

  const columns = data?.schema?.map(c => c.name) || []
  const setCfg = (key, val) => setConfig(prev => ({ ...prev, [key]: val }))

  // Fetch recommended target columns once when the view mounts
  useEffect(() => {
    axios.get(`${API}/deep/recommend-targets`)
      .then(res => setRecs(res.data.suggestions || []))
      .catch(() => setRecs([]))
  }, [])

  const handleSuggest = async (col = targetColumn) => {
    if (!col) return
    setLoadingSug(true); setSuggestion(null); setError(null)
    try {
      const res = await axios.post(`${API}/deep/suggest`, { target_column: col })
      setSuggestion(res.data)
      if (res.data.config) setConfig(prev => ({ ...prev, ...res.data.config }))
    } catch (e) {
      setError(e.response?.data?.detail || 'Could not analyse the target column.')
    } finally { setLoadingSug(false) }
  }

  const pickTarget = (col) => { setTargetColumn(col); setSuggestion(null); handleSuggest(col) }

  const handleTrain = async () => {
    if (!targetColumn) { setError('Select a target column to train a neural network.'); return }
    setTraining(true); setResult(null); setError(null)
    const steps = ['Encoding features…', 'Building network…', 'Initialising weights…', 'Training epochs…', 'Evaluating on hold-out set…']
    let i = 0; setTrainStep(steps[0])
    const iv = setInterval(() => { i++; if (i < steps.length) setTrainStep(steps[i]) }, 1000)
    try {
      const res = await axios.post(`${API}/deep/train`, { target_column: targetColumn, config })
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Training failed. Try a different target column or configuration.')
    } finally { clearInterval(iv); setTraining(false) }
  }

  if (training) return <TrainingScreen config={config} columns={columns} step={trainStep} />
  if (result)   return <ResultsView result={result} onReset={() => setResult(null)} isDark={isDark} />

  const card = `rounded-xl p-5 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl p-6"
        style={{ background: 'var(--df-card)', border: '1px solid var(--df-border)' }}>
        <div className="absolute inset-0 opacity-10"
          style={{ background: 'radial-gradient(circle at 50% 0%, #14b8a6, transparent 60%)' }} />
        <div className="relative flex items-center gap-4">
          <div className="p-3 bg-teal-500/10 border border-teal-500/20 rounded-xl shrink-0">
            <Network size={22} className="text-teal-400" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-black" style={{ color: 'var(--df-t1)' }}>Deep Learning Studio</h3>
            <p className="text-xs mt-0.5" style={{ color: 'var(--df-t3)' }}>
              Train a neural network, see what drives its predictions, and run live what-if scenarios
            </p>
          </div>
        </div>
      </div>

      {/* Recommended targets */}
      {recs.length > 0 && (
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <label className="flex items-center gap-2 font-semibold text-sm mb-3" style={{ color: 'var(--df-t1)' }}>
            <Wand2 size={14} className="text-teal-400" /> Recommended Targets
            <span className="text-xs font-normal" style={{ color: 'var(--df-t3)' }}>— what's worth predicting in this dataset</span>
          </label>
          <div className="flex flex-wrap gap-2.5">
            {recs.map(r => {
              const active = targetColumn === r.column
              return (
                <button key={r.column} onClick={() => pickTarget(r.column)}
                  className="group flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl border text-left transition-all"
                  style={{
                    background: active ? (isDark ? 'rgba(20,184,166,0.12)' : 'rgba(20,184,166,0.06)') : 'var(--df-input-bg)',
                    borderColor: active ? 'rgba(20,184,166,0.5)' : 'var(--df-border)',
                  }}>
                  <div>
                    <div className="flex items-center gap-1.5">
                      <span className="font-semibold text-sm" style={{ color: 'var(--df-t1)' }}>{r.column}</span>
                      {r.recommended && (
                        <span className="text-[8px] font-black uppercase tracking-wider text-teal-400 bg-teal-500/10 border border-teal-500/20 px-1.5 py-0.5 rounded-full">
                          Best
                        </span>
                      )}
                    </div>
                    <p className="text-[11px]" style={{ color: 'var(--df-t3)' }}>{r.reason}</p>
                  </div>
                  <ArrowRight size={14} className="opacity-0 group-hover:opacity-60 transition-opacity text-teal-400 shrink-0" />
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Target + AI suggest */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <label className="flex items-center gap-2 font-semibold text-sm mb-3" style={{ color: 'var(--df-t1)' }}>
            <Target size={14} className="text-teal-400" /> Target Column
          </label>
          <select value={targetColumn}
            onChange={e => { setTargetColumn(e.target.value); setSuggestion(null); if (e.target.value) handleSuggest(e.target.value) }}
            className="w-full rounded-lg px-3 py-2.5 text-sm outline-none border"
            style={{ background: 'var(--df-input-bg)', borderColor: 'var(--df-input-border)', color: 'var(--df-t1)' }}>
            <option value="">— Select a column to predict —</option>
            {columns.map(col => <option key={col} value={col}>{col}</option>)}
          </select>
          <p className="text-xs mt-2" style={{ color: 'var(--df-t3)' }}>Neural networks need a target — pick what to predict</p>
        </div>

        <div className={card} style={{ background: 'var(--df-card)' }}>
          <label className="flex items-center gap-2 font-semibold text-sm mb-3" style={{ color: 'var(--df-t1)' }}>
            <Sparkles size={14} className="text-teal-400" /> Detected Setup
          </label>
          {loadingSug ? (
            <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--df-t3)' }}>
              <Loader2 size={14} className="animate-spin" /> Analysing…
            </div>
          ) : suggestion?.task ? (
            <div className="space-y-1.5">
              <p className="text-sm" style={{ color: 'var(--df-t2)' }}>
                <span className="font-semibold capitalize" style={{ color: 'var(--df-t1)' }}>{suggestion.task}</span> task ·{' '}
                <span className="text-teal-400 font-semibold">{suggestion.n_features} features</span>
                {suggestion.n_classes ? <> · {suggestion.n_classes} classes</> : null}
              </p>
              <p className="text-xs flex items-center gap-1.5" style={{ color: 'var(--df-t3)' }}>
                <Cpu size={11} /> Engine: <span className="font-semibold">{suggestion.engine}</span> · architecture auto-tuned below
              </p>
            </div>
          ) : (
            <p className="text-sm" style={{ color: 'var(--df-t3)' }}>Pick a target and the network self-configures for it.</p>
          )}
        </div>
      </div>

      {/* Architecture presets */}
      <div className={card} style={{ background: 'var(--df-card)' }}>
        <label className="flex items-center gap-2 font-semibold text-sm mb-3" style={{ color: 'var(--df-t1)' }}>
          <Layers size={14} className="text-teal-400" /> Network Depth
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {LAYER_PRESETS.map(preset => {
            const active = JSON.stringify(preset.value) === JSON.stringify(config.hidden_layers)
            return (
              <button key={preset.label} onClick={() => setCfg('hidden_layers', preset.value)}
                className="text-left p-4 rounded-xl border transition-all"
                style={{
                  background: active ? (isDark ? 'rgba(20,184,166,0.08)' : 'rgba(20,184,166,0.05)') : 'var(--df-input-bg)',
                  borderColor: active ? 'rgba(20,184,166,0.5)' : 'var(--df-border)',
                }}>
                <div className="flex items-center justify-between">
                  <span className="font-bold text-sm" style={{ color: 'var(--df-t1)' }}>{preset.label}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20">
                    {preset.value.join('→')}
                  </span>
                </div>
                <p className="text-xs mt-1" style={{ color: 'var(--df-t3)' }}>{preset.desc}</p>
              </button>
            )
          })}
        </div>
      </div>

      {/* Hyperparameters */}
      <div className={card} style={{ background: 'var(--df-card)' }}>
        <label className="flex items-center gap-2 font-semibold text-sm mb-5" style={{ color: 'var(--df-t1)' }}>
          <Gauge size={14} className="text-teal-400" /> Hyperparameters
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-6">
          {HYPERS.map(h => {
            const val = config[h.key]
            const pct = ((val - h.min) / (h.max - h.min)) * 100
            const isInt = h.key === 'batch_size' || h.key === 'epochs'
            const onChange = (e) => setCfg(h.key, isInt ? parseInt(e.target.value) : parseFloat(e.target.value))
            const displayVal = h.key === 'learning_rate' ? val.toFixed(4) : val

            return (
              <div key={h.key} className="group">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium" style={{ color: 'var(--df-t2)' }}>{h.label}</span>
                  <span className="text-sm font-bold font-mono px-2.5 py-1 rounded-lg"
                    style={{ color: 'var(--df-primary)', background: 'rgba(20,184,166,0.08)' }}>
                    {displayVal}
                  </span>
                </div>
                <div className="relative h-8 flex items-center">
                  <div className="absolute inset-x-0 h-2 rounded-full overflow-hidden"
                    style={{ background: isDark ? 'rgba(30,41,59,0.8)' : '#e2e8f0' }}>
                    <div className="h-full rounded-full transition-all duration-150 ease-out"
                      style={{ width: `${pct}%`, background: 'linear-gradient(90deg, #0d9488, #14b8a6, #2dd4bf)' }} />
                  </div>
                  <input type="range" min={h.min} max={h.max} step={h.step} value={val}
                    onChange={onChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    style={{ margin: 0 }} />
                  <div className="absolute z-0 pointer-events-none transition-all duration-150 ease-out"
                    style={{ left: `calc(${pct}% - 8px)`, top: '50%', transform: 'translateY(-50%)' }}>
                    <div className="w-4 h-4 rounded-full shadow-md border-2 border-white"
                      style={{
                        background: 'linear-gradient(135deg, #0d9488, #14b8a6)',
                        boxShadow: '0 2px 8px rgba(20,184,166,0.35), 0 0 0 4px rgba(20,184,166,0.08)',
                      }} />
                  </div>
                </div>
                <div className="flex justify-between mt-1.5">
                  <span className="text-[10px] font-mono" style={{ color: 'var(--df-t4)' }}>{h.min}</span>
                  <span className="text-[11px]" style={{ color: 'var(--df-t3)' }}>{h.hint}</span>
                  <span className="text-[10px] font-mono" style={{ color: 'var(--df-t4)' }}>{h.max}</span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
          <AlertCircle size={16} className="shrink-0" /> {error}
        </div>
      )}

      <div className="flex items-center justify-between pt-2 flex-wrap gap-4" style={{ borderTop: '1px solid var(--df-border)' }}>
        <p className="text-sm" style={{ color: 'var(--df-t3)' }}>
          {targetColumn
            ? <>Network ready · target: <span className="text-teal-400">{targetColumn}</span></>
            : 'Select a target column to continue'}
        </p>
        <button onClick={handleTrain} disabled={!targetColumn}
          className="flex items-center gap-2.5 px-7 py-3 rounded-xl font-bold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed disabled:scale-100"
          style={{ background: 'linear-gradient(135deg, #0d9488, #2dd4bf)' }}>
          <Play size={15} fill="currentColor" /> Train Network
        </button>
      </div>
    </div>
  )
}

// ── Training screen ────────────────────────────────────────────────────────────
const TrainingScreen = ({ config, columns, step }) => (
  <div className="max-w-lg mx-auto py-16 text-center space-y-8">
    <div className="relative w-20 h-20 mx-auto">
      <div className="absolute inset-0 rounded-full border-2" style={{ borderColor: 'var(--df-border)' }} />
      <div className="absolute inset-0 rounded-full border-2 border-teal-400 border-t-transparent animate-spin" />
      <Network size={24} className="absolute inset-0 m-auto text-teal-400" />
    </div>
    <div>
      <h3 className="text-xl font-bold" style={{ color: 'var(--df-t1)' }}>Training Neural Network</h3>
      <p className="text-teal-400 text-sm mt-2 font-medium">{step}</p>
    </div>
    <div className="rounded-2xl p-4 text-left space-y-2"
      style={{ background: 'var(--df-card)', border: '1px solid var(--df-border)' }}>
      {[['Architecture', columns.length ? `${config.hidden_layers.join(' → ')} units` : '—'],
        ['Epochs', config.epochs], ['Learning rate', config.learning_rate]].map(([k, v]) => (
        <div key={k} className="flex items-center justify-between text-xs" style={{ color: 'var(--df-t3)' }}>
          <span>{k}</span><span className="font-semibold" style={{ color: 'var(--df-t2)' }}>{v}</span>
        </div>
      ))}
    </div>
  </div>
)

// ── Results ─────────────────────────────────────────────────────────────────
const ResultsView = ({ result, onReset, isDark }) => {
  const {
    metrics = {}, history = [], task, engine, n_params, training_time,
    architecture = [], primary_metric, metric_label,
    feature_importance = [], evaluation = {}, feature_spec = [], sample_note,
    time_note,
  } = result
  const hasValLoss = history.some(h => h.val_loss != null)

  const grid  = isDark ? '#1e293b' : '#e2e8f0'
  const axis  = isDark ? '#64748b' : '#94a3b8'
  const tipBg = isDark ? '#0f172a' : '#ffffff'
  const fmt = (v) => typeof v !== 'number' ? '—'
    : PERCENT_KEYS.has(primary_metric) ? `${(v * 100).toFixed(1)}%` : v.toFixed(4)

  const card = `rounded-2xl p-6 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`
  const tip  = { background: tipBg, border: '1px solid var(--df-border)', borderRadius: 12, fontSize: 12, color: 'var(--df-t1)' }

  return (
    <div className="space-y-8">
      {/* Metrics below come from a sample, or a shortened run, when the dataset
          is large enough that a full one would not fit in a single request. */}
      {[sample_note, time_note].filter(Boolean).map((note) => (
        <div key={note} className="flex items-start gap-3 px-4 py-3 rounded-xl bg-amber-500/10 border border-amber-500/30">
          <AlertCircle size={16} className="text-amber-500 mt-0.5 shrink-0" />
          <p className="text-amber-500 text-sm">{note}</p>
        </div>
      ))}

      {/* Banner */}
      <div className="relative overflow-hidden rounded-2xl p-6"
        style={{
          background: isDark
            ? 'linear-gradient(135deg, rgba(20,184,166,0.1) 0%, var(--df-card) 50%, rgba(45,212,191,0.1) 100%)'
            : 'linear-gradient(135deg, #f5f3ff 0%, #ffffff 50%, #eef2ff 100%)',
          border: '1px solid rgba(20,184,166,0.25)',
        }}>
        <div className="flex items-center gap-5 flex-wrap">
          <div className="p-4 bg-teal-500/15 border border-teal-500/25 rounded-2xl shrink-0">
            <Network size={28} className="text-teal-400" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs uppercase tracking-widest font-semibold" style={{ color: 'var(--df-t3)' }}>Neural Network Trained</p>
            <h3 className="text-2xl font-black mt-0.5" style={{ color: 'var(--df-t1)' }}>
              {metric_label}: {fmt(metrics[primary_metric])}
            </h3>
            <p className="text-sm mt-0.5 flex items-center gap-3 flex-wrap" style={{ color: 'var(--df-t2)' }}>
              <span className="flex items-center gap-1"><Cpu size={12} /> {engine}</span>
              <span className="flex items-center gap-1"><Zap size={12} /> {n_params?.toLocaleString()} params</span>
              <span className="flex items-center gap-1"><Clock size={12} /> {training_time}</span>
            </p>
          </div>
          <span className="hidden md:inline-block capitalize text-teal-400 font-bold text-sm bg-teal-500/10 px-3 py-1.5 rounded-lg border border-teal-500/20 shrink-0">
            {task}
          </span>
        </div>
      </div>

      {/* Loss + metric curves */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <ChartTitle icon={TrendingUp} color="text-teal-400" title="Training Loss" />
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={history} margin={{ top: 5, right: 8, left: -8, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false} />
              <XAxis dataKey="epoch" stroke={axis} fontSize={11} tickLine={false} />
              <YAxis stroke={axis} fontSize={11} tickLine={false} width={44} />
              <Tooltip contentStyle={tip} labelFormatter={v => `Epoch ${v}`} />
              {hasValLoss && <Legend wrapperStyle={{ fontSize: 12 }} />}
              <Line type="monotone" dataKey="train_loss" name="Train loss" stroke="#14b8a6" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
              {hasValLoss && <Line type="monotone" dataKey="val_loss" name="Validation loss" stroke="#f59e0b" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <ChartTitle icon={Gauge} color="text-emerald-400" title={`Validation ${task === 'regression' ? 'R²' : 'Accuracy'}`} />
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={history} margin={{ top: 5, right: 8, left: -8, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false} />
              <XAxis dataKey="epoch" stroke={axis} fontSize={11} tickLine={false} />
              <YAxis stroke={axis} fontSize={11} tickLine={false} width={44} domain={task === 'regression' ? ['auto', 'auto'] : [0, 1]} />
              <Tooltip contentStyle={tip} labelFormatter={v => `Epoch ${v}`} />
              <Line type="monotone" dataKey="val_metric" name={task === 'regression' ? 'Val R²' : 'Val accuracy'} stroke="#10b981" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature importance + evaluation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <ChartTitle icon={BarChart3} color="text-teal-400" title="Feature Importance"
            sub="How much each input drives predictions (permutation)" />
          <ResponsiveContainer width="100%" height={Math.max(180, feature_importance.length * 34)}>
            <BarChart data={feature_importance} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={grid} horizontal={false} />
              <XAxis type="number" stroke={axis} fontSize={11} tickLine={false} unit="%" />
              <YAxis type="category" dataKey="feature" stroke={axis} fontSize={11} tickLine={false} width={90} />
              <Tooltip contentStyle={tip} formatter={v => [`${v}%`, 'Importance']} cursor={{ fill: isDark ? 'rgba(20,184,166,0.08)' : 'rgba(20,184,166,0.05)' }} />
              <Bar dataKey="importance_pct" radius={[0, 4, 4, 0]} fill="#14b8a6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className={card} style={{ background: 'var(--df-card)' }}>
          {evaluation.kind === 'classification' ? (
            <ClassificationEval evaluation={evaluation} isDark={isDark} tip={tip} grid={grid} axis={axis} />
          ) : evaluation.kind === 'regression' ? (
            <RegressionEval evaluation={evaluation} tip={tip} grid={grid} axis={axis} />
          ) : null}
        </div>
      </div>

      {/* Metrics + architecture */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className={`${card} lg:col-span-2`} style={{ background: 'var(--df-card)' }}>
          <h4 className="font-bold text-sm mb-4" style={{ color: 'var(--df-t1)' }}>Test-Set Metrics</h4>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {Object.entries(metrics).map(([key, val]) => (
              <div key={key} className="rounded-xl p-4" style={{ background: 'var(--df-input-bg)', border: '1px solid var(--df-border)' }}>
                <p className="text-xs capitalize mb-1" style={{ color: 'var(--df-t3)' }}>{key.replace(/_/g, ' ')}</p>
                <p className="text-xl font-black" style={{ color: 'var(--df-t1)' }}>
                  {PERCENT_KEYS.has(key) && typeof val === 'number' ? `${(val * 100).toFixed(1)}%`
                    : typeof val === 'number' ? val.toLocaleString(undefined, { maximumFractionDigits: 4 }) : val}
                </p>
              </div>
            ))}
          </div>
        </div>
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <h4 className="font-bold text-sm mb-4 flex items-center gap-2" style={{ color: 'var(--df-t1)' }}>
            <Layers size={14} className="text-teal-400" /> Architecture
          </h4>
          <div className="space-y-2">
            {architecture.map((layer, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="w-6 h-6 rounded-lg flex items-center justify-center text-[10px] font-black shrink-0"
                  style={{ background: 'rgba(20,184,166,0.12)', border: '1px solid rgba(20,184,166,0.25)', color: '#2dd4bf' }}>{i + 1}</div>
                <span className="text-xs" style={{ color: 'var(--df-t2)' }}>{layer}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Prediction playground */}
      <PredictionPlayground featureSpec={feature_spec} target={result.target_column} isDark={isDark} />

      <button onClick={onReset}
        className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-colors"
        style={{ background: isDark ? 'rgba(30,41,59,0.8)' : '#f1f5f9', border: '1px solid var(--df-border)', color: 'var(--df-t2)' }}>
        <RotateCcw size={14} /> Reconfigure & Retrain
      </button>
    </div>
  )
}

const ChartTitle = ({ icon: Icon, color, title, sub }) => (
  <div className="mb-4">
    <div className="flex items-center gap-2">
      <Icon size={15} className={color} />
      <h4 className="font-bold text-sm" style={{ color: 'var(--df-t1)' }}>{title}</h4>
    </div>
    {sub && <p className="text-[11px] mt-0.5 ml-6" style={{ color: 'var(--df-t3)' }}>{sub}</p>}
  </div>
)

// ── Classification evaluation: confusion matrix (+ ROC for binary) ─────────────
const ClassificationEval = ({ evaluation, isDark, tip, grid, axis }) => {
  const { labels = [], confusion_matrix: cm = [], roc } = evaluation
  const max = Math.max(1, ...cm.flat())
  return (
    <>
      <ChartTitle icon={Grid3x3} color="text-amber-400" title="Confusion Matrix" sub="Rows = actual · Columns = predicted" />
      <div className="overflow-x-auto">
        <table className="text-xs mx-auto" style={{ color: 'var(--df-t2)' }}>
          <thead>
            <tr>
              <th></th>
              {labels.map(l => <th key={l} className="px-2 py-1 font-semibold" style={{ color: 'var(--df-t3)' }}>{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {cm.map((row, i) => (
              <tr key={i}>
                <td className="px-2 py-1 font-semibold text-right" style={{ color: 'var(--df-t3)' }}>{labels[i]}</td>
                {row.map((v, j) => {
                  const t = v / max
                  const correct = i === j
                  return (
                    <td key={j} className="w-14 h-14 text-center font-bold rounded-lg" style={{
                      background: correct
                        ? `rgba(16,185,129,${0.12 + 0.5 * t})`
                        : `rgba(239,68,68,${0.08 + 0.45 * t})`,
                      color: t > 0.5 ? '#fff' : 'var(--df-t1)',
                      border: '2px solid var(--df-card)',
                    }}>{v}</td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {roc && (
        <div className="mt-5 pt-4" style={{ borderTop: '1px solid var(--df-border)' }}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold" style={{ color: 'var(--df-t2)' }}>ROC Curve</span>
            <span className="text-xs font-bold text-teal-400">AUC {roc.auc}</span>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={roc.points} margin={{ top: 5, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={grid} />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke={axis} fontSize={10} tickLine={false} />
              <YAxis dataKey="tpr" type="number" domain={[0, 1]} stroke={axis} fontSize={10} tickLine={false} />
              <Tooltip contentStyle={tip} formatter={(v, n) => [v, n.toUpperCase()]} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={axis} strokeDasharray="4 4" />
              <Line type="monotone" dataKey="tpr" stroke="#14b8a6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  )
}

// ── Regression evaluation: predicted vs actual ────────────────────────────────
const RegressionEval = ({ evaluation, tip, grid, axis }) => {
  const pts = evaluation.points || []
  const vals = pts.flatMap(p => [p.actual, p.predicted])
  const lo = Math.min(...vals, 0), hi = Math.max(...vals, 1)
  return (
    <>
      <ChartTitle icon={TrendingUp} color="text-emerald-400" title="Predicted vs Actual"
        sub="Points on the dashed line are perfect predictions" />
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={grid} />
          <XAxis type="number" dataKey="actual" name="Actual" domain={[lo, hi]} stroke={axis} fontSize={10} tickLine={false} />
          <YAxis type="number" dataKey="predicted" name="Predicted" domain={[lo, hi]} stroke={axis} fontSize={10} tickLine={false} width={48} />
          <ZAxis range={[40, 40]} />
          <Tooltip contentStyle={tip} cursor={{ strokeDasharray: '3 3' }} />
          <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke={axis} strokeDasharray="4 4" />
          <Scatter data={pts} fill="#10b981" fillOpacity={0.6} />
        </ScatterChart>
      </ResponsiveContainer>
    </>
  )
}

// ── Prediction playground ─────────────────────────────────────────────────────
const PredictionPlayground = ({ featureSpec = [], target, isDark }) => {
  const [form, setForm] = useState(() =>
    Object.fromEntries(featureSpec.map(f => [f.name, f.default])))
  const [pred, setPred]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const run = async () => {
    setLoading(true); setError(null)
    try {
      const res = await axios.post(`${API}/deep/predict`, { inputs: form })
      setPred(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Prediction failed.')
    } finally { setLoading(false) }
  }

  const card = `rounded-2xl p-6 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`

  return (
    <div className={card} style={{ background: 'var(--df-card)' }}>
      <ChartTitle icon={Wand2} color="text-teal-400" title="Prediction Playground"
        sub={`Enter feature values and the network predicts ${target}`} />
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 mb-5">
        {featureSpec.map(f => (
          <div key={f.name}>
            <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--df-t2)' }}>{f.name}</label>
            {f.kind === 'categorical' ? (
              <select value={form[f.name]} onChange={e => setForm({ ...form, [f.name]: e.target.value })}
                className="w-full rounded-lg px-2.5 py-2 text-sm outline-none border"
                style={{ background: 'var(--df-input-bg)', borderColor: 'var(--df-input-border)', color: 'var(--df-t1)' }}>
                {f.categories.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            ) : (
              <input type="number" value={form[f.name]} step="any"
                onChange={e => setForm({ ...form, [f.name]: e.target.value === '' ? '' : parseFloat(e.target.value) })}
                className="w-full rounded-lg px-2.5 py-2 text-sm outline-none border"
                style={{ background: 'var(--df-input-bg)', borderColor: 'var(--df-input-border)', color: 'var(--df-t1)' }} />
            )}
          </div>
        ))}
      </div>

      <div className="flex items-center gap-4 flex-wrap">
        <button onClick={run} disabled={loading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
          style={{ background: 'linear-gradient(135deg, #0d9488, #2dd4bf)' }}>
          {loading ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />}
          {loading ? 'Predicting…' : 'Run Prediction'}
        </button>

        {error && <span className="text-red-400 text-sm flex items-center gap-1.5"><AlertCircle size={14} /> {error}</span>}

        {pred && !error && (
          <div className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/25">
            <CheckCircle2 size={16} className="text-emerald-400 shrink-0" />
            <div>
              <span className="text-xs" style={{ color: 'var(--df-t3)' }}>{target} =</span>{' '}
              <span className="font-black text-emerald-400">{pred.prediction}</span>
              {pred.confidence != null && (
                <span className="text-xs ml-2" style={{ color: 'var(--df-t3)' }}>({(pred.confidence * 100).toFixed(1)}% confident)</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Class probability distribution */}
      {pred?.distribution?.length > 0 && (
        <div className="mt-5 space-y-2">
          {pred.distribution.slice(0, 6).map(d => (
            <div key={d.class} className="flex items-center gap-3">
              <span className="text-xs font-mono w-20 truncate shrink-0" style={{ color: 'var(--df-t2)' }}>{d.class}</span>
              <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: 'var(--df-input-bg)' }}>
                <div className="h-full rounded-full" style={{ width: `${d.prob * 100}%`, background: 'linear-gradient(90deg, #0d9488, #2dd4bf)' }} />
              </div>
              <span className="text-xs font-mono w-12 text-right shrink-0" style={{ color: 'var(--df-t3)' }}>{(d.prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default DeepLearningView
