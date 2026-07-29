import React, { useState, useMemo } from 'react'
import axios from 'axios'
import {
  Brain, Play, Trophy, CheckCircle2, Loader2, Target, Sparkles,
  AlertCircle, RotateCcw, Check, Clock
} from 'lucide-react'
import { useTheme } from '../ThemeContext'

const ALL_MODELS = [
  { name: 'Linear Regression',   category: 'Regression',    desc: 'Predicts continuous output via a linear relationship',           badge: 'Regression only'    },
  { name: 'Logistic Regression', category: 'Classification', desc: 'Probabilistic binary / multiclass classifier',                  badge: 'Classification only' },
  { name: 'Decision Tree',       category: 'Both',           desc: 'Interpretable rule-based tree — handles non-linear patterns',   badge: 'Reg + Class'        },
  { name: 'Random Forest',       category: 'Both',           desc: 'Ensemble of trees — top performer on tabular data',             badge: 'Reg + Class'        },
  { name: 'Naive Bayes',         category: 'Classification', desc: 'Fast probabilistic Bayesian classifier',                        badge: 'Classification only' },
  { name: 'SVM',                 category: 'Both',           desc: 'Finds optimal hyperplane for complex decision boundaries',      badge: 'Reg + Class'        },
  { name: 'KNN',                 category: 'Both',           desc: 'Classifies by proximity to K nearest neighbors',               badge: 'Reg + Class'        },
  { name: 'K-Means',             category: 'Clustering',     desc: 'Unsupervised grouping of similar data points (no target needed)',badge: 'Clustering'        },
  { name: 'XGBoost',             category: 'Both',           desc: 'Gradient boosting — gold standard for structured tabular data',badge: 'Reg + Class'        },
]

const CATEGORIES = ['All', 'Regression', 'Classification', 'Both', 'Clustering']

const CAT_STYLE = {
  'Regression':    { text: 'text-sky-400',     bg: 'bg-sky-500/10',     border: 'border-sky-500/20'   },
  'Classification':{ text: 'text-violet-400',  bg: 'bg-violet-500/10',  border: 'border-violet-500/20' },
  'Both':          { text: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20'},
  'Clustering':    { text: 'text-amber-400',   bg: 'bg-amber-500/10',   border: 'border-amber-500/20'  },
}

const MODEL_COLORS = ['#0ea5e9','#8b5cf6','#10b981','#f59e0b','#ec4899','#6366f1','#14b8a6','#f97316']
const PERCENT_KEYS = new Set(['accuracy','f1_score','precision','recall','silhouette_score','r2_score'])

const PercentBar = ({ value, color, isDark }) => (
  <div className="flex items-center gap-2">
    <div className="flex-1 h-1.5 rounded-full overflow-hidden"
      style={{ background: isDark ? '#1e293b' : '#e2e8f0' }}>
      <div className="h-full rounded-full"
        style={{ width: `${Math.max(0, Math.min(100, value * 100)).toFixed(1)}%`, background: color }} />
    </div>
    <span className="font-bold text-xs w-12 text-right" style={{ color: 'var(--df-t1)' }}>
      {(value * 100).toFixed(1)}%
    </span>
  </div>
)

// ── Setup Screen ──────────────────────────────────────────────────────────────
const MLView = ({ data }) => {
  const { isDark } = useTheme()
  const [selectedModels,    setSelectedModels]    = useState(new Set(['Random Forest', 'XGBoost', 'Decision Tree', 'Logistic Regression']))
  const [targetColumn,      setTargetColumn]      = useState('')
  const [activeCategory,    setActiveCategory]    = useState('All')
  const [training,          setTraining]          = useState(false)
  const [trainingStep,      setTrainingStep]      = useState('')
  const [results,           setResults]           = useState(null)
  const [suggestion,        setSuggestion]        = useState(null)
  const [loadingSuggestion, setLoadingSuggestion] = useState(false)
  const [error,             setError]             = useState(null)

  const columns = data?.schema?.map(c => c.name) || []

  const filteredModels = useMemo(() =>
    activeCategory === 'All' ? ALL_MODELS : ALL_MODELS.filter(m => m.category === activeCategory),
    [activeCategory])

  const toggleModel = (name) => setSelectedModels(prev => {
    const next = new Set(prev)
    next.has(name) ? next.delete(name) : next.add(name)
    return next
  })

  const handleSuggest = async () => {
    setLoadingSuggestion(true); setSuggestion(null)
    try {
      const res = await axios.post(`${import.meta.env.VITE_API_URL}/suggest`, { target_column: targetColumn || null })
      setSuggestion(res.data); setSelectedModels(new Set(res.data.suggested_models))
    } catch {
      if (!targetColumn) {
        const s = { problem_type: 'clustering', suggested_models: ['K-Means'] }
        setSuggestion(s); setSelectedModels(new Set(s.suggested_models))
      } else {
        const col = data?.schema?.find(c => c.name === targetColumn)
        const isNum = col?.type === 'numeric'
        const s = isNum
          ? { problem_type: 'regression',     suggested_models: ['Random Forest','XGBoost','Linear Regression','Decision Tree','KNN','SVM'] }
          : { problem_type: 'classification', suggested_models: ['Random Forest','XGBoost','Logistic Regression','Decision Tree','Naive Bayes','SVM'] }
        setSuggestion(s); setSelectedModels(new Set(s.suggested_models))
      }
    } finally { setLoadingSuggestion(false) }
  }

  const handleTrain = async () => {
    if (selectedModels.size === 0) return
    setTraining(true); setResults(null); setError(null)
    const steps = ['Preparing dataset…','Engineering features…',`Training ${selectedModels.size} model${selectedModels.size > 1 ? 's' : ''}…`,'Running cross-validation…','Ranking by performance…']
    let i = 0; setTrainingStep(steps[0])
    const iv = setInterval(() => { i++; if (i < steps.length) setTrainingStep(steps[i]) }, 900)
    try {
      const res = await axios.post(`${import.meta.env.VITE_API_URL}/train`, {
        models: [...selectedModels], target_column: targetColumn || null,
      })
      setResults(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Training failed. Check your target column and try again.')
    } finally { clearInterval(iv); setTraining(false) }
  }

  // ── Training screen ──────────────────────────────────────────────────────────
  if (training) return (
    <div className="max-w-lg mx-auto py-16 text-center space-y-8">
      <div className="relative w-20 h-20 mx-auto">
        <div className="absolute inset-0 rounded-full border-2" style={{ borderColor: 'var(--df-border)' }} />
        <div className="absolute inset-0 rounded-full border-2 border-sky-400 border-t-transparent animate-spin" />
        <Brain size={24} className="absolute inset-0 m-auto text-sky-400" />
      </div>
      <div>
        <h3 className="text-xl font-bold" style={{ color: 'var(--df-t1)' }}>Training Models</h3>
        <p className="text-sky-400 text-sm mt-2 font-medium">{trainingStep}</p>
      </div>
      <div className="space-y-2 text-left rounded-2xl p-4"
        style={{ background: 'var(--df-card)', border: '1px solid var(--df-border)' }}>
        {[...selectedModels].map(m => (
          <div key={m} className="flex items-center gap-3 px-3 py-2 rounded-lg"
            style={{ background: isDark ? 'rgba(30,41,59,0.5)' : '#f1f5f9' }}>
            <Loader2 size={13} className="animate-spin text-sky-400 shrink-0" />
            <span className="text-sm" style={{ color: 'var(--df-t2)' }}>{m}</span>
          </div>
        ))}
      </div>
    </div>
  )

  // ── Results screen ───────────────────────────────────────────────────────────
  if (results) return (
    <ResultsView results={results} onRetrain={() => { setResults(null); setSuggestion(null) }} />
  )

  // ── Setup screen ─────────────────────────────────────────────────────────────
  const card = `rounded-xl p-5 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl p-6"
        style={{ background: 'var(--df-card)', border: '1px solid var(--df-border)' }}>
        <div className="absolute inset-0 opacity-10"
          style={{ background: 'radial-gradient(circle at 50% 0%, #0ea5e9, transparent 60%)' }} />
        <div className="relative flex items-center gap-4">
          <div className="p-3 bg-sky-500/10 border border-sky-500/20 rounded-xl shrink-0">
            <Brain size={22} className="text-sky-400" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-black" style={{ color: 'var(--df-t1)' }}>Automated ML Engine</h3>
            <p className="text-xs mt-0.5" style={{ color: 'var(--df-t3)' }}>
              Select models, set a target column, and let the engine rank the best performer
            </p>
          </div>
          <div className="hidden sm:flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-emerald-400 shrink-0">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            {selectedModels.size} selected
          </div>
        </div>
      </div>

      {/* Target + AI suggest */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <label className="flex items-center gap-2 font-semibold text-sm mb-3"
            style={{ color: 'var(--df-t1)' }}>
            <Target size={14} className="text-sky-400" /> Target Column
          </label>
          <select
            value={targetColumn}
            onChange={e => { setTargetColumn(e.target.value); setSuggestion(null) }}
            className="w-full rounded-lg px-3 py-2.5 text-sm outline-none border"
            style={{
              background: 'var(--df-input-bg)',
              borderColor: 'var(--df-input-border)',
              color: 'var(--df-t1)',
            }}
          >
            <option value="">— None (clustering only) —</option>
            {columns.map(col => <option key={col} value={col}>{col}</option>)}
          </select>
          <p className="text-xs mt-2" style={{ color: 'var(--df-t3)' }}>The column you want to predict</p>
        </div>

        <div className={card} style={{ background: 'var(--df-card)' }}>
          <label className="flex items-center gap-2 font-semibold text-sm mb-3"
            style={{ color: 'var(--df-t1)' }}>
            <Sparkles size={14} className="text-violet-400" /> AI Suggestion
          </label>
          {suggestion ? (
            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <CheckCircle2 size={14} className="text-emerald-400 mt-0.5 shrink-0" />
                <p className="text-sm" style={{ color: 'var(--df-t2)' }}>
                  <span className="font-semibold capitalize" style={{ color: 'var(--df-t1)' }}>{suggestion.problem_type}</span> task —{' '}
                  <span className="text-emerald-400 font-semibold">{suggestion.suggested_models.length} models</span> auto-selected
                </p>
              </div>
              <div className="flex flex-wrap gap-1 mt-2">
                {suggestion.suggested_models.map(m => (
                  <span key={m} className="text-[10px] font-semibold px-2 py-0.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-full">{m}</span>
                ))}
              </div>
              <button onClick={handleSuggest} className="text-xs text-violet-400 hover:text-violet-300 underline underline-offset-2 mt-1">
                Re-analyse
              </button>
            </div>
          ) : (
            <button onClick={handleSuggest} disabled={loadingSuggestion}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-semibold bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/20 text-violet-400 transition-colors disabled:opacity-50">
              {loadingSuggestion ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
              {loadingSuggestion ? 'Analysing…' : 'Suggest Best Models'}
            </button>
          )}
          <p className="text-xs mt-2" style={{ color: 'var(--df-t3)' }}>Auto-selects optimal models for your data type</p>
        </div>
      </div>

      {/* Category filter + bulk actions */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex gap-2 flex-wrap">
          {CATEGORIES.map(cat => (
            <button key={cat} onClick={() => setActiveCategory(cat)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors ${
                activeCategory === cat
                  ? 'bg-sky-500/20 border-sky-500/40 text-sky-400'
                  : isDark
                    ? 'border-slate-800 text-slate-500 hover:text-slate-300 hover:border-slate-700'
                    : 'border-slate-200 text-slate-500 hover:text-slate-700 hover:border-slate-300'
              }`}
              style={{ background: activeCategory === cat ? undefined : 'var(--df-card)' }}>
              {cat}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          {['Select All', 'Clear'].map((label, i) => (
            <button key={label}
              onClick={() => i === 0 ? setSelectedModels(new Set(filteredModels.map(m => m.name))) : setSelectedModels(new Set())}
              className="text-xs px-3 py-1.5 rounded-lg border transition-colors"
              style={{
                background: 'var(--df-card)',
                border: '1px solid var(--df-border)',
                color: 'var(--df-t2)',
              }}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Model grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
        {filteredModels.map(model => {
          const isSelected  = selectedModels.has(model.name)
          const isSuggested = suggestion?.suggested_models?.includes(model.name)
          const style       = CAT_STYLE[model.category]
          return (
            <button key={model.name} onClick={() => toggleModel(model.name)}
              className="relative text-left p-5 rounded-xl border transition-all duration-200 group hover:scale-[1.01]"
              style={{
                background: isSelected
                  ? isDark ? 'rgba(14,165,233,0.05)' : 'rgba(14,165,233,0.04)'
                  : 'var(--df-card)',
                border: isSelected
                  ? '1px solid rgba(14,165,233,0.4)'
                  : `1px solid var(--df-border)`,
                boxShadow: isSelected ? '0 4px 20px rgba(14,165,233,0.08)' : undefined,
              }}>
              <div className={`absolute top-4 right-4 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all shrink-0 ${
                isSelected ? 'bg-sky-500 border-sky-500' : isDark ? 'border-slate-600' : 'border-slate-300'
              }`}>
                {isSelected && <Check size={10} className="text-white" strokeWidth={3} />}
              </div>
              {isSuggested && (
                <span className="absolute top-3 left-3 inline-flex items-center gap-1 text-[9px] font-bold uppercase tracking-wider text-amber-400 bg-amber-500/10 border border-amber-500/20 px-1.5 py-0.5 rounded-full">
                  <Sparkles size={7} /> Suggested
                </span>
              )}
              <div className={isSuggested ? 'mt-6' : 'mt-0'}>
                <h4 className="font-bold text-sm pr-7 mb-1" style={{ color: 'var(--df-t1)' }}>{model.name}</h4>
                <p className="text-xs leading-relaxed mb-3 pr-2" style={{ color: 'var(--df-t3)' }}>{model.desc}</p>
                <span className={`inline-block text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${style.text} ${style.bg} ${style.border}`}>
                  {model.badge}
                </span>
              </div>
            </button>
          )
        })}
      </div>

      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
          <AlertCircle size={16} className="shrink-0" /> {error}
        </div>
      )}

      {/* Train CTA */}
      <div className="flex items-center justify-between pt-2 flex-wrap gap-4"
        style={{ borderTop: '1px solid var(--df-border)' }}>
        <p className="text-sm" style={{ color: 'var(--df-t3)' }}>
          {selectedModels.size === 0
            ? 'Select at least one model to continue'
            : `${selectedModels.size} model${selectedModels.size > 1 ? 's' : ''} ready to train`}
          {targetColumn && <span> · target: <span className="text-sky-400">{targetColumn}</span></span>}
        </p>
        <button onClick={handleTrain} disabled={selectedModels.size === 0}
          className="flex items-center gap-2.5 px-7 py-3 rounded-xl font-bold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed disabled:scale-100"
          style={{ background: 'linear-gradient(135deg, #0284c7, #6366f1)' }}>
          <Play size={15} fill="currentColor" />
          Train {selectedModels.size > 0 ? selectedModels.size : ''} Model{selectedModels.size !== 1 ? 's' : ''}
        </button>
      </div>
    </div>
  )
}

// ── Results Panel ─────────────────────────────────────────────────────────────
const ResultsView = ({ results, onRetrain }) => {
  const { isDark } = useTheme()
  const { results: modelResults = [], best_model, task } = results

  const successful = [...modelResults.filter(r => r.status === 'success')].sort((a, b) => {
    const pm = a.primary_metric
    if (pm === 'mse' || pm === 'mae') return (a.metrics[pm] || 0) - (b.metrics[pm] || 0)
    return (b.metrics[b.primary_metric] || 0) - (a.metrics[a.primary_metric] || 0)
  })
  const skipped  = modelResults.filter(r => r.status !== 'success')
  const bestResult = successful[0]
  const primaryKey   = task === 'regression' ? 'r2_score' : task === 'clustering' ? 'silhouette_score' : 'accuracy'
  const primaryLabel = task === 'regression' ? 'R² Score' : task === 'clustering' ? 'Silhouette' : 'Accuracy'
  const fmtPrimary   = (val) => typeof val !== 'number' ? '—'
    : PERCENT_KEYS.has(primaryKey) ? `${(val * 100).toFixed(1)}%` : val.toFixed(4)

  const card = `rounded-2xl p-6 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`

  return (
    <div className="space-y-8">
      {/* Best model banner */}
      {best_model && bestResult && (
        <div className="relative overflow-hidden rounded-2xl p-6"
          style={{
            background: isDark
              ? 'linear-gradient(135deg, rgba(14,165,233,0.08) 0%, var(--df-card) 50%, rgba(99,102,241,0.08) 100%)'
              : 'linear-gradient(135deg, #eff6ff 0%, #ffffff 50%, #f5f3ff 100%)',
            border: '1px solid rgba(14,165,233,0.25)',
          }}>
          <div className="flex items-center gap-5 flex-wrap">
            <div className="p-4 bg-amber-500/15 border border-amber-500/25 rounded-2xl shrink-0">
              <Trophy size={28} className="text-amber-400" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs uppercase tracking-widest font-semibold" style={{ color: 'var(--df-t3)' }}>Best Performing Model</p>
              <h3 className="text-2xl font-black mt-0.5 truncate" style={{ color: 'var(--df-t1)' }}>{best_model}</h3>
              <p className="text-sm mt-0.5" style={{ color: 'var(--df-t2)' }}>
                {primaryLabel}: <span className="font-bold" style={{ color: 'var(--df-t1)' }}>{fmtPrimary(bestResult.metrics?.[primaryKey])}</span>
                {' '}· Trained in {bestResult.training_time}
              </p>
            </div>
            <div className="hidden md:block text-right shrink-0">
              <p className="text-xs mb-1" style={{ color: 'var(--df-t3)' }}>Task Type</p>
              <span className="capitalize text-sky-400 font-bold text-sm bg-sky-500/10 px-3 py-1.5 rounded-lg border border-sky-500/20">
                {task}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Leaderboard */}
      {successful.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {successful.map((result, idx) => {
            const isBest = result.model === best_model
            const color  = MODEL_COLORS[idx % MODEL_COLORS.length]
            return (
              <div key={result.model}
                className={`rounded-2xl p-6 border transition-all ${
                  isBest
                    ? 'border-sky-500/40 shadow-lg shadow-sky-500/5'
                    : isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'
                }`}
                style={{ background: 'var(--df-card)' }}>
                <div className="flex items-center justify-between mb-5">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-xl flex items-center justify-center font-black text-sm"
                      style={{ background: color + '22', border: `1px solid ${color}44`, color }}>
                      {idx + 1}
                    </div>
                    <div>
                      <h4 className="font-bold text-sm" style={{ color: 'var(--df-t1)' }}>{result.model}</h4>
                      <p className="text-xs flex items-center gap-1" style={{ color: 'var(--df-t3)' }}>
                        <Clock size={10} /> {result.training_time}
                      </p>
                    </div>
                  </div>
                  {isBest && (
                    <span className="flex items-center gap-1 text-[10px] font-bold text-amber-400 bg-amber-500/10 px-2 py-1 rounded-lg border border-amber-500/20">
                      <Trophy size={10} /> BEST
                    </span>
                  )}
                </div>
                <div className="space-y-3">
                  {Object.entries(result.metrics || {}).map(([key, val]) => (
                    <div key={key}>
                      <p className="text-xs capitalize mb-1.5" style={{ color: 'var(--df-t3)' }}>
                        {key.replace(/_/g, ' ')}
                      </p>
                      {PERCENT_KEYS.has(key) && typeof val === 'number'
                        ? <PercentBar value={val} color={color} isDark={isDark} />
                        : <p className="font-bold text-sm" style={{ color: 'var(--df-t1)' }}>
                            {typeof val === 'number' ? val.toLocaleString(undefined, { maximumFractionDigits: 4 }) : val}
                          </p>
                      }
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Skipped */}
      {skipped.length > 0 && (
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <p className="font-semibold text-sm mb-4" style={{ color: 'var(--df-t2)' }}>Skipped / Not Applicable</p>
          <div className="space-y-3">
            {skipped.map(r => (
              <div key={r.model} className="flex items-start gap-3">
                <AlertCircle size={14} className={`shrink-0 mt-0.5 ${r.status === 'not_applicable' ? 'text-slate-600' : 'text-red-500/60'}`} />
                <div>
                  <span className="text-sm font-medium" style={{ color: 'var(--df-t2)' }}>{r.model}</span>
                  <p className="text-xs mt-0.5" style={{ color: 'var(--df-t3)' }}>{r.note || 'Not applicable for this task'}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <button onClick={onRetrain}
        className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-colors"
        style={{
          background: isDark ? 'rgba(30,41,59,0.8)' : '#f1f5f9',
          border: '1px solid var(--df-border)',
          color: 'var(--df-t2)',
        }}>
        <RotateCcw size={14} /> Select Different Models
      </button>
    </div>
  )
}

export default MLView
