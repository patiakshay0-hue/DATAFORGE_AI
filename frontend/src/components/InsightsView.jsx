import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Lightbulb, Info, AlertTriangle, CheckCircle2,
  Sparkles, Loader2, TrendingUp, AlertCircle
} from 'lucide-react'
import { useTheme } from '../ThemeContext'

const typeConfig = {
  success: { icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', accent: 'bg-emerald-500' },
  warning: { icon: AlertTriangle,color: 'text-amber-400',   bg: 'bg-amber-500/10',   border: 'border-amber-500/20',  accent: 'bg-amber-500'   },
  insight: { icon: Lightbulb,    color: 'text-teal-400',    bg: 'bg-teal-500/10',    border: 'border-teal-500/20',   accent: 'bg-teal-500'    },
  info:    { icon: Info,         color: 'text-teal-400',  bg: 'bg-teal-500/10',  border: 'border-teal-500/20', accent: 'bg-teal-500'  },
}

const InsightsView = ({ data, onNavigate }) => {
  const { isDark } = useTheme()
  const [insights, setInsights] = useState([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(null)

  useEffect(() => {
    const fetch_ = async () => {
      setLoading(true); setError(null)
      try {
        const response = await axios.get(`${import.meta.env.VITE_API_URL}/insights`)
        setInsights(response.data)
      } catch (err) {
        setError('Could not load insights. Make sure the backend is running.')
      } finally {
        setLoading(false)
      }
    }
    fetch_()
  }, [])

  if (loading) return (
    <div className="flex flex-col items-center justify-center py-32 gap-6">
      <div className="relative">
        <div className="w-16 h-16 rounded-full" style={{ border: '2px solid var(--df-border)' }} />
        <div className="absolute inset-0 w-16 h-16 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: 'var(--df-primary)', borderTopColor: 'transparent' }} />
      </div>
      <div className="text-center">
        <p className="font-semibold text-lg" style={{ color: 'var(--df-t1)' }}>Generating AI Insights</p>
        <p className="text-sm mt-1" style={{ color: 'var(--df-t3)' }}>Analyzing patterns, correlations & anomalies…</p>
      </div>
    </div>
  )

  if (error) return (
    <div className="flex flex-col items-center justify-center py-32 gap-4">
      <div className="p-4 bg-red-500/10 rounded-2xl border border-red-500/20">
        <AlertCircle className="text-red-400 w-10 h-10" />
      </div>
      <div className="text-center">
        <p className="text-red-400 font-semibold text-lg">Failed to Load Insights</p>
        <p className="text-sm mt-2 max-w-md" style={{ color: 'var(--df-t3)' }}>{error}</p>
      </div>
    </div>
  )

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Header banner */}
      <div className="relative overflow-hidden rounded-2xl p-8"
        style={{
          background: isDark
            ? 'linear-gradient(135deg, #0d1523 0%, rgba(20,184,166,0.08) 50%, #0d1523 100%)'
            : 'linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 50%, #f0fdfa 100%)',
          border: '1px solid rgba(20,184,166,0.25)',
        }}>
        <div className="relative flex items-center gap-5">
          <div className="p-4 backdrop-blur rounded-2xl" style={{ background: 'rgba(20,184,166,0.2)', border: '1px solid rgba(20,184,166,0.3)' }}>
            <Sparkles className="w-8 h-8" style={{ color: 'var(--df-primary)' }} />
          </div>
          <div>
            <h3 className="text-2xl font-bold" style={{ color: 'var(--df-t1)' }}>AI-Powered Insights</h3>
            <p className="mt-1 text-sm" style={{ color: 'var(--df-t2)' }}>
              {insights.length} automated findings detected from your dataset
            </p>
          </div>
        </div>
      </div>

      {/* Insights Grid */}
      {insights.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {insights.map((insight, idx) => {
            const cfg      = typeConfig[insight.type] || typeConfig.info
            const IconComp = cfg.icon
            return (
              <div key={idx}
                className={`group relative overflow-hidden border ${cfg.border} rounded-2xl p-6 transition-all duration-200 ${
                  isDark ? 'hover:brightness-110' : 'hover:shadow-md'
                }`}
                style={{ background: 'var(--df-card)' }}>
                <div className={`absolute top-0 left-0 right-0 h-0.5 ${cfg.accent} opacity-60`} />
                <div className="flex gap-4">
                  <div className={`shrink-0 p-3 ${cfg.bg} rounded-xl`}>
                    <IconComp className={`${cfg.color} w-5 h-5`} />
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                      <h4 className="font-bold" style={{ color: 'var(--df-t1)' }}>{insight.title}</h4>
                      {insight.type === 'insight' && <Sparkles size={13} style={{ color: 'var(--df-primary)' }} />}
                    </div>
                    <p className="text-sm leading-relaxed" style={{ color: 'var(--df-t2)' }}>{insight.text}</p>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-20" style={{ color: 'var(--df-t3)' }}>
          <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-30" />
          <p className="font-medium">No insights generated yet.</p>
          <p className="text-sm mt-1">Upload a richer dataset to unlock automated analysis.</p>
        </div>
      )}

      {/* LLM Upsell Banner */}
      <div className="relative overflow-hidden rounded-2xl p-8"
        style={{
          background: 'var(--df-card)',
          border: '1px solid var(--df-border)',
        }}>
        <div className="absolute right-0 top-0 bottom-0 w-64 opacity-10"
          style={{ background: 'radial-gradient(circle at 80% 50%, #14b8a6, transparent 70%)' }} />
        <div className="relative flex flex-col md:flex-row items-center gap-8">
          <div className="flex-1 space-y-3">
            <span className="text-xs font-bold uppercase tracking-widest px-3 py-1 rounded-full"
              style={{ color: 'var(--df-primary)', background: 'rgba(20,184,166,0.1)', border: '1px solid rgba(20,184,166,0.2)' }}>
              Pro Feature
            </span>
            <h3 className="text-xl font-bold" style={{ color: 'var(--df-t1)' }}>Chat with your Data</h3>
            <p className="text-sm leading-relaxed max-w-md" style={{ color: 'var(--df-t2)' }}>
              Ask natural language questions like <em>"Which region has the highest sales?"</em> and get instant, cited answers powered by an LLM.
            </p>
            <button
              onClick={() => onNavigate?.('chat')}
              className="mt-2 px-6 py-2.5 text-white rounded-xl font-semibold text-sm"
              style={{ background: 'var(--df-primary)' }}
            >
              Unlock LLM Engine →
            </button>
          </div>
          <div className="shrink-0 w-32 h-32 rounded-full flex items-center justify-center" style={{ background: 'rgba(20,184,166,0.1)', border: '1px solid rgba(20,184,166,0.2)' }}>
            <Sparkles size={48} style={{ color: 'var(--df-primary)' }} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default InsightsView
