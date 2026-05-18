import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Lightbulb, Info, AlertTriangle, CheckCircle2,
  Sparkles, Loader2, TrendingUp, AlertCircle
} from 'lucide-react'

const typeConfig = {
  success: {
    icon: CheckCircle2,
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/20',
    accent: 'bg-emerald-500'
  },
  warning: {
    icon: AlertTriangle,
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
    accent: 'bg-amber-500'
  },
  insight: {
    icon: Lightbulb,
    color: 'text-sky-400',
    bg: 'bg-sky-500/10',
    border: 'border-sky-500/20',
    accent: 'bg-sky-500'
  },
  info: {
    icon: Info,
    color: 'text-violet-400',
    bg: 'bg-violet-500/10',
    border: 'border-violet-500/20',
    accent: 'bg-violet-500'
  }
}

const InsightsView = ({ data }) => {
  const [insights, setInsights] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchInsights = async () => {
      setLoading(true)
      setError(null)
      try {
        const response = await axios.get(`${import.meta.env.VITE_API_URL}/insights`)
        setInsights(response.data)
      } catch (err) {
        setError('Could not load insights. Make sure the backend is running on port 8000.')
        console.error('Failed to fetch insights:', err)
      } finally {
        setLoading(false)
      }
    }
    fetchInsights()
  }, [])

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-6">
        <div className="relative">
          <div className="w-16 h-16 rounded-full border-2 border-sky-500/20" />
          <div className="absolute inset-0 w-16 h-16 rounded-full border-2 border-sky-500 border-t-transparent animate-spin" />
        </div>
        <div className="text-center">
          <p className="text-white font-semibold text-lg">Generating AI Insights</p>
          <p className="text-slate-500 text-sm mt-1">Analyzing patterns, correlations & anomalies…</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-4">
        <div className="p-4 bg-red-500/10 rounded-2xl border border-red-500/20">
          <AlertCircle className="text-red-400 w-10 h-10" />
        </div>
        <div className="text-center">
          <p className="text-red-400 font-semibold text-lg">Failed to Load Insights</p>
          <p className="text-slate-500 text-sm mt-2 max-w-md">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Header banner */}
      <div className="relative overflow-hidden bg-gradient-to-r from-slate-900 via-sky-950/40 to-slate-900 border border-sky-500/20 rounded-2xl p-8">
        <div className="absolute inset-0 opacity-30"
          style={{ backgroundImage: 'radial-gradient(circle at 20% 50%, #0ea5e9 0%, transparent 50%), radial-gradient(circle at 80% 50%, #8b5cf6 0%, transparent 50%)' }} />
        <div className="relative flex items-center gap-5">
          <div className="p-4 bg-sky-500/20 backdrop-blur rounded-2xl border border-sky-500/30">
            <Sparkles className="text-sky-400 w-8 h-8" />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-white">AI-Powered Insights</h3>
            <p className="text-slate-400 mt-1 text-sm">
              {insights.length} automated findings detected from your dataset
            </p>
          </div>
        </div>
      </div>

      {/* Insights Grid */}
      {insights.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {insights.map((insight, idx) => {
            const cfg = typeConfig[insight.type] || typeConfig.info
            const IconComp = cfg.icon
            return (
              <div key={idx}
                className={`group relative overflow-hidden bg-slate-900 border ${cfg.border} rounded-2xl p-6 hover:brightness-110 transition-all duration-300`}>
                {/* accent top bar */}
                <div className={`absolute top-0 left-0 right-0 h-0.5 ${cfg.accent} opacity-60`} />
                <div className="flex gap-4">
                  <div className={`flex-shrink-0 p-3 ${cfg.bg} rounded-xl`}>
                    <IconComp className={`${cfg.color} w-5 h-5`} />
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                      <h4 className="text-white font-bold">{insight.title}</h4>
                      {insight.type === 'insight' && (
                        <Sparkles size={13} className="text-sky-400" />
                      )}
                    </div>
                    <p className="text-slate-400 text-sm leading-relaxed">{insight.text}</p>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-20 text-slate-500">
          <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-30" />
          <p className="font-medium">No insights generated yet.</p>
          <p className="text-sm mt-1">Upload a richer dataset to unlock automated analysis.</p>
        </div>
      )}

      {/* LLM Upsell Banner */}
      <div className="relative overflow-hidden bg-slate-900 border border-slate-800 rounded-2xl p-8 mt-4">
        <div className="absolute right-0 top-0 bottom-0 w-64 opacity-10"
          style={{ background: 'radial-gradient(circle at 80% 50%, #8b5cf6, transparent 70%)' }} />
        <div className="relative flex flex-col md:flex-row items-center gap-8">
          <div className="flex-1 space-y-3">
            <span className="text-xs font-bold uppercase tracking-widest text-violet-400 bg-violet-500/10 px-3 py-1 rounded-full border border-violet-500/20">
              Pro Feature
            </span>
            <h3 className="text-xl font-bold text-white">Chat with your Data</h3>
            <p className="text-slate-400 text-sm leading-relaxed max-w-md">
              Ask natural language questions like <em>"Which region has the highest sales?"</em> and get instant, cited answers powered by an LLM.
            </p>
            <button className="mt-2 px-6 py-2.5 bg-violet-600 hover:bg-violet-500 text-white rounded-xl font-semibold text-sm transition-colors">
              Unlock LLM Engine →
            </button>
          </div>
          <div className="flex-shrink-0 w-32 h-32 bg-violet-500/10 border border-violet-500/20 rounded-full flex items-center justify-center">
            <Sparkles size={48} className="text-violet-400" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default InsightsView
