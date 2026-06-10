import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts'
import { BarChart3, PieChart as PieIcon, Table } from 'lucide-react'
import { useTheme } from '../ThemeContext'

const COLORS = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1']

const EDAView = ({ data }) => {
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

export default EDAView
