import React from 'react'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'
import { Activity, Zap, Globe, ArrowUpRight, ArrowDownRight, TrendingUp, Layers } from 'lucide-react'
import { useTheme } from '../ThemeContext'

const DbIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <ellipse cx="12" cy="5" rx="9" ry="3" />
    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
  </svg>
)

const COLORS = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1']

const DashboardView = ({ data }) => {
  const { isDark } = useTheme()

  if (!data) return (
    <div className="text-center p-10" style={{ color: 'var(--df-t3)' }}>No data available.</div>
  )

  const { charts = {}, rows = 0, columns = 0 } = data
  const chartEntries  = Object.entries(charts)
  const firstBar      = chartEntries.find(([, v]) => v.type === 'bar')
  const firstPie      = chartEntries.find(([, v]) => v.type === 'pie')
  const areaData      = firstBar ? firstBar[1].data : []
  const pieData       = firstPie ? firstPie[1].data.slice(0, 6) : []
  const pieChartName  = firstPie ? firstPie[0].replace('count_', '') : ''

  const gridColor  = isDark ? '#1e293b' : '#e2e8f0'
  const axisColor  = isDark ? '#475569' : '#94a3b8'
  const ttBg       = isDark ? '#1e293b' : '#ffffff'
  const ttBorder   = isDark ? '#334155' : '#e2e8f0'

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div style={{ background: ttBg, border: `1px solid ${ttBorder}` }}
        className="rounded-xl px-4 py-3 shadow-xl text-sm">
        <p style={{ color: 'var(--df-t2)' }} className="text-xs mb-1">{label}</p>
        <p style={{ color: 'var(--df-t1)' }} className="font-bold">{payload[0].value}</p>
      </div>
    )
  }

  const card = `rounded-2xl p-6 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`

  const kpis = [
    { label: 'Total Records',  value: Number(rows).toLocaleString(), icon: DbIcon,  colorClass: 'text-sky-400',     bgClass: 'bg-sky-500/10',     trend: '+12%',  up: true  },
    { label: 'Total Features', value: columns,                       icon: Layers,  colorClass: 'text-emerald-400', bgClass: 'bg-emerald-500/10', trend: 'Stable',up: true  },
    { label: 'Quality Score',  value: '94%',                         icon: Zap,     colorClass: 'text-amber-400',   bgClass: 'bg-amber-500/10',   trend: '+5%',   up: true  },
    { label: 'Data Coverage',  value: '98.2%',                       icon: Globe,   colorClass: 'text-violet-400',  bgClass: 'bg-violet-500/10',  trend: '-1%',   up: false },
  ]

  return (
    <div className="space-y-8">
      {/* KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        {kpis.map((kpi, idx) => (
          <div key={idx}
            className={`relative overflow-hidden rounded-2xl p-6 border group ${
              isDark ? 'border-slate-800 hover:border-slate-600' : 'border-slate-200 shadow-sm hover:shadow-md hover:border-slate-300'
            }`}
            style={{ background: 'var(--df-card)' }}>
            <div className="absolute top-0 right-0 w-24 h-24 opacity-5 rounded-full blur-2xl bg-blue-400 group-hover:opacity-10" />
            <div className="flex justify-between items-start mb-4">
              <div className={`p-2.5 rounded-xl ${kpi.bgClass}`}>
                <kpi.icon size={20} className={kpi.colorClass} />
              </div>
              <span className={`flex items-center gap-1 text-xs font-bold px-2 py-1 rounded-full ${
                kpi.up ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
              }`}>
                {kpi.up ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                {kpi.trend}
              </span>
            </div>
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--df-t3)' }}>{kpi.label}</p>
            <h4 className="text-3xl font-black mt-1" style={{ color: 'var(--df-t1)' }}>{kpi.value}</h4>
          </div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Area Chart */}
        <div className={`lg:col-span-2 ${card}`} style={{ background: 'var(--df-card)' }}>
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="font-bold text-lg" style={{ color: 'var(--df-t1)' }}>Distribution Trend</h3>
              <p className="text-xs mt-0.5" style={{ color: 'var(--df-t3)' }}>Primary numeric feature analysis</p>
            </div>
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-sky-500/10 border border-sky-500/20 rounded-lg">
              <TrendingUp size={13} className="text-sky-400" />
              <span className="text-sky-400 text-xs font-semibold">Live Data</span>
            </div>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={areaData} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                <defs>
                  <linearGradient id="grad1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#0ea5e9" stopOpacity={isDark ? 0.25 : 0.15} />
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                <XAxis dataKey="bin" stroke={axisColor} fontSize={10} tickLine={false} />
                <YAxis stroke={axisColor} fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="count" stroke="#0ea5e9" strokeWidth={2.5}
                  fillOpacity={1} fill="url(#grad1)" dot={false} activeDot={{ r: 5, fill: '#0ea5e9' }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Pie / Progress */}
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <h3 className="font-bold text-lg mb-1" style={{ color: 'var(--df-t1)' }}>Composition</h3>
          <p className="text-xs mb-5" style={{ color: 'var(--df-t3)' }}>
            {pieChartName ? `${pieChartName} breakdown` : 'Feature distribution'}
          </p>

          {pieData.length > 0 ? (
            <>
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="50%" innerRadius={50} outerRadius={70}
                      paddingAngle={4} dataKey="value">
                      {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-2 mt-4">
                {pieData.map((item, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                      <span className="truncate max-w-30" style={{ color: 'var(--df-t2)' }}>{item.name}</span>
                    </div>
                    <span className="font-semibold" style={{ color: 'var(--df-t1)' }}>{item.value}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="space-y-4 mt-2">
              {chartEntries.slice(0, 5).map(([key, chart], i) => {
                const total = chart.data.reduce((s, d) => s + (d.count || 0), 0)
                const max   = Math.max(...chart.data.map(d => d.count || 0))
                return (
                  <div key={key} className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <span style={{ color: 'var(--df-t2)' }}>{key.replace('dist_', '')}</span>
                      <span style={{ color: 'var(--df-t3)' }}>{total} pts</span>
                    </div>
                    <div className="h-1.5 w-full rounded-full overflow-hidden"
                      style={{ background: isDark ? '#1e293b' : '#e2e8f0' }}>
                      <div className="h-full rounded-full"
                        style={{ width: `${Math.min(100, (max / (total || 1)) * 100).toFixed(0)}%`, background: COLORS[i % COLORS.length] }} />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Multi-feature bar charts */}
      {chartEntries.filter(([, v]) => v.type === 'bar').length > 1 && (
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <h3 className="font-bold text-lg mb-6" style={{ color: 'var(--df-t1)' }}>Multi-Feature Distributions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8">
            {chartEntries.filter(([, v]) => v.type === 'bar').slice(1, 4).map(([key, chart]) => (
              <div key={key}>
                <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: 'var(--df-t3)' }}>
                  {key.replace('dist_', '')}
                </p>
                <div className="h-36">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chart.data} margin={{ top: 0, right: 0, left: -30, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                      <XAxis dataKey="bin" stroke={axisColor} fontSize={9} tickLine={false} />
                      <YAxis stroke={axisColor} fontSize={9} tickLine={false} axisLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="count" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default DashboardView
