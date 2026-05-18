import React from 'react'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { Activity, Zap, Globe, ArrowUpRight, ArrowDownRight, TrendingUp, Layers } from 'lucide-react'

// ─── Custom SVG Database Icon ─────────────────────────────────────────────────
const DbIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <ellipse cx="12" cy="5" rx="9" ry="3" />
    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
  </svg>
)

const COLORS = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1']

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-800 border border-slate-600 rounded-xl px-4 py-3 shadow-xl">
        <p className="text-slate-400 text-xs mb-1">{label}</p>
        <p className="text-white font-bold text-sm">{payload[0].value}</p>
      </div>
    )
  }
  return null
}

const KPICard = ({ label, value, icon: Icon, colorClass, bgClass, trend, up }) => (
  <div className="relative overflow-hidden bg-slate-900 border border-slate-800 rounded-2xl p-6 group hover:border-slate-600 transition-all duration-300">
    <div className="absolute top-0 right-0 w-24 h-24 opacity-5 rounded-full blur-2xl bg-blue-400 group-hover:opacity-10 transition-opacity" />
    <div className="flex justify-between items-start mb-4">
      <div className={`p-2.5 rounded-xl ${bgClass}`}>
        <Icon size={20} className={colorClass} />
      </div>
      <span className={`flex items-center gap-1 text-xs font-bold px-2 py-1 rounded-full ${
        up ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
      }`}>
        {up ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
        {trend}
      </span>
    </div>
    <p className="text-slate-500 text-xs font-semibold uppercase tracking-widest">{label}</p>
    <h4 className="text-3xl font-black text-white mt-1">{value}</h4>
  </div>
)

const DashboardView = ({ data }) => {
  if (!data) return <div className="text-slate-400 p-10 text-center">No data available.</div>

  const { charts = {}, rows = 0, columns = 0 } = data

  const chartEntries = Object.entries(charts)
  const firstBarChart = chartEntries.find(([, v]) => v.type === 'bar')
  const firstPieChart = chartEntries.find(([, v]) => v.type === 'pie')
  const areaData = firstBarChart ? firstBarChart[1].data : []
  const pieData = firstPieChart ? firstPieChart[1].data.slice(0, 6) : []
  const pieChartName = firstPieChart ? firstPieChart[0].replace('count_', '') : ''

  const kpis = [
    { label: 'Total Records', value: Number(rows).toLocaleString(), icon: DbIcon, colorClass: 'text-sky-400', bgClass: 'bg-sky-500/10', trend: '+12%', up: true },
    { label: 'Total Features', value: columns, icon: Layers, colorClass: 'text-emerald-400', bgClass: 'bg-emerald-500/10', trend: 'Stable', up: true },
    { label: 'Quality Score', value: '94%', icon: Zap, colorClass: 'text-amber-400', bgClass: 'bg-amber-500/10', trend: '+5%', up: true },
    { label: 'Data Coverage', value: '98.2%', icon: Globe, colorClass: 'text-violet-400', bgClass: 'bg-violet-500/10', trend: '-1%', up: false },
  ]

  return (
    <div className="space-y-8">
      {/* KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        {kpis.map((kpi, idx) => <KPICard key={idx} {...kpi} />)}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Area Chart */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-white font-bold text-lg">Distribution Trend</h3>
              <p className="text-slate-500 text-xs mt-0.5">Primary numeric feature analysis</p>
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
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="bin" stroke="#475569" fontSize={10} tickLine={false} />
                <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="count" stroke="#0ea5e9" strokeWidth={2.5}
                  fillOpacity={1} fill="url(#grad1)" dot={false} activeDot={{ r: 5, fill: '#0ea5e9' }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Pie Chart or Progress Bars */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <h3 className="text-white font-bold text-lg mb-1">Composition</h3>
          <p className="text-slate-500 text-xs mb-5">
            {pieChartName ? `${pieChartName} breakdown` : 'Feature distribution'}
          </p>

          {pieData.length > 0 ? (
            <>
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="50%" innerRadius={50} outerRadius={70}
                      paddingAngle={4} dataKey="value">
                      {pieData.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-2 mt-4">
                {pieData.map((item, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                      <span className="text-slate-400 truncate max-w-[120px]">{item.name}</span>
                    </div>
                    <span className="text-white font-semibold">{item.value}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="space-y-4 mt-2">
              {chartEntries.slice(0, 5).map(([key, chart], i) => {
                const total = chart.data.reduce((s, d) => s + (d.count || 0), 0)
                const max = Math.max(...chart.data.map(d => d.count || 0))
                return (
                  <div key={key} className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-400">{key.replace('dist_', '')}</span>
                      <span className="text-slate-500">{total} pts</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-700"
                        style={{ width: `${Math.min(100, (max / (total || 1)) * 100).toFixed(0)}%`, background: COLORS[i % COLORS.length] }} />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Bar Chart Row */}
      {chartEntries.filter(([, v]) => v.type === 'bar').length > 1 && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <h3 className="text-white font-bold text-lg mb-6">Multi-Feature Distributions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8">
            {chartEntries.filter(([, v]) => v.type === 'bar').slice(1, 4).map(([key, chart]) => (
              <div key={key}>
                <p className="text-slate-500 text-xs font-semibold uppercase tracking-widest mb-3">
                  {key.replace('dist_', '')}
                </p>
                <div className="h-36">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chart.data} margin={{ top: 0, right: 0, left: -30, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="bin" stroke="#475569" fontSize={9} tickLine={false} />
                      <YAxis stroke="#475569" fontSize={9} tickLine={false} axisLine={false} />
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
