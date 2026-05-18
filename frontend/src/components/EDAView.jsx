import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line
} from 'recharts'
import { BarChart3, PieChart as PieIcon, TrendingUp, Table } from 'lucide-react'

const COLORS = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1']

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 shadow-2xl text-sm">
      <p className="text-slate-400 mb-1">{label}</p>
      <p className="text-white font-bold">{payload[0]?.value}</p>
    </div>
  )
}

const ChartCard = ({ title, subtitle, children }) => (
  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 hover:border-slate-700 transition-colors">
    <p className="text-slate-500 text-[10px] font-bold uppercase tracking-widest mb-0.5">{subtitle}</p>
    <h4 className="text-white font-bold mb-5">{title}</h4>
    {children}
  </div>
)

const EDAView = ({ data }) => {
  const [activeStatTab, setActiveStatTab] = useState('summary')
  if (!data) return null
  const { charts = {}, summary = {}, correlation = {}, missing = {}, rows = 0, columns = 0 } = data

  const chartEntries = Object.entries(charts)
  const barCharts = chartEntries.filter(([, v]) => v.type === 'bar')
  const pieCharts = chartEntries.filter(([, v]) => v.type === 'pie')

  const summaryRows = Object.entries(summary).slice(0, 10)
  const missingEntries = Object.entries(missing).filter(([, v]) => v > 0)

  return (
    <div className="space-y-8">
      {/* Overview strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Total Rows',        value: rows?.toLocaleString() ?? 0,    color: 'text-sky-400' },
          { label: 'Total Columns',     value: columns,                        color: 'text-violet-400' },
          { label: 'Numeric Features',  value: barCharts.length,               color: 'text-emerald-400' },
          { label: 'Missing Cells',     value: missingEntries.length > 0 ? Object.values(missing).reduce((a,b)=>a+b,0) : 0, color: missingEntries.length > 0 ? 'text-amber-400' : 'text-emerald-400' },
        ].map((s) => (
          <div key={s.label} className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <p className="text-slate-500 text-xs uppercase tracking-widest font-semibold">{s.label}</p>
            <p className={`text-3xl font-black mt-1 ${s.color}`}>{s.value}</p>
          </div>
        ))}
      </div>

      {/* Stats table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
        <div className="flex items-center justify-between px-6 py-5 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <Table size={18} className="text-sky-400" />
            <h3 className="text-white font-bold">Statistical Summary</h3>
          </div>
          <div className="flex text-xs">
            {['summary', 'missing'].map(t => (
              <button key={t} onClick={() => setActiveStatTab(t)}
                className={`px-4 py-1.5 rounded-lg capitalize font-medium transition-colors ${
                  activeStatTab === t ? 'bg-sky-500/20 text-sky-400' : 'text-slate-500 hover:text-slate-300'
                }`}>{t}</button>
            ))}
          </div>
        </div>
        <div className="overflow-x-auto">
          {activeStatTab === 'summary' ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/80">
                  {['Feature', 'Count', 'Mean', 'Std', 'Min', 'Median', 'Max'].map(h => (
                    <th key={h} className="px-5 py-3 text-left text-slate-500 text-[11px] uppercase tracking-wider font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {summaryRows.map(([col, s]) => (
                  <tr key={col} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-5 py-3.5 text-white font-semibold">{col}</td>
                    <td className="px-5 py-3.5 text-slate-400">{s.count ?? '—'}</td>
                    <td className="px-5 py-3.5 text-slate-300">{typeof s.mean === 'number' ? s.mean.toFixed(2) : '—'}</td>
                    <td className="px-5 py-3.5 text-slate-300">{typeof s.std === 'number' ? s.std.toFixed(2) : '—'}</td>
                    <td className="px-5 py-3.5 text-slate-400">{s.min ?? '—'}</td>
                    <td className="px-5 py-3.5 text-slate-400">{s['50%'] ?? '—'}</td>
                    <td className="px-5 py-3.5 text-slate-400">{s.max ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="px-5 py-3 text-left text-slate-500 text-[11px] uppercase tracking-wider">Column</th>
                  <th className="px-5 py-3 text-left text-slate-500 text-[11px] uppercase tracking-wider">Missing Count</th>
                  <th className="px-5 py-3 text-left text-slate-500 text-[11px] uppercase tracking-wider">% Missing</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {Object.entries(missing).map(([col, count]) => (
                  <tr key={col} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-5 py-3.5 text-white font-semibold">{col}</td>
                    <td className="px-5 py-3.5">
                      <span className={`font-bold ${count > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>{count}</span>
                    </td>
                    <td className="px-5 py-3.5 text-slate-400">
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
          <h3 className="text-white font-bold text-lg mb-4 flex items-center gap-2">
            <BarChart3 size={18} className="text-sky-400" /> Distributions
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
            {barCharts.map(([key, cfg]) => (
              <ChartCard key={key} title={key.replace('dist_', '')} subtitle="Distribution">
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={cfg.data} margin={{ top: 0, right: 0, left: -25, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="bin" stroke="#475569" fontSize={9} tickLine={false} />
                      <YAxis stroke="#475569" fontSize={9} tickLine={false} axisLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="count" fill="#0ea5e9" radius={[3, 3, 0, 0]}>
                        {cfg.data.map((_, i) => (
                          <Cell key={i} fill={COLORS[i % COLORS.length]} fillOpacity={0.85} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </ChartCard>
            ))}
          </div>
        </div>
      )}

      {/* Pie / categorical charts */}
      {pieCharts.length > 0 && (
        <div>
          <h3 className="text-white font-bold text-lg mb-4 flex items-center gap-2">
            <PieIcon size={18} className="text-violet-400" /> Categorical Breakdown
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
            {pieCharts.map(([key, cfg]) => (
              <ChartCard key={key} title={key.replace('count_', '')} subtitle="Category">
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
                    <div key={i} className="flex items-center gap-1.5 text-xs text-slate-500">
                      <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                      <span className="truncate">{d.name}</span>
                    </div>
                  ))}
                </div>
              </ChartCard>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default EDAView
