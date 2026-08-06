import React, { useState, useMemo } from 'react'
import { Hash, Type, Calendar, ToggleLeft, Search, ChevronLeft, ChevronRight } from 'lucide-react'
import { useTheme } from '../ThemeContext'

const PAGE_SIZE = 10

const typeMap = {
  numeric:     { icon: Hash,       color: 'text-teal-400',    bg: 'bg-teal-500/10',    label: 'Numeric' },
  categorical: { icon: Type,       color: 'text-teal-400', bg: 'bg-teal-500/10', label: 'Text'    },
  datetime:    { icon: Calendar,   color: 'text-emerald-400',bg: 'bg-emerald-500/10',label: 'Date'    },
  boolean:     { icon: ToggleLeft, color: 'text-amber-400',  bg: 'bg-amber-500/10',  label: 'Bool'    },
}

const PreviewView = ({ data }) => {
  const { isDark } = useTheme()
  const [search, setSearch] = useState('')
  const [page,   setPage]   = useState(0)

  if (!data?.schema || !data?.preview) return null
  const { schema, preview } = data

  const filteredSchema = useMemo(() =>
    schema.filter(c => c.name.toLowerCase().includes(search.toLowerCase())), [schema, search])

  const totalPages  = Math.ceil(preview.length / PAGE_SIZE)
  const visibleRows = preview.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  const card    = `rounded-2xl border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`
  const btnPage = `p-1.5 rounded-lg border text-sm font-semibold transition-colors ${
    isDark
      ? 'bg-slate-800 border-slate-700 text-slate-400 hover:text-white disabled:opacity-30'
      : 'bg-white border-slate-200 text-slate-500 hover:text-slate-900 disabled:opacity-30'
  }`

  return (
    <div className="space-y-8">
      {/* Schema Cards */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-bold text-lg" style={{ color: 'var(--df-t1)' }}>Schema Detection</h3>
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--df-t3)' }} />
            <input
              type="text" placeholder="Search columns…" value={search}
              onChange={e => setSearch(e.target.value)}
              className="rounded-lg pl-8 pr-4 py-2 text-sm outline-none w-48 border"
              style={{
                background: 'var(--df-input-bg)',
                borderColor: 'var(--df-input-border)',
                color: 'var(--df-t1)',
              }}
            />
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
          {filteredSchema.map((col) => {
            const t    = typeMap[col.type] || typeMap.categorical
            const Icon = t.icon
            return (
              <div key={col.name}
                className={`${card} rounded-xl p-4 transition-all duration-200 ${
                  isDark ? 'hover:border-slate-600' : 'hover:border-slate-300 hover:shadow-md'
                }`}
                style={{ background: 'var(--df-card)' }}>
                <div className="flex items-center justify-between mb-3">
                  <div className={`p-2 rounded-lg ${t.bg}`}>
                    <Icon size={14} className={t.color} />
                  </div>
                  <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded ${t.bg} ${t.color}`}>
                    {t.label}
                  </span>
                </div>
                <p className="font-semibold text-sm truncate" style={{ color: 'var(--df-t1)' }}>{col.name}</p>
                <div className="flex gap-3 mt-3 pt-3 text-xs" style={{ borderTop: '1px solid var(--df-border)', color: 'var(--df-t3)' }}>
                  <span>{col.unique} unique</span>
                  {col.missing > 0
                    ? <span className="text-amber-400 font-semibold">{col.missing} missing</span>
                    : <span className="text-emerald-500">complete</span>
                  }
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Data Table */}
      <div className={`${card} overflow-hidden`} style={{ background: 'var(--df-card)' }}>
        <div className="flex items-center justify-between px-6 py-4" style={{ borderBottom: '1px solid var(--df-border)' }}>
          <h3 className="font-bold" style={{ color: 'var(--df-t1)' }}>Raw Data Preview</h3>
          <div className="flex items-center gap-3 text-sm" style={{ color: 'var(--df-t3)' }}>
            <span>{preview.length} rows total</span>
            <span style={{ color: 'var(--df-t4)' }}>|</span>
            <span>{schema.length} columns</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{
                borderBottom: '1px solid var(--df-border)',
                background: isDark ? 'rgba(13,21,35,0.8)' : '#f8fafc',
              }}>
                <th className="px-5 py-3 text-left text-[11px] font-semibold w-10" style={{ color: 'var(--df-t4)' }}>#</th>
                {schema.map(col => (
                  <th key={col.name} className="px-5 py-3 text-left text-[11px] uppercase tracking-wider font-semibold whitespace-nowrap"
                    style={{ color: 'var(--df-t3)' }}>
                    {col.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--df-border)' }}
                  onMouseEnter={e => e.currentTarget.style.background = 'var(--df-row-hover)'}
                  onMouseLeave={e => e.currentTarget.style.background = ''}>
                  <td className="px-5 py-3 font-mono text-xs" style={{ color: 'var(--df-t4)' }}>
                    {page * PAGE_SIZE + i + 1}
                  </td>
                  {schema.map(col => {
                    const val   = String(row[col.name] ?? '')
                    const isNum = col.type === 'numeric'
                    const isEmpty = val === '' || val === 'null' || val === 'undefined'
                    return (
                      <td key={col.name}
                        className="px-5 py-3 whitespace-nowrap max-w-40 truncate"
                        style={{ color: isNum ? '#38bdf8' : 'var(--df-t2)', fontFamily: isNum ? 'monospace' : undefined }}
                        title={val}>
                        {isEmpty
                          ? <span className="italic text-xs" style={{ color: 'var(--df-t4)' }}>null</span>
                          : val}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {totalPages > 1 && (
          <div className="flex items-center justify-between px-6 py-4" style={{ borderTop: '1px solid var(--df-border)' }}>
            <p className="text-xs" style={{ color: 'var(--df-t3)' }}>
              Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, preview.length)} of {preview.length}
            </p>
            <div className="flex items-center gap-2">
              <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
                className={btnPage}>
                <ChevronLeft size={14} />
              </button>
              {Array.from({ length: totalPages }, (_, i) => (
                <button key={i} onClick={() => setPage(i)}
                  className={`w-7 h-7 rounded-lg text-xs font-semibold transition-colors ${
                    i === page
                      ? 'bg-teal-600 text-white'
                      : isDark
                        ? 'bg-slate-800 border border-slate-700 text-slate-400 hover:text-white'
                        : 'bg-white border border-slate-200 text-slate-500 hover:text-slate-900'
                  }`}>{i + 1}
                </button>
              ))}
              <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page === totalPages - 1}
                className={btnPage}>
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default PreviewView
