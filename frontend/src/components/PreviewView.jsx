import React, { useState, useMemo } from 'react'
import { Hash, Type, Calendar, ToggleLeft, Search, ChevronLeft, ChevronRight } from 'lucide-react'

const PAGE_SIZE = 10

const typeMap = {
  numeric:     { icon: Hash,        color: 'text-sky-400',    bg: 'bg-sky-500/10',    label: 'Numeric' },
  categorical: { icon: Type,        color: 'text-violet-400', bg: 'bg-violet-500/10', label: 'Text' },
  datetime:    { icon: Calendar,    color: 'text-emerald-400',bg: 'bg-emerald-500/10',label: 'Date' },
  boolean:     { icon: ToggleLeft,  color: 'text-amber-400',  bg: 'bg-amber-500/10',  label: 'Bool' },
}

const PreviewView = ({ data }) => {
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)

  if (!data?.schema || !data?.preview) return null
  const { schema, preview, eda } = data

  const filteredSchema = useMemo(() =>
    schema.filter(c => c.name.toLowerCase().includes(search.toLowerCase())), [schema, search])

  const totalPages = Math.ceil(preview.length / PAGE_SIZE)
  const visibleRows = preview.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  return (
    <div className="space-y-8">
      {/* Schema Cards */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-bold text-lg">Schema Detection</h3>
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text" placeholder="Search columns…" value={search}
              onChange={e => setSearch(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg pl-8 pr-4 py-2 text-sm text-slate-300 placeholder-slate-600 outline-none focus:border-sky-500 transition-colors w-48"
            />
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
          {filteredSchema.map((col) => {
            const t = typeMap[col.type] || typeMap.categorical
            const Icon = t.icon
            return (
              <div key={col.name}
                className="bg-slate-900 border border-slate-800 rounded-xl p-4 hover:border-slate-600 transition-colors group">
                <div className="flex items-center justify-between mb-3">
                  <div className={`p-2 rounded-lg ${t.bg}`}>
                    <Icon size={14} className={t.color} />
                  </div>
                  <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded ${t.bg} ${t.color}`}>
                    {t.label}
                  </span>
                </div>
                <p className="text-white font-semibold text-sm truncate">{col.name}</p>
                <div className="flex gap-3 mt-3 pt-3 border-t border-slate-800 text-xs text-slate-600">
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
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <h3 className="text-white font-bold">Raw Data Preview</h3>
          <div className="flex items-center gap-3 text-sm text-slate-500">
            <span>{preview.length} rows total</span>
            <span className="text-slate-700">|</span>
            <span>{schema.length} columns</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80">
                <th className="px-5 py-3 text-left text-slate-600 text-[11px] font-semibold w-10">#</th>
                {schema.map(col => (
                  <th key={col.name} className="px-5 py-3 text-left text-slate-500 text-[11px] uppercase tracking-wider font-semibold whitespace-nowrap">
                    {col.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {visibleRows.map((row, i) => (
                <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                  <td className="px-5 py-3 text-slate-700 font-mono text-xs">{page * PAGE_SIZE + i + 1}</td>
                  {schema.map(col => {
                    const val = String(row[col.name] ?? '')
                    const isNum = col.type === 'numeric'
                    return (
                      <td key={col.name} className={`px-5 py-3 whitespace-nowrap max-w-[160px] truncate ${
                        isNum ? 'text-sky-300 font-mono' : 'text-slate-300'
                      }`} title={val}>
                        {val === '' || val === 'null' || val === 'undefined'
                          ? <span className="text-slate-700 italic text-xs">null</span>
                          : val}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-6 py-4 border-t border-slate-800">
            <p className="text-slate-600 text-xs">
              Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, preview.length)} of {preview.length}
            </p>
            <div className="flex items-center gap-2">
              <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
                className="p-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-400 hover:text-white disabled:opacity-30 transition-colors">
                <ChevronLeft size={14} />
              </button>
              {Array.from({ length: totalPages }, (_, i) => (
                <button key={i} onClick={() => setPage(i)}
                  className={`w-7 h-7 rounded-lg text-xs font-semibold transition-colors ${
                    i === page
                      ? 'bg-sky-600 text-white'
                      : 'bg-slate-800 border border-slate-700 text-slate-400 hover:text-white'
                  }`}>{i + 1}</button>
              ))}
              <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page === totalPages - 1}
                className="p-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-400 hover:text-white disabled:opacity-30 transition-colors">
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
