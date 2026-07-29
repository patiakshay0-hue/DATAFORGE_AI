import React, { useState, useRef } from 'react'
import axios from 'axios'
import {
  FileCog, UploadCloud, Loader2, AlertCircle, CheckCircle2, Download,
  FileSpreadsheet, FileJson, FileArchive, Images, Table2, ArrowRight,
  Play, ScanEye, FileText
} from 'lucide-react'
import { useTheme } from '../ThemeContext'

const API = import.meta.env.VITE_API_URL

const FORMAT_ICON = { xlsx: FileSpreadsheet, xls: FileSpreadsheet, json: FileJson, csv: Table2,
  tsv: Table2, txt: FileText, parquet: Table2 }

const ImportConvertView = ({ onDataLoaded, onNavigate }) => {
  const { isDark } = useTheme()
  const [inspecting, setInspecting] = useState(false)
  const [info,       setInfo]       = useState(null)   // inspect result
  const [converted,  setConverted]  = useState(null)   // convert result
  const [busy,       setBusy]       = useState('')     // action in flight
  const [error,      setError]      = useState(null)
  const [choice,     setChoice]     = useState({})     // {sheet} or {file}
  const fileInput = useRef(null)

  const card = `rounded-2xl p-6 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`

  const reset = () => { setInfo(null); setConverted(null); setError(null); setChoice({}) }

  const inspect = async (file) => {
    if (!file) return
    setInspecting(true); reset()
    try {
      const fd = new FormData(); fd.append('file', file)
      const res = await axios.post(`${API}/convert/inspect`, fd)
      setInfo(res.data)
      if (res.data.kind === 'tabular' && res.data.sheets) setChoice({ sheet: res.data.sheets[0] })
      if (res.data.kind === 'data_zip' && res.data.files?.length) setChoice({ file: res.data.files[0].name })
    } catch (e) {
      setError(e.response?.data?.detail || 'Could not read that file.')
    } finally { setInspecting(false) }
  }

  const runConvert = async () => {
    setBusy('convert'); setError(null)
    try {
      const res = await axios.post(`${API}/convert/convert`, { choice })
      setConverted(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Conversion failed.')
    } finally { setBusy('') }
  }

  const buildMetadata = async () => {
    setBusy('meta'); setError(null)
    try {
      const res = await axios.post(`${API}/convert/image-metadata`)
      setConverted(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Could not build metadata CSV.')
    } finally { setBusy('') }
  }

  const download = () => { window.location.href = `${API}/convert/download` }

  const loadForAnalysis = async () => {
    setBusy('load'); setError(null)
    try {
      const res = await axios.post(`${API}/convert/load`)
      onDataLoaded?.(res.data)          // sets the active dataset + navigates to preview
    } catch (e) {
      setError(e.response?.data?.detail || 'Could not load for analysis.')
    } finally { setBusy('') }
  }

  const trainClassifier = async () => {
    setBusy('vision'); setError(null)
    try {
      await axios.post(`${API}/convert/send-to-vision`)
      onNavigate?.('vision')            // Image Classifier auto-loads this dataset
    } catch (e) {
      setError(e.response?.data?.detail || 'Could not send images to the classifier.')
    } finally { setBusy('') }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl p-6" style={{ background: 'var(--df-card)', border: '1px solid var(--df-border)' }}>
        <div className="absolute inset-0 opacity-10" style={{ background: 'radial-gradient(circle at 50% 0%, #0ea5e9, transparent 60%)' }} />
        <div className="relative flex items-center gap-4">
          <div className="p-3 bg-sky-500/10 border border-sky-500/20 rounded-xl shrink-0">
            <FileCog size={22} className="text-sky-400" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-black" style={{ color: 'var(--df-t1)' }}>Import & Convert</h3>
            <p className="text-xs mt-0.5" style={{ color: 'var(--df-t3)' }}>
              Drop any file — Excel, JSON, TSV, Parquet, a zip of data, or a zip of images. We detect it, convert to CSV, and route it where it belongs.
            </p>
          </div>
        </div>
      </div>

      {/* Dropzone */}
      <input ref={fileInput} type="file" accept=".csv,.xlsx,.xls,.json,.tsv,.txt,.parquet,.zip" className="hidden"
        onChange={e => inspect(e.target.files?.[0])} />
      <div
        onClick={() => !inspecting && fileInput.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); inspect(e.dataTransfer.files?.[0]) }}
        className="rounded-2xl border-2 border-dashed p-10 text-center cursor-pointer"
        style={{ borderColor: 'var(--df-border)', background: 'var(--df-card)' }}>
        {inspecting ? (
          <div className="flex flex-col items-center gap-3">
            <Loader2 size={32} className="text-sky-400 animate-spin" />
            <p className="text-sm font-medium" style={{ color: 'var(--df-t2)' }}>Inspecting file…</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className="w-14 h-14 rounded-2xl flex items-center justify-center" style={{ background: 'rgba(14,165,233,0.1)', border: '1px solid rgba(14,165,233,0.2)' }}>
              <UploadCloud size={26} className="text-sky-400" />
            </div>
            <p className="text-base font-bold" style={{ color: 'var(--df-t1)' }}>Drop any file to convert</p>
            <p className="text-xs" style={{ color: 'var(--df-t3)' }}>
              .xlsx · .json · .tsv · .txt · .parquet · .zip (data or images) · or <span className="text-sky-400 font-semibold">browse</span>
            </p>
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
          <AlertCircle size={16} className="shrink-0" /> {error}
        </div>
      )}

      {/* Unsupported / single image */}
      {info && (info.kind === 'unsupported' || info.kind === 'image_single') && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-sm">
          <AlertCircle size={16} className="text-amber-400 shrink-0 mt-0.5" />
          <span className="text-amber-300">{info.note}</span>
        </div>
      )}

      {/* Tabular / data-zip: choose + convert */}
      {info && (info.kind === 'tabular' || info.kind === 'data_zip') && (
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <DetectedRow info={info} isDark={isDark} />

          {/* Excel sheet picker */}
          {info.kind === 'tabular' && info.sheets?.length > 1 && (
            <div className="mt-4">
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--df-t2)' }}>Worksheet</label>
              <select value={choice.sheet || info.sheets[0]} onChange={e => setChoice({ sheet: e.target.value })}
                className="rounded-lg px-3 py-2 text-sm outline-none border"
                style={{ background: 'var(--df-input-bg)', borderColor: 'var(--df-input-border)', color: 'var(--df-t1)' }}>
                {info.sheets.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          )}

          {/* Zip file picker */}
          {info.kind === 'data_zip' && (
            <div className="mt-4">
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--df-t2)' }}>
                {info.files.length} data file(s) in archive — pick one
              </label>
              <select value={choice.file || info.files[0].name} onChange={e => setChoice({ file: e.target.value })}
                className="w-full rounded-lg px-3 py-2 text-sm outline-none border"
                style={{ background: 'var(--df-input-bg)', borderColor: 'var(--df-input-border)', color: 'var(--df-t1)' }}>
                {info.files.map(f => <option key={f.name} value={f.name}>{f.name} · .{f.format}</option>)}
              </select>
            </div>
          )}

          <button onClick={runConvert} disabled={busy === 'convert'}
            className="mt-5 flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
            style={{ background: 'linear-gradient(135deg, #0ea5e9, #6366f1)' }}>
            {busy === 'convert' ? <Loader2 size={14} className="animate-spin" /> : <Table2 size={14} />}
            {info.already_csv ? 'Load CSV' : 'Convert to CSV'}
          </button>
        </div>
      )}

      {/* Image zip: preview + two actions */}
      {info && info.kind === 'image_zip' && (
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-violet-500/10 border border-violet-500/20"><Images size={16} className="text-violet-400" /></div>
            <div>
              <p className="font-bold text-sm" style={{ color: 'var(--df-t1)' }}>Image dataset detected</p>
              <p className="text-xs" style={{ color: 'var(--df-t3)' }}>{info.total} images · {Object.keys(info.classes).length} classes</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 mb-5">
            {Object.entries(info.classes).map(([c, n]) => (
              <span key={c} className="text-xs px-3 py-1.5 rounded-lg" style={{ background: 'var(--df-input-bg)', border: '1px solid var(--df-border)', color: 'var(--df-t2)' }}>
                {c} · <span className="text-violet-400 font-semibold">{n}</span>
              </span>
            ))}
          </div>
          {!info.trainable && (
            <p className="text-xs text-amber-400 mb-4 flex items-center gap-1.5">
              <AlertCircle size={12} /> Need ≥ 2 classes with ≥ 2 images each to train a classifier — metadata CSV still works.
            </p>
          )}
          <div className="flex flex-wrap gap-3">
            <button onClick={trainClassifier} disabled={!info.trainable || busy === 'vision'}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
              style={{ background: 'linear-gradient(135deg, #7c3aed, #6366f1)' }}>
              {busy === 'vision' ? <Loader2 size={14} className="animate-spin" /> : <ScanEye size={14} />}
              Train Classifier
            </button>
            <button onClick={buildMetadata} disabled={busy === 'meta'}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-colors disabled:opacity-50"
              style={{ background: 'var(--df-input-bg)', border: '1px solid var(--df-border)', color: 'var(--df-t2)' }}>
              {busy === 'meta' ? <Loader2 size={14} className="animate-spin" /> : <Table2 size={14} />}
              Build Metadata CSV
            </button>
          </div>
        </div>
      )}

      {/* Converted result: preview + download / analyse */}
      {converted && (
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
            <h4 className="font-bold text-sm flex items-center gap-2" style={{ color: 'var(--df-t1)' }}>
              <CheckCircle2 size={15} className="text-emerald-400" /> {converted.csv_name}
            </h4>
            <span className="text-xs" style={{ color: 'var(--df-t3)' }}>
              {converted.preview.n_rows.toLocaleString()} rows · {converted.preview.n_cols} columns
            </span>
          </div>

          <div className="overflow-x-auto rounded-xl" style={{ border: '1px solid var(--df-border)' }}>
            <table className="w-full text-xs" style={{ color: 'var(--df-t2)' }}>
              <thead>
                <tr style={{ background: 'var(--df-input-bg)' }}>
                  {converted.preview.columns.map(c => (
                    <th key={c} className="px-3 py-2 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--df-t2)' }}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {converted.preview.rows.map((row, i) => (
                  <tr key={i} style={{ borderTop: '1px solid var(--df-border)' }}>
                    {converted.preview.columns.map(c => (
                      <td key={c} className="px-3 py-2 whitespace-nowrap" style={{ color: 'var(--df-t3)' }}>{String(row[c])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap gap-3 mt-5">
            <button onClick={download}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-colors"
              style={{ background: 'var(--df-input-bg)', border: '1px solid var(--df-border)', color: 'var(--df-t2)' }}>
              <Download size={14} /> Download CSV
            </button>
            <button onClick={loadForAnalysis} disabled={busy === 'load'}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
              style={{ background: 'linear-gradient(135deg, #10b981, #0ea5e9)' }}>
              {busy === 'load' ? <Loader2 size={14} className="animate-spin" /> : <ArrowRight size={14} />}
              Load for Analysis
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

const DetectedRow = ({ info, isDark }) => {
  const Icon = info.kind === 'data_zip' ? FileArchive : (FORMAT_ICON[info.format] || Table2)
  const label = info.kind === 'data_zip' ? 'Zip archive of data files' : `${(info.format || '').toUpperCase()} file`
  return (
    <div className="flex items-center gap-3">
      <div className="p-2 rounded-lg bg-sky-500/10 border border-sky-500/20"><Icon size={16} className="text-sky-400" /></div>
      <div>
        <p className="font-bold text-sm" style={{ color: 'var(--df-t1)' }}>{info.filename}</p>
        <p className="text-xs" style={{ color: 'var(--df-t3)' }}>
          {label}
          {info.kind === 'tabular' && info.preview && <> · {info.preview.n_rows} rows × {info.preview.n_cols} cols</>}
        </p>
      </div>
    </div>
  )
}

export default ImportConvertView
