import React, { useState } from 'react'
import axios from 'axios'
import { Upload, AlertCircle, Loader2, FileSpreadsheet, FileJson, Table2 } from 'lucide-react'

const formats = [
  { ext: 'CSV', icon: Table2, color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
  { ext: 'XLSX', icon: FileSpreadsheet, color: 'text-sky-400', bg: 'bg-sky-500/10' },
  { ext: 'JSON', icon: FileJson, color: 'text-violet-400', bg: 'bg-violet-500/10' },
]

const FileUpload = ({ onUploadSuccess }) => {
  const [dragActive, setDragActive] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [progress, setProgress] = useState(0)

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation()
    setDragActive(e.type === 'dragenter' || e.type === 'dragover')
  }

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files?.[0]) processFile(e.dataTransfer.files[0])
  }

  const processFile = async (file) => {
    setLoading(true); setError(null); setProgress(0)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => setProgress(Math.round((e.loaded * 100) / e.total))
      })
      onUploadSuccess(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Check your file format and try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto py-10 space-y-8">
      {/* Hero text */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-sky-500/10 border border-sky-500/20 rounded-full text-sky-400 text-xs font-semibold uppercase tracking-widest mb-2">
          <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-pulse" />
          Automated Analysis Engine
        </div>
        <h2 className="text-3xl font-black text-white">Upload Your Dataset</h2>
        <p className="text-slate-400 max-w-lg mx-auto text-sm leading-relaxed">
          Drop any CSV, Excel, or JSON file and our engine will automatically clean, analyze, and generate insights in seconds.
        </p>
      </div>

      {/* Drop Zone */}
      <label
        className={`relative flex flex-col items-center justify-center p-16 rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-300 ${
          dragActive
            ? 'border-sky-500 bg-sky-500/5 scale-[1.01]'
            : 'border-slate-700 bg-slate-900/50 hover:border-slate-600 hover:bg-slate-900'
        }`}
        onDragEnter={handleDrag} onDragLeave={handleDrag}
        onDragOver={handleDrag} onDrop={handleDrop}
      >
        <input
          type="file" className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          accept=".csv,.xlsx,.xls,.json"
          onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])}
          disabled={loading}
        />

        {loading ? (
          <div className="flex flex-col items-center gap-5 pointer-events-none">
            <div className="relative w-16 h-16">
              <div className="absolute inset-0 rounded-full border-2 border-sky-500/20" />
              <div className="absolute inset-0 rounded-full border-2 border-sky-400 border-t-transparent animate-spin" />
              <Loader2 className="absolute inset-0 m-auto text-sky-400 w-6 h-6 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '2s' }} />
            </div>
            <div className="text-center">
              <p className="text-white font-semibold">Processing your dataset…</p>
              <p className="text-slate-500 text-sm mt-1">Running EDA & schema detection</p>
            </div>
            <div className="w-48 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-sky-500 rounded-full transition-all duration-300"
                style={{ width: `${progress || 60}%` }}
              />
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-5 pointer-events-none">
            <div className={`w-18 h-18 p-5 rounded-2xl border transition-all duration-300 ${
              dragActive ? 'bg-sky-500/20 border-sky-500/40' : 'bg-slate-800 border-slate-700'
            }`}>
              <Upload size={32} className={dragActive ? 'text-sky-400' : 'text-slate-500'} />
            </div>
            <div className="text-center">
              <p className="text-white font-semibold text-lg">
                {dragActive ? 'Drop it here!' : 'Drag & drop your file'}
              </p>
              <p className="text-slate-500 text-sm mt-1">or click to browse your computer</p>
            </div>
          </div>
        )}
      </label>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400">
          <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}

      {/* Supported formats */}
      <div className="grid grid-cols-3 gap-4">
        {formats.map(({ ext, icon: Icon, color, bg }) => (
          <div key={ext} className="flex items-center gap-3 p-4 bg-slate-900 border border-slate-800 rounded-xl">
            <div className={`p-2.5 rounded-lg ${bg}`}>
              <Icon size={18} className={color} />
            </div>
            <div>
              <p className="text-white text-sm font-semibold">{ext}</p>
              <p className="text-slate-600 text-xs">Supported</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default FileUpload
