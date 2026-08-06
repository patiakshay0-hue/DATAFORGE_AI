import React, { useState } from 'react'
import axios from 'axios'
import { Upload, AlertCircle, Loader2, FileSpreadsheet, FileJson, Table2 } from 'lucide-react'
import { useTheme } from '../ThemeContext'

const formats = [
  { ext: 'CSV',  icon: Table2,        color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
  { ext: 'XLSX', icon: FileSpreadsheet,color: 'text-teal-400',   bg: 'bg-teal-500/10'    },
  { ext: 'JSON', icon: FileJson,       color: 'text-cyan-400',   bg: 'bg-cyan-500/10'    },
]

const FileUpload = ({ onUploadSuccess }) => {
  const { isDark } = useTheme()
  const [dragActive, setDragActive] = useState(false)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)
  const [progress,   setProgress]   = useState(0)
  const [slow,       setSlow]       = useState(false)

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
    setLoading(true); setError(null); setProgress(0); setSlow(false)
    // A sleeping free-tier backend answers nothing for ~a minute. Say so rather
    // than leaving the user watching a bar that has already reached 100%.
    const slowTimer = setTimeout(() => setSlow(true), 8000)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await axios.post(`${import.meta.env.VITE_API_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => setProgress(Math.round((e.loaded * 100) / e.total))
      })
      onUploadSuccess(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Check your file format and try again.')
    } finally {
      clearTimeout(slowTimer)
      setLoading(false)
    }
  }

  const dropBorderColor = dragActive
    ? 'var(--df-primary)'
    : isDark ? '#1e293b' : '#e2e8f0'

  const dropBg = dragActive
    ? isDark ? 'rgba(20,184,166,0.08)' : 'rgba(20,184,166,0.04)'
    : 'var(--df-card)'

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Drop zone */}
      <div
        onDragEnter={handleDrag} onDragLeave={handleDrag}
        onDragOver={handleDrag}  onDrop={handleDrop}
        className="relative rounded-2xl border-2 border-dashed transition-all duration-200 cursor-pointer"
        style={{ borderColor: dropBorderColor, background: dropBg }}
        onClick={() => !loading && document.getElementById('file-input').click()}
      >
        <input
          id="file-input" type="file"
          accept=".csv,.xlsx,.xls,.json"
          className="hidden"
          onChange={e => e.target.files?.[0] && processFile(e.target.files[0])}
        />
        <div className="py-16 px-8 flex flex-col items-center text-center">
          {loading ? (
            <>
              <Loader2 size={40} className="animate-spin mb-4" style={{ color: 'var(--df-primary)' }} />
              <p className="font-semibold mb-3" style={{ color: 'var(--df-t1)' }}>
                {progress < 100 ? 'Uploading…' : 'Analyzing your dataset…'}
              </p>
              <div className="w-48 h-1.5 rounded-full overflow-hidden" style={{ background: isDark ? '#1e293b' : '#e2e8f0' }}>
                <div className="h-full rounded-full transition-all duration-300" style={{ background: 'var(--df-primary)', width: `${progress}%` }} />
              </div>
              <p className="text-xs mt-2" style={{ color: 'var(--df-t3)' }}>{progress}%</p>
              {slow && (
                <p className="text-xs mt-3 max-w-xs" style={{ color: 'var(--df-t3)' }}>
                  The server may be waking from sleep — the first request after a quiet
                  period can take up to a minute. Later uploads are much faster.
                </p>
              )}
            </>
          ) : (
            <>
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5" style={{ background: 'rgba(20,184,166,0.1)', border: '1px solid rgba(20,184,166,0.2)' }}>
                <Upload size={28} style={{ color: 'var(--df-primary)' }} />
              </div>
              <p className="text-xl font-bold mb-2" style={{ color: 'var(--df-t1)' }}>
                Drop your dataset here
              </p>
              <p className="text-sm mb-1" style={{ color: 'var(--df-t2)' }}>
                or <span className="font-semibold cursor-pointer" style={{ color: 'var(--df-primary)' }}>browse files</span>
              </p>
              <p className="text-xs" style={{ color: 'var(--df-t3)' }}>
                Supports CSV, XLSX, and JSON up to 50 MB
              </p>
            </>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/30">
          <AlertCircle size={16} className="text-red-400 mt-0.5 shrink-0" />
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Supported formats */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: 'var(--df-t3)' }}>
          Supported Formats
        </p>
        <div className="grid grid-cols-3 gap-3">
          {formats.map(({ ext, icon: Icon, color, bg }) => (
            <div key={ext}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${
                isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'
              }`}
              style={{ background: 'var(--df-card)' }}>
              <div className={`p-2 rounded-lg ${bg}`}>
                <Icon size={16} className={color} />
              </div>
              <div>
                <p className="text-sm font-bold" style={{ color: 'var(--df-t1)' }}>{ext}</p>
                <p className="text-[10px]" style={{ color: 'var(--df-t3)' }}>Supported</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default FileUpload
