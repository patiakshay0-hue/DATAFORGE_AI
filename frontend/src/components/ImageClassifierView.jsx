import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import {
  Images, UploadCloud, Loader2, Play, RotateCcw, AlertCircle, CheckCircle2,
  Cpu, Zap, Clock, Layers, TrendingUp, Grid3x3, ImagePlus, FolderTree, ScanEye
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { useTheme } from '../ThemeContext'

const API = import.meta.env.VITE_API_URL

const ImageClassifierView = () => {
  const { isDark } = useTheme()
  const [ready,     setReady]     = useState(null)   // torch availability
  const [dataset,   setDataset]   = useState(null)   // upload summary
  const [uploading, setUploading] = useState(false)
  const [training,  setTraining]  = useState(false)
  const [trainStep, setTrainStep] = useState('')
  const [epochs,    setEpochs]    = useState(15)
  const [result,    setResult]    = useState(null)
  const [error,     setError]     = useState(null)
  const zipInput = useRef(null)

  useEffect(() => {
    axios.get(`${API}/vision/status`).then(r => setReady(r.data.ready)).catch(() => setReady(false))
    // pick up a dataset handed over from the Import & Convert tab
    axios.get(`${API}/vision/dataset`)
      .then(r => { if (r.data?.status === 'success') setDataset(r.data) })
      .catch(() => {})
  }, [])

  const card = `rounded-2xl p-6 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`

  const uploadZip = async (file) => {
    if (!file) return
    setUploading(true); setError(null); setDataset(null); setResult(null)
    try {
      const fd = new FormData(); fd.append('file', file)
      const res = await axios.post(`${API}/vision/upload`, fd)
      setDataset(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Could not read the image archive.')
    } finally { setUploading(false) }
  }

  const train = async () => {
    setTraining(true); setError(null); setResult(null)
    const steps = ['Decoding images…', 'Loading MobileNetV2 backbone…', 'Extracting embeddings…', 'Training classifier head…', 'Evaluating…']
    let i = 0; setTrainStep(steps[0])
    const iv = setInterval(() => { i++; if (i < steps.length) setTrainStep(steps[i]) }, 1200)
    try {
      const res = await axios.post(`${API}/vision/train`, { config: { epochs } })
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Training failed.')
    } finally { clearInterval(iv); setTraining(false) }
  }

  // ── torch missing ──────────────────────────────────────────────────────────
  if (ready === false) return (
    <div className="max-w-lg mx-auto py-20 text-center space-y-4">
      <div className="w-16 h-16 mx-auto rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
        <AlertCircle size={26} className="text-amber-400" />
      </div>
      <h3 className="text-lg font-bold" style={{ color: 'var(--df-t1)' }}>Image engine not installed</h3>
      <p className="text-sm" style={{ color: 'var(--df-t3)' }}>
        The CNN image classifier needs PyTorch + torchvision on the backend. Install them and restart the server:
      </p>
      <code className="block text-xs bg-black/30 rounded-lg px-4 py-2.5 text-emerald-400 font-mono">
        pip install torch torchvision
      </code>
    </div>
  )

  if (training) return (
    <div className="max-w-lg mx-auto py-16 text-center space-y-8">
      <div className="relative w-20 h-20 mx-auto">
        <div className="absolute inset-0 rounded-full border-2" style={{ borderColor: 'var(--df-border)' }} />
        <div className="absolute inset-0 rounded-full border-2 border-violet-400 border-t-transparent animate-spin" />
        <ScanEye size={24} className="absolute inset-0 m-auto text-violet-400" />
      </div>
      <div>
        <h3 className="text-xl font-bold" style={{ color: 'var(--df-t1)' }}>Training Image Classifier</h3>
        <p className="text-violet-400 text-sm mt-2 font-medium">{trainStep}</p>
        <p className="text-xs mt-3" style={{ color: 'var(--df-t3)' }}>
          {dataset?.total} images · {dataset?.classes?.length} classes · {epochs} epochs
        </p>
      </div>
    </div>
  )

  if (result) return <VisionResults result={result} dataset={dataset} isDark={isDark}
    onReset={() => setResult(null)} onNewData={() => { setResult(null); setDataset(null) }} />

  // ── Upload + dataset summary ─────────────────────────────────────────────────
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl p-6" style={{ background: 'var(--df-card)', border: '1px solid var(--df-border)' }}>
        <div className="absolute inset-0 opacity-10" style={{ background: 'radial-gradient(circle at 50% 0%, #8b5cf6, transparent 60%)' }} />
        <div className="relative flex items-center gap-4">
          <div className="p-3 bg-violet-500/10 border border-violet-500/20 rounded-xl shrink-0">
            <Images size={22} className="text-violet-400" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-black" style={{ color: 'var(--df-t1)' }}>Image Classifier</h3>
            <p className="text-xs mt-0.5" style={{ color: 'var(--df-t3)' }}>
              Upload a zip of labelled image folders, train a CNN, and classify new images
            </p>
          </div>
        </div>
      </div>

      {/* Dropzone */}
      <input ref={zipInput} type="file" accept=".zip" className="hidden"
        onChange={e => uploadZip(e.target.files?.[0])} />
      <div
        onClick={() => !uploading && zipInput.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); uploadZip(e.dataTransfer.files?.[0]) }}
        className="rounded-2xl border-2 border-dashed p-10 text-center cursor-pointer transition-colors"
        style={{ borderColor: 'var(--df-border)', background: 'var(--df-card)' }}>
        {uploading ? (
          <div className="flex flex-col items-center gap-3">
            <Loader2 size={32} className="text-violet-400 animate-spin" />
            <p className="text-sm font-medium" style={{ color: 'var(--df-t2)' }}>Unpacking & validating images…</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className="w-14 h-14 rounded-2xl flex items-center justify-center" style={{ background: 'rgba(139,92,246,0.1)', border: '1px solid rgba(139,92,246,0.2)' }}>
              <UploadCloud size={26} className="text-violet-400" />
            </div>
            <p className="text-base font-bold" style={{ color: 'var(--df-t1)' }}>Drop a .zip of images here</p>
            <p className="text-xs" style={{ color: 'var(--df-t3)' }}>or <span className="text-violet-400 font-semibold">browse files</span></p>
          </div>
        )}
      </div>

      {/* Zip structure hint */}
      <div className="flex items-start gap-3 p-4 rounded-xl" style={{ background: 'var(--df-input-bg)', border: '1px solid var(--df-border)' }}>
        <FolderTree size={16} className="text-violet-400 mt-0.5 shrink-0" />
        <div className="text-xs" style={{ color: 'var(--df-t3)' }}>
          <span className="font-semibold" style={{ color: 'var(--df-t2)' }}>Expected structure:</span> one folder per class, each holding that class's images.
          <code className="block mt-1.5 font-mono" style={{ color: 'var(--df-t2)' }}>
            dataset.zip → cats/ (img1.jpg, img2.jpg…) · dogs/ (img1.jpg…)
          </code>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
          <AlertCircle size={16} className="shrink-0" /> {error}
        </div>
      )}

      {/* Dataset summary */}
      {dataset && (
        <>
          <div className={card} style={{ background: 'var(--df-card)' }}>
            <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
              <h4 className="font-bold text-sm flex items-center gap-2" style={{ color: 'var(--df-t1)' }}>
                <CheckCircle2 size={15} className="text-emerald-400" /> Dataset ready
              </h4>
              <span className="text-xs" style={{ color: 'var(--df-t3)' }}>
                {dataset.total} images · {dataset.classes.length} classes
              </span>
            </div>
            <div className="space-y-4">
              {dataset.classes.map(cls => (
                <div key={cls}>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-sm font-semibold" style={{ color: 'var(--df-t1)' }}>{cls}</span>
                    <span className="text-xs" style={{ color: 'var(--df-t3)' }}>{dataset.counts[cls]} images</span>
                  </div>
                  <div className="flex gap-2 overflow-x-auto">
                    {(dataset.thumbnails[cls] || []).map((src, i) => (
                      <img key={i} src={src} alt={cls}
                        className="w-16 h-16 rounded-lg object-cover shrink-0"
                        style={{ border: '1px solid var(--df-border)' }} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
            {dataset.warnings?.length > 0 && (
              <p className="text-xs mt-4 text-amber-400 flex items-center gap-1.5">
                <AlertCircle size={12} /> {dataset.warnings.length} file(s) skipped as unreadable
              </p>
            )}
          </div>

          {/* Train controls */}
          <div className={card} style={{ background: 'var(--df-card)' }}>
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex-1 min-w-[200px]">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm font-medium flex items-center gap-2" style={{ color: 'var(--df-t2)' }}>
                    <Layers size={14} className="text-violet-400" /> Training epochs
                  </span>
                  <span className="text-sm font-bold font-mono text-violet-400">{epochs}</span>
                </div>
                <input type="range" min={5} max={40} step={1} value={epochs}
                  onChange={e => setEpochs(parseInt(e.target.value))}
                  className="w-full accent-violet-500 cursor-pointer" />
                <p className="text-[11px] mt-1" style={{ color: 'var(--df-t3)' }}>
                  Transfer learning on a frozen MobileNetV2 backbone — fast even on CPU
                </p>
              </div>
              <button onClick={train}
                className="flex items-center gap-2.5 px-7 py-3 rounded-xl font-bold text-white text-sm transition-all hover:scale-105 active:scale-95"
                style={{ background: 'linear-gradient(135deg, #7c3aed, #6366f1)' }}>
                <Play size={15} fill="currentColor" /> Train Classifier
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// ── Results ─────────────────────────────────────────────────────────────────
const VisionResults = ({ result, dataset, isDark, onReset, onNewData }) => {
  const {
    metrics = {}, history = [], classes = [], confusion_matrix: cm = [], labels = [],
    backbone, n_params, training_time, architecture = [], n_train, n_val,
  } = result
  const grid  = isDark ? '#1e293b' : '#e2e8f0'
  const axis  = isDark ? '#64748b' : '#94a3b8'
  const tipBg = isDark ? '#0f172a' : '#ffffff'
  const tip   = { background: tipBg, border: '1px solid var(--df-border)', borderRadius: 12, fontSize: 12, color: 'var(--df-t1)' }
  const card  = `rounded-2xl p-6 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`
  const cmMax = Math.max(1, ...cm.flat())

  return (
    <div className="space-y-8">
      {/* Banner */}
      <div className="relative overflow-hidden rounded-2xl p-6" style={{
        background: isDark
          ? 'linear-gradient(135deg, rgba(139,92,246,0.1) 0%, var(--df-card) 50%, rgba(99,102,241,0.1) 100%)'
          : 'linear-gradient(135deg, #f5f3ff 0%, #ffffff 50%, #eef2ff 100%)',
        border: '1px solid rgba(139,92,246,0.25)' }}>
        <div className="flex items-center gap-5 flex-wrap">
          <div className="p-4 bg-violet-500/15 border border-violet-500/25 rounded-2xl shrink-0">
            <Images size={28} className="text-violet-400" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs uppercase tracking-widest font-semibold" style={{ color: 'var(--df-t3)' }}>Image Classifier Trained</p>
            <h3 className="text-2xl font-black mt-0.5" style={{ color: 'var(--df-t1)' }}>
              Accuracy: {(metrics.accuracy * 100).toFixed(1)}%
            </h3>
            <p className="text-sm mt-0.5 flex items-center gap-3 flex-wrap" style={{ color: 'var(--df-t2)' }}>
              <span className="flex items-center gap-1"><Cpu size={12} /> {backbone}</span>
              <span className="flex items-center gap-1"><Zap size={12} /> {n_params?.toLocaleString()} params</span>
              <span className="flex items-center gap-1"><Clock size={12} /> {training_time}</span>
            </p>
          </div>
          <span className="hidden md:inline-block text-violet-400 font-bold text-sm bg-violet-500/10 px-3 py-1.5 rounded-lg border border-violet-500/20 shrink-0">
            {classes.length} classes · {n_train}/{n_val} split
          </span>
        </div>
      </div>

      {/* Curve + confusion */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className={card} style={{ background: 'var(--df-card)' }}>
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp size={15} className="text-sky-400" />
            <h4 className="font-bold text-sm" style={{ color: 'var(--df-t1)' }}>Training Progress</h4>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={history} margin={{ top: 5, right: 8, left: -8, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={grid} vertical={false} />
              <XAxis dataKey="epoch" stroke={axis} fontSize={11} tickLine={false} />
              <YAxis stroke={axis} fontSize={11} tickLine={false} width={44} />
              <Tooltip contentStyle={tip} labelFormatter={v => `Epoch ${v}`} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="train_loss" name="Train loss" stroke="#0ea5e9" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
              <Line type="monotone" dataKey="val_loss" name="Val loss" stroke="#f59e0b" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
              <Line type="monotone" dataKey="val_metric" name="Val accuracy" stroke="#10b981" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className={card} style={{ background: 'var(--df-card)' }}>
          <div className="flex items-center gap-2 mb-1">
            <Grid3x3 size={15} className="text-amber-400" />
            <h4 className="font-bold text-sm" style={{ color: 'var(--df-t1)' }}>Confusion Matrix</h4>
          </div>
          <p className="text-[11px] mb-4 ml-6" style={{ color: 'var(--df-t3)' }}>Rows = actual · Columns = predicted</p>
          <div className="overflow-x-auto">
            <table className="text-xs mx-auto" style={{ color: 'var(--df-t2)' }}>
              <thead>
                <tr><th></th>{labels.map(l => <th key={l} className="px-2 py-1 font-semibold" style={{ color: 'var(--df-t3)' }}>{l}</th>)}</tr>
              </thead>
              <tbody>
                {cm.map((row, i) => (
                  <tr key={i}>
                    <td className="px-2 py-1 font-semibold text-right" style={{ color: 'var(--df-t3)' }}>{labels[i]}</td>
                    {row.map((v, j) => {
                      const t = v / cmMax, correct = i === j
                      return (
                        <td key={j} className="w-12 h-12 text-center font-bold rounded-lg" style={{
                          background: correct ? `rgba(16,185,129,${0.12 + 0.5 * t})` : `rgba(239,68,68,${0.08 + 0.45 * t})`,
                          color: t > 0.5 ? '#fff' : 'var(--df-t1)', border: '2px solid var(--df-card)' }}>{v}</td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Architecture */}
      <div className={card} style={{ background: 'var(--df-card)' }}>
        <h4 className="font-bold text-sm mb-4 flex items-center gap-2" style={{ color: 'var(--df-t1)' }}>
          <Layers size={14} className="text-violet-400" /> Architecture
        </h4>
        <div className="flex flex-wrap gap-2">
          {architecture.map((layer, i) => (
            <React.Fragment key={i}>
              <span className="text-xs px-3 py-2 rounded-lg" style={{ background: 'var(--df-input-bg)', border: '1px solid var(--df-border)', color: 'var(--df-t2)' }}>{layer}</span>
              {i < architecture.length - 1 && <span className="self-center text-violet-400">→</span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Predict playground */}
      <VisionPredict classes={classes} isDark={isDark} />

      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={onReset}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-colors"
          style={{ background: isDark ? 'rgba(30,41,59,0.8)' : '#f1f5f9', border: '1px solid var(--df-border)', color: 'var(--df-t2)' }}>
          <RotateCcw size={14} /> Retrain
        </button>
        <button onClick={onNewData}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-colors"
          style={{ background: isDark ? 'rgba(30,41,59,0.8)' : '#f1f5f9', border: '1px solid var(--df-border)', color: 'var(--df-t2)' }}>
          <UploadCloud size={14} /> New dataset
        </button>
      </div>
    </div>
  )
}

// ── Single-image prediction ───────────────────────────────────────────────────
const VisionPredict = ({ classes, isDark }) => {
  const [pred, setPred]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const imgInput = useRef(null)
  const card = `rounded-2xl p-6 border ${isDark ? 'border-slate-800' : 'border-slate-200 shadow-sm'}`

  const classify = async (file) => {
    if (!file) return
    setLoading(true); setError(null); setPred(null)
    try {
      const fd = new FormData(); fd.append('file', file)
      const res = await axios.post(`${API}/vision/predict`, fd)
      setPred(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Prediction failed.')
    } finally { setLoading(false) }
  }

  return (
    <div className={card} style={{ background: 'var(--df-card)' }}>
      <div className="flex items-center gap-2 mb-1">
        <ImagePlus size={15} className="text-violet-400" />
        <h4 className="font-bold text-sm" style={{ color: 'var(--df-t1)' }}>Classify a New Image</h4>
      </div>
      <p className="text-[11px] mb-4 ml-6" style={{ color: 'var(--df-t3)' }}>
        Upload any image and the trained network predicts one of: {classes.join(', ')}
      </p>

      <input ref={imgInput} type="file" accept="image/*" className="hidden"
        onChange={e => classify(e.target.files?.[0])} />

      <div className="flex items-center gap-6 flex-wrap">
        <button onClick={() => imgInput.current?.click()} disabled={loading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-white text-sm transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
          style={{ background: 'linear-gradient(135deg, #7c3aed, #6366f1)' }}>
          {loading ? <Loader2 size={14} className="animate-spin" /> : <ImagePlus size={14} />}
          {loading ? 'Classifying…' : 'Upload Image'}
        </button>
        {error && <span className="text-red-400 text-sm flex items-center gap-1.5"><AlertCircle size={14} /> {error}</span>}
      </div>

      {pred && (
        <div className="mt-5 flex items-start gap-5 flex-wrap">
          <img src={pred.thumbnail} alt="uploaded" className="w-32 h-32 rounded-xl object-cover" style={{ border: '1px solid var(--df-border)' }} />
          <div className="flex-1 min-w-[200px]">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 size={18} className="text-emerald-400" />
              <span className="text-lg font-black text-emerald-400">{pred.prediction}</span>
              <span className="text-xs" style={{ color: 'var(--df-t3)' }}>({(pred.confidence * 100).toFixed(1)}% confident)</span>
            </div>
            <div className="space-y-2">
              {pred.distribution.slice(0, 6).map(d => (
                <div key={d.class} className="flex items-center gap-3">
                  <span className="text-xs font-mono w-20 truncate shrink-0" style={{ color: 'var(--df-t2)' }}>{d.class}</span>
                  <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: 'var(--df-input-bg)' }}>
                    <div className="h-full rounded-full" style={{ width: `${d.prob * 100}%`, background: 'linear-gradient(90deg, #7c3aed, #6366f1)' }} />
                  </div>
                  <span className="text-xs font-mono w-12 text-right shrink-0" style={{ color: 'var(--df-t3)' }}>{(d.prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ImageClassifierView
