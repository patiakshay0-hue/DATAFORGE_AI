import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  Sparkles, Send, Loader2, Bot, User,
  MessageSquare, Trash2, Copy, Check, KeyRound
} from 'lucide-react'
import { useTheme } from '../ThemeContext'

const SUGGESTIONS = [
  'What are the top 3 insights from this dataset?',
  'Which columns have the strongest correlation?',
  'Are there any anomalies or outliers I should know about?',
  'Summarize the distribution of each numeric column.',
  'What machine learning task is this dataset best suited for?',
  'Which column would make the best target variable and why?',
]

// ── Markdown-lite renderer ────────────────────────────────────────────────────
const renderText = (text) => {
  if (!text) return null
  const lines = text.split('\n')
  const elements = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (line.startsWith('```')) {
      const codeLines = []
      i++
      while (i < lines.length && !lines[i].startsWith('```')) { codeLines.push(lines[i]); i++ }
      elements.push(
        <pre key={i} className="bg-slate-950 border border-slate-700 rounded-lg p-4 text-xs text-emerald-300 font-mono overflow-x-auto my-3 whitespace-pre-wrap">
          {codeLines.join('\n')}
        </pre>
      )
    } else if (line.startsWith('### ')) {
      elements.push(<p key={i} className="font-bold text-sm mt-3 mb-1" style={{ color: 'var(--df-t1)' }}>{line.slice(4)}</p>)
    } else if (line.startsWith('## ')) {
      elements.push(<p key={i} className="font-bold mt-3 mb-1" style={{ color: 'var(--df-t1)' }}>{line.slice(3)}</p>)
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      elements.push(
        <div key={i} className="flex gap-2 my-0.5">
          <span className="text-sky-400 mt-1 shrink-0">•</span>
          <span className="text-sm leading-relaxed" style={{ color: 'var(--df-t2)' }}>{inlineFormat(line.slice(2))}</span>
        </div>
      )
    } else if (/^\d+\.\s/.test(line)) {
      const num = line.match(/^(\d+)\./)[1]
      elements.push(
        <div key={i} className="flex gap-2 my-0.5">
          <span className="text-sky-400 shrink-0 w-5 text-sm">{num}.</span>
          <span className="text-sm leading-relaxed" style={{ color: 'var(--df-t2)' }}>{inlineFormat(line.replace(/^\d+\.\s/, ''))}</span>
        </div>
      )
    } else if (line === '---' || line === '***') {
      elements.push(<hr key={i} style={{ borderColor: 'var(--df-border)' }} className="my-3" />)
    } else if (line.trim() === '') {
      elements.push(<div key={i} className="h-2" />)
    } else {
      elements.push(<p key={i} className="text-sm leading-relaxed" style={{ color: 'var(--df-t2)' }}>{inlineFormat(line)}</p>)
    }
    i++
  }
  return elements
}

const inlineFormat = (text) => {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**'))
      return <strong key={i} className="font-semibold" style={{ color: 'var(--df-t1)' }}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`'))
      return <code key={i} className="bg-slate-800 text-emerald-300 px-1 py-0.5 rounded text-xs font-mono">{part.slice(1, -1)}</code>
    return part
  })
}

// ── Single message bubble ─────────────────────────────────────────────────────
const MessageBubble = ({ msg }) => {
  const { isDark } = useTheme()
  const [copied, setCopied] = useState(false)
  const isUser = msg.role === 'user'

  const copy = () => {
    navigator.clipboard.writeText(msg.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (isUser) return (
    <div className="flex justify-end gap-3 group">
      <div className="max-w-[75%]">
        <div className="rounded-2xl rounded-tr-md px-5 py-3"
          style={{
            background: isDark ? 'rgba(14,165,233,0.15)' : '#e0f2fe',
            border: `1px solid ${isDark ? 'rgba(14,165,233,0.25)' : '#bae6fd'}`,
          }}>
          <p className="text-sm leading-relaxed" style={{ color: 'var(--df-t1)' }}>{msg.content}</p>
        </div>
      </div>
      <div className="w-8 h-8 rounded-full bg-sky-600 flex items-center justify-center shrink-0 mt-1">
        <User size={14} className="text-white" />
      </div>
    </div>
  )

  return (
    <div className="flex gap-3 group">
      <div className="w-8 h-8 rounded-full shrink-0 mt-1 flex items-center justify-center"
        style={{ background: 'linear-gradient(135deg, #0ea5e9, #6366f1)' }}>
        <Bot size={14} className="text-white" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="rounded-2xl rounded-tl-md px-5 py-4"
          style={{ background: 'var(--df-card)', border: '1px solid var(--df-border)' }}>
          {msg.streaming
            ? <div className="space-y-1">{renderText(msg.content)}<span className="inline-block w-1.5 h-4 bg-sky-400 animate-pulse ml-0.5 rounded-sm align-middle" /></div>
            : <div className="space-y-1">{renderText(msg.content)}</div>
          }
        </div>
        {!msg.streaming && msg.content && (
          <button onClick={copy}
            className="flex items-center gap-1 mt-1.5 ml-2 text-xs transition-colors opacity-0 group-hover:opacity-100"
            style={{ color: copied ? '#34d399' : 'var(--df-t3)' }}>
            {copied ? <Check size={11} className="text-emerald-400" /> : <Copy size={11} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        )}
      </div>
    </div>
  )
}

// ── API key warning banner ────────────────────────────────────────────────────
const ApiKeyBanner = () => (
  <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex gap-3">
    <KeyRound size={16} className="text-amber-400 shrink-0 mt-0.5" />
    <div className="flex-1">
      <p className="text-amber-400 font-semibold text-sm">API Key Required</p>
      <p className="text-xs mt-1 leading-relaxed" style={{ color: 'var(--df-t2)' }}>
        Add your Anthropic API key to <code className="bg-slate-800 px-1 rounded text-emerald-300">backend/.env</code>:
      </p>
      <pre className="rounded-lg px-3 py-2 text-xs text-emerald-300 font-mono mt-2"
        style={{ background: isDark => isDark ? '#0d1523' : '#f1f5f9', border: '1px solid var(--df-border)' }}>
        ANTHROPIC_API_KEY=sk-ant-...
      </pre>
      <p className="text-xs mt-2" style={{ color: 'var(--df-t3)' }}>
        Get a key at <span className="text-sky-400">console.anthropic.com</span>, then restart the backend.
      </p>
    </div>
  </div>
)

// ── Main ChatView ─────────────────────────────────────────────────────────────
const ChatView = ({ data }) => {
  const { isDark } = useTheme()
  const [messages,    setMessages]    = useState([])
  const [input,       setInput]       = useState('')
  const [loading,     setLoading]     = useState(false)
  const [apiStatus,   setApiStatus]   = useState(null)
  const [checkingKey, setCheckingKey] = useState(true)
  const bottomRef = useRef(null)
  const inputRef  = useRef(null)

  useEffect(() => {
    const checkKey = async () => {
      try {
        const res  = await fetch(`${import.meta.env.VITE_API_URL}/chat/status`)
        const json = await res.json()
        setApiStatus(json)
      } catch {
        setApiStatus({ configured: false, message: 'Could not reach backend' })
      } finally {
        setCheckingKey(false)
      }
    }
    checkKey()
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = useCallback(async (text) => {
    const question = (text || input).trim()
    if (!question || loading) return
    setInput('')
    setLoading(true)

    const history = messages.filter(m => !m.streaming)
    setMessages(prev => [
      ...prev,
      { role: 'user', content: question },
      { role: 'assistant', content: '', streaming: true, id: Date.now() },
    ])

    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          history: history.map(m => ({ role: m.role, content: m.content })),
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Request failed')
      }

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let   buffer  = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6)
          if (raw === '[DONE]') break
          try {
            const parsed = JSON.parse(raw)
            if (parsed.error) throw new Error(parsed.error)
            if (parsed.text) {
              setMessages(prev => {
                const copy = [...prev]
                const last = copy[copy.length - 1]
                if (last?.streaming) last.content += parsed.text
                return copy
              })
            }
          } catch (e) {
            if (e.message !== 'Unexpected end of JSON input') {
              setMessages(prev => {
                const copy = [...prev]
                const last = copy[copy.length - 1]
                if (last?.streaming) { last.content = `Error: ${e.message}`; last.streaming = false }
                return copy
              })
            }
          }
        }
      }

      setMessages(prev => {
        const copy = [...prev]
        const last = copy[copy.length - 1]
        if (last?.streaming) last.streaming = false
        return copy
      })
    } catch (err) {
      setMessages(prev => {
        const copy = [...prev]
        const last = copy[copy.length - 1]
        if (last?.streaming) { last.content = `Error: ${err.message}`; last.streaming = false }
        return copy
      })
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [input, loading, messages])

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-full max-h-[calc(100vh-130px)]">

      {/* Header */}
      <div className="shrink-0 flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl shrink-0"
            style={{ background: 'linear-gradient(135deg, rgba(14,165,233,0.15), rgba(99,102,241,0.15))' }}>
            <Sparkles size={18} className="text-violet-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-bold" style={{ color: 'var(--df-t1)' }}>Chat with your Data</h3>
              <span className="text-[9px] font-black uppercase tracking-widest text-violet-400 bg-violet-500/10 border border-violet-500/20 px-2 py-0.5 rounded-full">Pro</span>
            </div>
            <p className="text-xs" style={{ color: 'var(--df-t3)' }}>
              {data?.filename ? `Analysing: ${data.filename} · ${data.eda?.rows?.toLocaleString()} rows` : 'Ask anything about your dataset'}
            </p>
          </div>
        </div>
        {messages.length > 0 && (
          <button onClick={() => setMessages([])}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
            style={{
              background: isDark ? 'rgba(30,41,59,0.8)' : '#f1f5f9',
              border: '1px solid var(--df-border)',
              color: 'var(--df-t2)',
            }}>
            <Trash2 size={12} /> Clear chat
          </button>
        )}
      </div>

      {/* API key status */}
      {!checkingKey && (
        !apiStatus?.configured ? (
          <div className="shrink-0 mb-5"><ApiKeyBanner /></div>
        ) : (
          <div className="shrink-0 mb-4 flex items-center gap-2 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-3 py-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Claude AI connected · {data?.filename ? `Context from ${data.filename}` : 'Dataset loaded'}
          </div>
        )
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-5 pr-1 min-h-0">
        {isEmpty && (
          <div className="py-8">
            <div className="relative overflow-hidden rounded-2xl p-8 text-center mb-8"
              style={{
                background: isDark
                  ? 'linear-gradient(135deg, #0d1523 0%, rgba(14,165,233,0.06) 50%, #0d1523 100%)'
                  : 'linear-gradient(135deg, #eff6ff 0%, #e0f2fe 50%, #eff6ff 100%)',
                border: '1px solid var(--df-border)',
              }}>
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center"
                style={{ background: 'linear-gradient(135deg, #0ea5e9, #6366f1)' }}>
                <MessageSquare size={28} className="text-white" />
              </div>
              <h4 className="font-bold text-lg" style={{ color: 'var(--df-t1)' }}>Ask anything about your data</h4>
              <p className="text-sm mt-2 max-w-md mx-auto leading-relaxed" style={{ color: 'var(--df-t2)' }}>
                Claude has full context of your dataset — schema, statistics, sample rows, and correlations.
                Ask in plain English.
              </p>
            </div>

            <p className="text-xs font-semibold uppercase tracking-widest mb-4 text-center" style={{ color: 'var(--df-t3)' }}>
              Suggested Questions
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {SUGGESTIONS.map((s, i) => (
                <button key={i} onClick={() => sendMessage(s)}
                  disabled={!apiStatus?.configured || loading}
                  className="text-left px-4 py-3 rounded-xl text-sm transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{
                    background: 'var(--df-card)',
                    border: '1px solid var(--df-border)',
                    color: 'var(--df-t2)',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(14,165,233,0.4)'; e.currentTarget.style.color = 'var(--df-t1)' }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--df-border)'; e.currentTarget.style.color = 'var(--df-t2)' }}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => <MessageBubble key={msg.id || i} msg={msg} />)}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="shrink-0 mt-4 pt-4" style={{ borderTop: '1px solid var(--df-border)' }}>
        <div className="flex gap-3 items-end rounded-2xl px-4 py-3 transition-colors"
          style={{
            background: 'var(--df-card)',
            border: `1px solid ${loading ? 'rgba(14,165,233,0.4)' : 'var(--df-border)'}`,
          }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder={
              !apiStatus?.configured
                ? 'Configure your API key to start chatting…'
                : 'Ask a question about your data… (Enter to send, Shift+Enter for new line)'
            }
            disabled={loading || !apiStatus?.configured}
            rows={1}
            className="flex-1 bg-transparent text-sm outline-none resize-none min-h-6 max-h-30 disabled:opacity-40"
            style={{ color: 'var(--df-t1)', lineHeight: '1.5' }}
            onInput={e => {
              e.target.style.height = 'auto'
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
            }}
          />
          <button
            onClick={() => sendMessage()}
            disabled={!input.trim() || loading || !apiStatus?.configured}
            className="shrink-0 w-9 h-9 rounded-xl flex items-center justify-center disabled:opacity-30 disabled:cursor-not-allowed hover:scale-105 active:scale-95"
            style={{ background: 'linear-gradient(135deg, #0284c7, #6366f1)' }}
          >
            {loading
              ? <Loader2 size={16} className="text-white animate-spin" />
              : <Send size={15} className="text-white" />}
          </button>
        </div>
        <p className="text-[10px] mt-2 text-center" style={{ color: 'var(--df-t4)' }}>
          Powered by Claude · Responses based on your uploaded dataset
        </p>
      </div>
    </div>
  )
}

export default ChatView
