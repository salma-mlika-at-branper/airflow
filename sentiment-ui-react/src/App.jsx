import { useState, useRef, useEffect, useCallback } from 'react'
import './App.css'

function SentimentPill({ label }) {
  const map = {
    positive: { color: 'var(--accent)',  bg: 'rgba(74,240,176,0.1)',  border: 'rgba(74,240,176,0.25)' },
    negative: { color: 'var(--red)',     bg: 'rgba(255,79,106,0.1)',  border: 'rgba(255,79,106,0.25)' },
    neutral:  { color: 'var(--blue)',    bg: 'rgba(91,143,255,0.1)',  border: 'rgba(91,143,255,0.25)' },
  }
  const s = map[(label || '').toLowerCase()] || map.neutral
  return (
    <span style={{
      fontFamily: 'var(--font-mono)',
      fontSize: '0.65rem',
      fontWeight: 700,
      letterSpacing: '0.12em',
      textTransform: 'uppercase',
      color: s.color,
      background: s.bg,
      border: `1px solid ${s.border}`,
      borderRadius: '3px',
      padding: '2px 8px',
    }}>{label}</span>
  )
}

function ScoreBar({ label, value }) {
  const colors = {
    positive: 'var(--accent)',
    negative: 'var(--red)',
    neutral:  'var(--blue)',
  }
  const color = colors[(label || '').toLowerCase()] || 'var(--text-dim)'
  return (
    <div style={{ marginBottom: '6px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</span>
        <span style={{ fontSize: '0.65rem', color, fontWeight: 700 }}>{parseFloat(value).toFixed(2)}%</span>
      </div>
      <div style={{ height: '3px', background: 'var(--surface2)', borderRadius: '2px', overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${Math.min(parseFloat(value), 100)}%`,
          background: color,
          borderRadius: '2px',
          transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1)',
          boxShadow: `0 0 6px ${color}`,
        }} />
      </div>
    </div>
  )
}

function SentimentCard({ data }) {
  const scores = data.scores
    ? typeof data.scores === 'object' && !Array.isArray(data.scores)
      ? Object.entries(data.scores).map(([k, v]) => ({ label: k, value: v }))
      : data.scores.map(s => ({ label: s.label, value: s.score > 1 ? s.score : s.score * 100 }))
    : []

  return (
    <div className="sentiment-card">
      <div className="sentiment-card-header">
        <span className="card-tag">// sentiment.result</span>
        <SentimentPill label={data.label} />
      </div>
      <div className="confidence-row">
        <span className="conf-label">confidence</span>
        <span className="conf-value">{parseFloat(data.confidence).toFixed(2)}%</span>
      </div>
      <div style={{ marginTop: '10px' }}>
        {scores.map(s => <ScoreBar key={s.label} label={s.label} value={s.value} />)}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="message assistant">
      <div className="msg-meta">
        <span className="msg-role">SENTINEL</span>
      </div>
      <div className="typing-dots">
        <span /><span /><span />
      </div>
    </div>
  )
}

function Message({ msg }) {
  if (!msg) return null
  const isUser = msg.role === 'user'
  const isSystem = msg.role === 'system'

  if (isSystem) {
    return (
      <div className="system-message">
        <span>{msg.content}</span>
      </div>
    )
  }

  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="msg-meta">
        <span className="msg-role">{isUser ? 'YOU' : 'SENTINEL'}</span>
        <span className="msg-time">{msg.time}</span>
      </div>
      <div className="msg-body">
        {msg.sentimentData && <SentimentCard data={msg.sentimentData} />}
        {msg.content && (
          <p className="msg-text" style={{
            fontFamily: msg.sentimentData ? 'var(--font-mono)' : 'var(--font-body)',
            fontSize: msg.sentimentData ? '0.8rem' : '0.95rem'
          }}>
            {msg.content}
          </p>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [messages, setMessages]   = useState([])
  const [input, setInput]         = useState('')
  const [loading, setLoading]     = useState(false)
  const [mode, setMode]           = useState('test')
  const [history, setHistory]     = useState([])
  const bottomRef                 = useRef(null)
  const inputRef                  = useRef(null)

  useEffect(() => {
    const boot = [
      { role: 'system', content: '> SENTINEL v1.0 — Tunisian Dialect Sentiment Engine' },
      { role: 'system', content: '> Model loaded: cardiffnlp/twitter-xlm-roberta-base (fine-tuned)' },
      { role: 'system', content: '> LLM backend: mistral:7b via Ollama' },
      { role: 'system', content: '> Ready. Type a message or paste text to analyse.' },
    ]
    let i = 0
    const tick = setInterval(() => {
      if (i >= boot.length) { clearInterval(tick); return }
      const msg = boot[i]
      if (msg) setMessages(prev => [...prev, msg])
      i++
    }, 300)
    return () => clearInterval(tick)
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const now = () => new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })

  const addMessage = (role, content, extra = {}) => {
    setMessages(prev => [...prev, { role, content, time: now(), ...extra }])
  }

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    setLoading(true)
    addMessage('user', text)

    try {
      if (mode === 'test') {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        addMessage('assistant', null, { sentimentData: data })

      } else {
        const res = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        addMessage('assistant', null, { sentimentData: data })
        if (data.opinion) addMessage('assistant', data.opinion)
      }

    } catch (err) {
      addMessage('assistant', `[ERROR] ${err.message}`)
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [input, loading, mode])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const clearChat = () => {
    setMessages([])
    setHistory([])
    setTimeout(() => addMessage('system', '> Session cleared. New conversation started.'), 100)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="status-dot" />
          <span className="header-title">SENTINEL</span>
          <span className="header-sub">Tunisian Sentiment Engine</span>
        </div>
        <div className="header-right">
          <div className="mode-toggle">
             <button className={`mode-btn ${mode === 'test' ? 'active' : ''}`} onClick={() => setMode('test')}>TEST</button>
             <button className={`mode-btn ${mode === 'interpret' ? 'active' : ''}`} onClick={() => setMode('interpret')}>GENERATE INTERPRETATION</button>
            </div>
          <button className="clear-btn" onClick={clearChat}>CLR</button>
        </div>
      </header>

      <div className="messages-wrap">
        <div className="messages">
          {messages.map((msg, i) => <Message key={i} msg={msg} />)}
          {loading && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="input-area">
        <div className="input-mode-hint">
          {mode === 'test'
            ? '// TEST mode — paste text to classify'
            : '// INTERPRET mode — paste text for interpretation'}
        </div>
        <div className="input-row">
          <span className="prompt-caret">&gt;</span>
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={mode === 'analyse' ? 'Paste text to analyse...' : 'Ask something...'}
            rows={1}
            dir="auto"
            disabled={loading}
            onInput={e => {
              e.target.style.height = 'auto'
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
            }}
          />
          <button
            className={`send-btn ${loading ? 'loading' : ''}`}
            onClick={handleSend}
            disabled={loading || !input.trim()}
          >
            {loading ? <span className="send-spinner" /> : 'SEND'}
          </button>
        </div>
        <div className="input-footer">
          <span>SHIFT+ENTER for newline · ENTER to send</span>
          <span>{input.length} chars</span>
        </div>
      </div>
    </div>
  )
}