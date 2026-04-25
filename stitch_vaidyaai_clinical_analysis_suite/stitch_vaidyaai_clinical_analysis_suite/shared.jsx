{

// shared.jsx — atomic UI components for VAIDYAA AI
// Exports to window for cross-script access

const { useState, useEffect, useRef } = React;

// ─── Tokens ──────────────────────────────────────────────────────────────────
const STATUS_META = {
  pending:    { label: 'Pending',    color: 'oklch(0.55 0.01 260)', bg: 'oklch(0.55 0.01 260 / 0.12)' },
  processing: { label: 'Processing', color: 'oklch(0.46 0.19 145)', bg: 'oklch(0.46 0.19 145 / 0.10)' },
  verified:   { label: 'Verified',   color: 'oklch(0.46 0.19 145)', bg: 'oklch(0.46 0.18 145 / 0.1)' },
  refuted:    { label: 'Refuted',    color: 'oklch(0.65 0.20 25)',  bg: 'oklch(0.65 0.20 25  / 0.12)' },
  uncertain:  { label: 'Uncertain',  color: 'oklch(0.75 0.14 60)',  bg: 'oklch(0.75 0.14 60  / 0.12)' },
  failed:     { label: 'Failed',     color: 'oklch(0.60 0.18 25)',  bg: 'oklch(0.60 0.18 25  / 0.12)' },
  success:    { label: 'Success',    color: 'oklch(0.46 0.19 145)', bg: 'oklch(0.46 0.18 145 / 0.1)' },
};

// ─── StatusBadge ─────────────────────────────────────────────────────────────
function StatusBadge({ status, size = 'sm' }) {
  const meta = STATUS_META[status] || STATUS_META.pending;
  const isProcessing = status === 'processing';
  const fs = size === 'lg' ? '13px' : '11px';
  const px = size === 'lg' ? '10px' : '7px';
  const py = size === 'lg' ? '5px' : '3px';
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '5px',
      fontSize: fs, fontWeight: 600, letterSpacing: '0.04em',
      textTransform: 'uppercase', fontFamily: 'var(--font-mono)',
      color: meta.color, background: meta.bg,
      border: `1px solid ${meta.color}40`,
      borderRadius: '4px', padding: `${py} ${px}`,
      whiteSpace: 'nowrap',
    }}>
      {isProcessing && (
        <span style={{
          width: '6px', height: '6px', borderRadius: '50%',
          background: meta.color, display: 'inline-block',
          animation: 'pulse-dot 1.2s ease-in-out infinite',
        }} />
      )}
      {!isProcessing && (
        <span style={{
          width: '5px', height: '5px', borderRadius: '50%',
          background: meta.color, display: 'inline-block',
        }} />
      )}
      {meta.label}
    </span>
  );
}

// ─── ConfidenceMeter ─────────────────────────────────────────────────────────
function ConfidenceMeter({ value = 0, label = 'Confidence' }) {
  const [displayed, setDisplayed] = useState(0);
  useEffect(() => {
    let start = null;
    const duration = 900;
    const animate = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setDisplayed(Math.round(ease * value));
      if (progress < 1) requestAnimationFrame(animate);
    };
    const raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, [value]);

  const color = value >= 80 ? 'oklch(0.46 0.19 145)'
    : value >= 55 ? 'oklch(0.75 0.14 60)'
    : 'oklch(0.65 0.20 25)';

  return (
    <div style={{ width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
        <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>{label}</span>
        <span style={{ fontSize: '14px', fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>{displayed}%</span>
      </div>
      <div style={{ height: '6px', background: 'var(--bg-elevated)', borderRadius: '3px', overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${displayed}%`, background: color,
          borderRadius: '3px', transition: 'width 0.05s linear',
          boxShadow: `0 0 8px ${color}80`,
        }} />
      </div>
    </div>
  );
}

// ─── PipelineStepper ─────────────────────────────────────────────────────────
function PipelineStepper({ steps, currentStep }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
      {steps.map((step, i) => {
        const done = i < currentStep;
        const active = i === currentStep;
        const waiting = i > currentStep;
        const color = done ? 'oklch(0.46 0.19 145)'
          : active ? 'oklch(0.46 0.19 145)'
          : 'var(--text-muted)';
        return (
          <div key={i} style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <div style={{
                width: '24px', height: '24px', borderRadius: '50%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                border: `2px solid ${color}`,
                background: done ? color : active ? `${color}20` : 'transparent',
                flexShrink: 0, fontSize: '11px', fontWeight: 700,
                color: done ? '#fff' : color,
                transition: 'all 0.3s ease',
                boxShadow: active ? `0 0 12px ${color}60` : 'none',
              }}>
                {done ? '✓' : i + 1}
              </div>
              {i < steps.length - 1 && (
                <div style={{
                  width: '2px', height: '28px',
                  background: done ? 'oklch(0.46 0.19 145)' : 'var(--border)',
                  transition: 'background 0.3s ease',
                }} />
              )}
            </div>
            <div style={{ paddingTop: '3px', paddingBottom: i < steps.length - 1 ? '28px' : '0' }}>
              <div style={{ fontSize: '13px', fontWeight: 600, color: waiting ? 'var(--text-muted)' : 'var(--text-primary)', transition: 'color 0.3s' }}>
                {step.label}
                {active && <span style={{ marginLeft: '8px', fontSize: '11px', color: 'oklch(0.46 0.19 145)', fontFamily: 'var(--font-mono)', animation: 'fade-in-out 1.4s ease infinite' }}>running…</span>}
              </div>
              {step.desc && (
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>{step.desc}</div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── FileUploadZone ───────────────────────────────────────────────────────────
function FileUploadZone({ accept, acceptLabel, onFile, file, error }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  };

  return (
    <div
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      style={{
        border: `2px dashed ${error ? 'oklch(0.65 0.20 25)' : dragging ? 'oklch(0.46 0.19 145)' : 'var(--border-strong)'}`,
        borderRadius: '8px', padding: '28px 20px', cursor: 'pointer',
        textAlign: 'center', transition: 'all 0.2s ease',
        background: dragging ? 'oklch(0.46 0.19 145 / 0.07)' : file ? 'oklch(0.46 0.19 145 / 0.08)' : 'var(--bg-elevated)',
      }}
    >
      <input ref={inputRef} type="file" accept={accept} style={{ display: 'none' }}
        onChange={e => e.target.files[0] && onFile(e.target.files[0])} />
      {file ? (
        <>
          <div style={{ fontSize: '28px', marginBottom: '8px' }}>📄</div>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>{file.name}</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {(file.size / 1024).toFixed(1)} KB · Click to replace
          </div>
        </>
      ) : (
        <>
          <div style={{ fontSize: '28px', marginBottom: '8px', opacity: 0.5 }}>⬆</div>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
            Drop file here or <span style={{ color: 'oklch(0.46 0.19 145)' }}>browse</span>
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>{acceptLabel}</div>
        </>
      )}
      {error && <div style={{ fontSize: '12px', color: 'oklch(0.65 0.20 25)', marginTop: '8px', fontWeight: 500 }}>{error}</div>}
    </div>
  );
}

// ─── PatientIdField ───────────────────────────────────────────────────────────
function PatientIdField({ value, onChange }) {
  return (
    <div>
      <label style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Patient ID <span style={{ color: 'var(--text-muted)', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>(optional)</span>
      </label>
      <input
        value={value} onChange={e => onChange(e.target.value)}
        placeholder="e.g. PT-20240125-001"
        style={{
          width: '100%', background: 'var(--bg-elevated)', border: '1px solid var(--border)',
          borderRadius: '6px', padding: '9px 12px', color: 'var(--text-primary)',
          fontSize: '13px', fontFamily: 'var(--font-mono)', outline: 'none',
          transition: 'border-color 0.2s', boxSizing: 'border-box',
        }}
        onFocus={e => e.target.style.borderColor = 'oklch(0.46 0.19 145)'}
        onBlur={e => e.target.style.borderColor = 'var(--border)'}
      />
    </div>
  );
}

// ─── MedicalDisclaimer ────────────────────────────────────────────────────────
function MedicalDisclaimer() {
  return (
    <div style={{
      background: 'oklch(0.75 0.14 60 / 0.08)', border: '1px solid oklch(0.75 0.14 60 / 0.3)',
      borderRadius: '6px', padding: '10px 14px', display: 'flex', gap: '10px', alignItems: 'flex-start',
    }}>
      <span style={{ fontSize: '14px', flexShrink: 0, marginTop: '1px' }}>⚠</span>
      <p style={{ fontSize: '11px', color: 'oklch(0.75 0.14 60)', margin: 0, lineHeight: 1.6 }}>
        <strong>Medical Disclaimer:</strong> VAIDYAA AI outputs are for informational and research purposes only. Results do not constitute medical advice, diagnosis, or treatment. Always consult a licensed medical professional before making clinical decisions.
      </p>
    </div>
  );
}

// ─── CitationList ─────────────────────────────────────────────────────────────
function CitationList({ citations = [] }) {
  if (!citations.length) return null;
  return (
    <div>
      <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '10px' }}>Sources & Citations</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {citations.map((c, i) => (
          <div key={i} style={{
            background: 'var(--bg-elevated)', borderRadius: '6px',
            padding: '10px 12px', border: '1px solid var(--border)',
          }}>
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>[{i+1}] {c.title}</div>
            {c.source && <div style={{ fontSize: '11px', color: 'oklch(0.46 0.19 145)', marginTop: '2px' }}>{c.source}</div>}
            {c.snippet && <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px', lineHeight: 1.5 }}>{c.snippet}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── ApiErrorAlert ────────────────────────────────────────────────────────────
function ApiErrorAlert({ message }) {
  if (!message) return null;
  return (
    <div style={{
      background: 'oklch(0.65 0.20 25 / 0.1)', border: '1px solid oklch(0.65 0.20 25 / 0.4)',
      borderRadius: '6px', padding: '12px 14px', display: 'flex', gap: '10px',
    }}>
      <span style={{ color: 'oklch(0.65 0.20 25)', fontSize: '14px', flexShrink: 0 }}>✕</span>
      <span style={{ fontSize: '13px', color: 'oklch(0.65 0.20 25)' }}>{message}</span>
    </div>
  );
}

// ─── UncertaintyBanner ────────────────────────────────────────────────────────
function UncertaintyBanner({ message }) {
  return (
    <div style={{
      background: 'oklch(0.75 0.14 60 / 0.1)', border: '1px solid oklch(0.75 0.14 60 / 0.35)',
      borderRadius: '6px', padding: '12px 14px', display: 'flex', gap: '10px', alignItems: 'center',
    }}>
      <span style={{ fontSize: '16px' }}>◬</span>
      <span style={{ fontSize: '13px', color: 'oklch(0.75 0.14 60)', fontWeight: 500 }}>{message}</span>
    </div>
  );
}

// ─── EmptyResultState ─────────────────────────────────────────────────────────
function EmptyResultState({ icon = '◫', title, subtitle }) {
  return (
    <div style={{ textAlign: 'center', padding: '48px 24px' }}>
      <div style={{ fontSize: '36px', opacity: 0.2, marginBottom: '12px' }}>{icon}</div>
      <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-secondary)' }}>{title}</div>
      {subtitle && <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>{subtitle}</div>}
    </div>
  );
}

// ─── ResultHeader ─────────────────────────────────────────────────────────────
function ResultHeader({ status, jobId, elapsed }) {
  const meta = STATUS_META[status] || STATUS_META.pending;
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{
          width: '40px', height: '40px', borderRadius: '8px',
          background: meta.bg, border: `1px solid ${meta.color}40`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '18px', color: meta.color,
        }}>
          {status === 'verified' ? '✓' : status === 'refuted' ? '✕' : status === 'uncertain' ? '◬' : status === 'failed' ? '!' : '…'}
        </div>
        <div>
          <StatusBadge status={status} size="lg" />
          {jobId && <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px', fontFamily: 'var(--font-mono)' }}>JOB {jobId}</div>}
        </div>
      </div>
      {elapsed && <div style={{ fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>⏱ {elapsed}s</div>}
    </div>
  );
}

// ─── SectionLabel ─────────────────────────────────────────────────────────────
function SectionLabel({ children }) {
  return (
    <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
      {children}
    </div>
  );
}

// ─── Card ─────────────────────────────────────────────────────────────────────
function Card({ children, style = {} }) {
  return (
    <div style={{
      background: 'var(--bg-surface)', border: '1px solid var(--border)',
      borderRadius: '10px', padding: '20px', ...style,
    }}>
      {children}
    </div>
  );
}

// ─── SegmentedControl ─────────────────────────────────────────────────────────
function SegmentedControl({ options, value, onChange }) {
  return (
    <div style={{
      display: 'inline-flex', background: 'var(--bg-elevated)',
      border: '1px solid var(--border)', borderRadius: '8px', padding: '3px', gap: '2px',
    }}>
      {options.map(opt => (
        <button key={opt.value} onClick={() => onChange(opt.value)} style={{
          padding: '7px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer',
          fontSize: '12px', fontWeight: 600, transition: 'all 0.18s ease',
          background: value === opt.value ? 'oklch(0.46 0.19 145)' : 'transparent',
          color: value === opt.value ? '#fff' : 'var(--text-secondary)',
        }}>
          {opt.label}
        </button>
      ))}
    </div>
  );
}

// ─── Button ──────────────────────────────────────────────────────────────────
function Button({ children, onClick, variant = 'primary', disabled, style = {} }) {
  const [hov, setHov] = useState(false);
  const bg = variant === 'primary'
    ? (disabled ? 'oklch(0.70 0.13 195 / 0.4)' : hov ? 'oklch(0.52 0.19 145)' : 'oklch(0.46 0.19 145)')
    : hov ? 'var(--bg-hover)' : 'var(--bg-elevated)';
  return (
    <button onClick={disabled ? undefined : onClick}
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{
        padding: '10px 20px', borderRadius: '7px', border: variant === 'ghost' ? '1px solid var(--border)' : 'none',
        background: bg, color: variant === 'primary' ? '#fff' : 'var(--text-secondary)',
        fontSize: '13px', fontWeight: 600, cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'all 0.18s ease', opacity: disabled ? 0.7 : 1, ...style,
      }}>
      {children}
    </button>
  );
}

// ─── EntityChip ───────────────────────────────────────────────────────────────
function EntityChip({ label, type }) {
  const colors = {
    condition:   'oklch(0.46 0.19 145)',
    medication:  'oklch(0.46 0.19 145)',
    lab_value:   'oklch(0.75 0.14 60)',
    finding:     'oklch(0.72 0.13 300)',
  };
  const c = colors[type] || colors.finding;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '4px',
      padding: '3px 9px', borderRadius: '20px', fontSize: '12px',
      background: `${c}18`, border: `1px solid ${c}40`, color: c, fontWeight: 500,
    }}>
      {label}
    </span>
  );
}

Object.assign(window, {
  StatusBadge, ConfidenceMeter, PipelineStepper,
  FileUploadZone, PatientIdField, MedicalDisclaimer,
  CitationList, ApiErrorAlert, UncertaintyBanner,
  EmptyResultState, ResultHeader, SectionLabel,
  Card, SegmentedControl, Button, EntityChip,
  STATUS_META,
});

}
