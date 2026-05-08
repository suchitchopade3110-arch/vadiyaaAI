{

// report-analyzer.jsx — Report Analyzer (Phase 3)
// UX-17: Brief/Full toggle REMOVED - full explanation always shown
// UX-14: Patient history diff view
// UX-16: WebSocket with polling fallback
// UX-20: Drug interaction chips
// UX-19: Bias audit dashboard

const { useState, useRef } = React;
const API_BASE = `${window.location.origin}/api/v1`;

const REPORT_STEPS = {
  lab: [
    { label: 'OCR Extraction', desc: 'PaddleOCR reading report' },
    { label: 'Groq NER', desc: 'Extracting values and conditions' },
    { label: 'Ensemble Model', desc: 'XGBoost + anomaly + clinical consensus' },
    { label: 'Differential Dx', desc: 'Differential diagnosis chain' },
    { label: 'LLM/RAG Synthesis', desc: 'Clinical narrative generation' },
  ],
  clinical: [
    { label: 'OCR Extraction', desc: 'Parsing clinical note' },
    { label: 'Groq NER', desc: 'Conditions, medications, procedures' },
    { label: 'Ensemble Model', desc: 'Severity and risk assessment' },
    { label: 'Differential Dx', desc: 'Differential diagnosis chain' },
    { label: 'LLM/RAG Synthesis', desc: 'Summarization and recommendations' },
  ],
  discharge: [
    { label: 'OCR Extraction', desc: 'Discharge summary parsing' },
    { label: 'Groq NER', desc: 'Diagnoses, medications, follow-up' },
    { label: 'Ensemble Model', desc: 'Readmission risk model' },
    { label: 'Differential Dx', desc: 'Risk factor identification' },
    { label: 'LLM/RAG Synthesis', desc: 'Care gap identification' },
  ],
};

const getFlagColor = (flag) => {
  const f = (flag || '').toUpperCase();
  if (f.includes('CRITICAL') || f.includes('HIGH') || f === 'H') return 'oklch(0.65 0.20 25)';
  if (f.includes('LOW') || f.includes('ABNORMAL') || f === 'L') return 'oklch(0.65 0.20 25)';
  return 'oklch(0.46 0.19 145)';
};

function MiniLabel({ children, style = {} }) {
  return (
    <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px', ...style }}>
      {children}
    </div>
  );
}

function ExplanationBlock({ brief, full }) {
  if (!brief && !full) return null;
  const text = full || brief;
  return (
    <div>
      <MiniLabel>Plain Language Summary</MiniLabel>
      <div style={{ fontSize: '14px', color: 'var(--text-primary)', lineHeight: 1.7, whiteSpace: 'pre-wrap', padding: '4px 0' }}>
        {text}
      </div>
    </div>
  );
}

function DrugInteractionChips({ medications = [] }) {
  const interactions = {
    'metformin + lisinopril': { severity: 'low', note: 'Monitor renal function' },
    'atorvastatin + amlodipine': { severity: 'low', note: 'Possible statin level increase' },
    'warfarin + aspirin': { severity: 'high', note: 'Increased bleeding risk' },
    'metformin + contrast media': { severity: 'high', note: 'Hold metformin before contrast' },
    'lisinopril + potassium': { severity: 'medium', note: 'Risk of hyperkalemia' },
    'digoxin + amiodarone': { severity: 'high', note: 'Digoxin toxicity risk' },
    'ssri + tramadol': { severity: 'high', note: 'Serotonin syndrome risk' },
    'warfarin + nsaid': { severity: 'high', note: 'Major bleeding risk' },
    'metformin + alcohol': { severity: 'medium', note: 'Lactic acidosis risk' },
    'statin + fibrate': { severity: 'medium', note: 'Myopathy risk' },
  };
  const meds = medications.map((item) => String(item?.name || item || '').toLowerCase());
  const detected = Object.entries(interactions)
    .filter(([pair]) => pair.split(' + ').every((drug) => meds.some((med) => med.includes(drug))))
    .map(([pair, info]) => ({ pair, ...info }));

  if (!detected.length) return null;
  const colors = { high: 'oklch(0.65 0.20 25)', medium: 'oklch(0.75 0.14 60)', low: 'oklch(0.46 0.19 145)' };
  return (
    <div style={{ marginTop: '12px' }}>
      <MiniLabel>Drug Interactions Detected</MiniLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {detected.map((item, index) => {
          const color = colors[item.severity] || colors.low;
          return (
            <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 12px', borderRadius: '6px', background: `${color}10`, border: `1px solid ${color}40` }}>
              <span style={{ fontSize: '11px', fontWeight: 800, color, background: `${color}20`, padding: '2px 7px', borderRadius: '3px', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>{item.severity}</span>
              <span style={{ fontSize: '12px', color: 'var(--text-secondary)', flex: 1 }}>
                <strong style={{ color: 'var(--text-primary)' }}>{item.pair}</strong> - {item.note}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DifferentialDxBlock({ ddx }) {
  const [open, setOpen] = useState(false);
  if (!ddx || ddx.verdict === 'insufficient_evidence') return null;
  const urgencyColor = { emergency: 'oklch(0.60 0.22 25)', urgent: 'oklch(0.65 0.20 25)', follow_up: 'oklch(0.75 0.14 60)', routine: 'oklch(0.46 0.19 145)' };
  const items = ddx.differentials || [];
  return (
    <Card style={{ borderLeft: '4px solid oklch(0.46 0.19 145)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }} onClick={() => setOpen((value) => !value)}>
        <MiniLabel style={{ marginBottom: 0 }}>Differential Diagnosis</MiniLabel>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{open ? 'Hide' : 'Show'} {items.length} diagnoses</span>
      </div>
      {open && (
        <div style={{ marginTop: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {items.map((item, index) => {
            const color = urgencyColor[item.urgency] || urgencyColor.routine;
            return (
              <div key={index} style={{ padding: '12px', background: 'var(--bg-elevated)', borderRadius: '8px', border: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', flexWrap: 'wrap' }}>
                  <div>
                    <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginRight: '8px' }}>#{item.rank || index + 1}</span>
                    <strong style={{ fontSize: '14px', color: 'var(--text-primary)' }}>{item.diagnosis}</strong>
                    {item.icd10 && <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: '8px', fontFamily: 'var(--font-mono)' }}>{item.icd10}</span>}
                  </div>
                  <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                    <span style={{ fontSize: '11px', fontWeight: 800, color, background: `${color}18`, padding: '2px 8px', borderRadius: '3px', textTransform: 'uppercase' }}>{item.urgency || 'routine'}</span>
                    <span style={{ fontSize: '13px', fontWeight: 800, color: 'oklch(0.46 0.19 145)', fontFamily: 'var(--font-mono)' }}>{Math.round((item.confidence || 0) * 100)}%</span>
                  </div>
                </div>
                {!!item.supporting_evidence?.length && <div style={{ marginTop: '8px', fontSize: '12px', color: 'var(--text-muted)' }}>{item.supporting_evidence.join(' · ')}</div>}
                {!!item.recommended_tests?.length && <div style={{ marginTop: '6px', fontSize: '11px', color: 'oklch(0.75 0.14 60)' }}>Suggested: {item.recommended_tests.join(', ')}</div>}
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}

function PatientHistoryDiff({ comparison, history }) {
  const [open, setOpen] = useState(false);
  if (!comparison && (!history || !history.length)) return null;

  const labDeltas = comparison?.lab_deltas || [...(comparison?.worsened_values || []), ...(comparison?.improved_values || [])];
  const riskDelta = comparison?.risk_delta;
  const riskDirection = comparison?.risk_direction || comparison?.risk_trend?.trend || 'unchanged';
  const directionColor = (value) => value === 'up' || value === 'worsened' ? 'oklch(0.65 0.20 25)' : value === 'down' || value === 'improved' ? 'oklch(0.46 0.19 145)' : 'var(--text-muted)';
  const directionIcon = (value) => value === 'up' || value === 'worsened' ? '▲' : value === 'down' || value === 'improved' ? '▼' : '→';

  return (
    <Card>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }} onClick={() => setOpen((value) => !value)}>
        <MiniLabel style={{ marginBottom: 0 }}>Patient History Comparison</MiniLabel>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{open ? 'Hide' : 'Show'} {history?.length || 0} prior reports</span>
      </div>
      {open && comparison && (
        <div style={{ marginTop: '14px' }}>
          <div style={{ display: 'flex', gap: '12px', marginBottom: '14px', flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: '140px', padding: '10px', background: 'var(--bg-elevated)', borderRadius: '7px', border: '1px solid var(--border)', textAlign: 'center' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Risk Change</div>
              <div style={{ fontSize: '22px', fontWeight: 900, color: directionColor(riskDirection), fontFamily: 'var(--font-mono)' }}>{riskDelta == null ? '--' : `${riskDelta > 0 ? '+' : ''}${riskDelta}`}</div>
            </div>
            <div style={{ flex: 1, minWidth: '140px', padding: '10px', background: 'var(--bg-elevated)', borderRadius: '7px', border: '1px solid var(--border)', textAlign: 'center' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Trend</div>
              <div style={{ fontSize: '22px', color: directionColor(riskDirection) }}>{directionIcon(riskDirection)}</div>
              <div style={{ fontSize: '11px', fontWeight: 700, color: directionColor(riskDirection), textTransform: 'capitalize' }}>{riskDirection}</div>
            </div>
          </div>
          {!!labDeltas?.length && (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    {['Test', 'Previous', 'Current', 'Change', 'Trend'].map((header) => (
                      <th key={header} style={{ padding: '6px 10px', textAlign: 'left', fontSize: '10px', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {labDeltas.slice(0, 8).map((item, index) => {
                    const trend = item.direction || item.trend || 'unchanged';
                    return (
                      <tr key={index} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '7px 10px', fontWeight: 700, color: 'var(--text-primary)' }}>{item.test}</td>
                        <td style={{ padding: '7px 10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{item.previous ?? item.previous_value ?? '--'} {item.unit}</td>
                        <td style={{ padding: '7px 10px', fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 700 }}>{item.current ?? item.current_value ?? '--'} {item.unit}</td>
                        <td style={{ padding: '7px 10px', fontFamily: 'var(--font-mono)', color: directionColor(trend) }}>{item.delta != null ? `${item.delta > 0 ? '+' : ''}${item.delta}` : '--'}</td>
                        <td style={{ padding: '7px 10px', color: directionColor(trend), fontWeight: 800 }}>{directionIcon(trend)} {trend}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          {comparison.summary && <div style={{ marginTop: '10px', fontSize: '12px', color: 'var(--text-muted)' }}>{comparison.summary}</div>}
        </div>
      )}
      {open && !comparison && !!history?.length && <div style={{ marginTop: '12px', fontSize: '12px', color: 'var(--text-muted)' }}>{history.length} prior report(s) found.</div>}
    </Card>
  );
}

function BiasAuditDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/admin/bias-audit`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail || 'Unable to load bias audit');
      setData(payload);
    } catch (error) {
      setData({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  const AccBar = ({ label, value, overall }) => {
    const pct = Math.round((value || 0) * 100);
    const gap = Math.abs(pct - Math.round((overall || 0) * 100));
    const flagged = gap > 10;
    const color = flagged ? 'oklch(0.65 0.20 25)' : 'oklch(0.46 0.19 145)';
    return (
      <div style={{ marginBottom: '8px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{label}</span>
          <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color, fontWeight: 800 }}>{pct}% {flagged ? 'FLAG' : 'OK'}</span>
        </div>
        <div style={{ height: '5px', background: 'var(--bg-elevated)', borderRadius: '3px', overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: '3px' }} />
        </div>
      </div>
    );
  };

  return (
    <Card style={{ border: '1px solid oklch(0.75 0.14 60 / 0.4)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }} onClick={() => { setOpen((value) => !value); if (!open && !data) load(); }}>
        <MiniLabel style={{ marginBottom: 0 }}>Bias Audit Dashboard</MiniLabel>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{ fontSize: '10px', background: 'oklch(0.75 0.14 60 / 0.15)', color: 'oklch(0.75 0.14 60)', padding: '2px 7px', borderRadius: '3px', fontWeight: 800 }}>ADMIN</span>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{open ? 'Hide' : 'Show'}</span>
        </div>
      </div>
      {open && (
        <div style={{ marginTop: '14px' }}>
          {loading && <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Loading audit data...</div>}
          {data?.error && <div style={{ fontSize: '12px', color: 'oklch(0.65 0.20 25)' }}>Error: {data.error}</div>}
          {data && !data.error && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Overall accuracy: <strong style={{ color: 'var(--text-primary)' }}>{Math.round((data.overall_metrics?.accuracy || 0) * 100)}%</strong> · {data.total_records} records · {data.audit_version}
              </div>
              {['gender_metrics', 'age_group_metrics', 'region_metrics'].map((groupKey) => (
                <div key={groupKey}>
                  <MiniLabel>{groupKey.replace('_metrics', '').replace('_', ' ')}</MiniLabel>
                  {Object.entries(data[groupKey] || {}).map(([group, metrics]) => (
                    <AccBar key={group} label={group} value={metrics.accuracy} overall={data.overall_metrics?.accuracy} />
                  ))}
                </div>
              ))}
              <div style={{ fontSize: '11px', color: 'oklch(0.75 0.14 60)', padding: '8px', background: 'oklch(0.75 0.14 60 / 0.08)', borderRadius: '6px' }}>
                FLAG = accuracy gap greater than 10% from overall and requires domain lead review.
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function LabTable({ rows, anomalyMap = {} }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['Test', 'Result', 'Reference', 'Flag'].map((header) => (
              <th key={header} style={{ padding: '9px 12px', textAlign: 'left', fontSize: '11px', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows?.map((row, index) => {
            const testKey = (row.test || row.name || '').toLowerCase().replace(/_/g, ' ').trim();
            const anomaly = anomalyMap[testKey];
            const reference = anomaly?.reference || row.reference || row.ref || '--';
            const flag = anomaly?.severity || anomaly?.flag || row.flag || 'NORMAL';
            const isAbnormal = anomaly?.status === 'ABNORMAL' || /HIGH|LOW|CRITICAL|ABNORMAL/i.test(flag);
            return (
              <tr key={index} style={{ borderBottom: index < rows.length - 1 ? '1px solid var(--border)' : 'none' }}>
                <td style={{ padding: '9px 12px', fontWeight: 700, color: 'var(--text-primary)' }}>{row.test || row.name}</td>
                <td style={{ padding: '9px 12px', fontFamily: 'var(--font-mono)', color: isAbnormal ? getFlagColor(flag) : 'var(--text-primary)', fontWeight: isAbnormal ? 800 : 600 }}>{row.result ?? row.value ?? '--'}</td>
                <td style={{ padding: '9px 12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>{reference}</td>
                <td style={{ padding: '9px 12px' }}>
                  <span style={{ fontSize: '10px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: getFlagColor(flag), background: `${getFlagColor(flag)}18`, padding: '2px 7px', borderRadius: '3px', border: `1px solid ${getFlagColor(flag)}40` }}>{flag}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ShapRow({ factor, score }) {
  const safeScore = Number(score || 0);
  const pct = Math.min(100, Math.abs(safeScore) * 100);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', width: '180px', flexShrink: 0 }}>{factor}</div>
      <div style={{ flex: 1, height: '7px', background: 'var(--bg-elevated)', borderRadius: '4px', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: 'oklch(0.46 0.19 145)', borderRadius: '4px', transition: 'width 0.9s ease' }} />
      </div>
      <div style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'oklch(0.46 0.19 145)', width: '38px', textAlign: 'right' }}>{safeScore.toFixed(2)}</div>
    </div>
  );
}

function normalizeShap(payload) {
  const raw = payload.shap_top_factors || payload.risk_factors || payload.shap_values || [];
  if (Array.isArray(raw)) {
    return raw.map((item) => ({
      factor: item?.feature || item?.factor || item?.condition || String(item || 'Unknown'),
      score: item?.shap ?? item?.score ?? item?.value ?? 0,
    }));
  }
  return Object.entries(raw || {}).map(([factor, score]) => ({ factor, score }));
}

function processPayload(payload, reportType) {
  const entities = payload.extracted_entities || {};
  const risk = payload.risk_score || 0;
  const rawLabValues = entities.lab_values || payload.lab_values || {};
  const labRows = Array.isArray(rawLabValues)
    ? rawLabValues.map((item) => ({
        name: item.field || item.test || item.name,
        test: item.test || item.field || item.name,
        value: item.result ?? item.value ?? '--',
        result: item.result ?? item.value ?? '--',
        ref: item.reference,
        reference: item.reference,
        flag: item.flag || 'NORMAL',
      }))
    : Object.entries(rawLabValues || {}).map(([key, value]) => ({
        name: key,
        test: key,
        value: value?.value ?? value ?? '--',
        result: value?.value ?? value ?? '--',
        ref: value?.ref || '--',
        reference: value?.reference || value?.ref || '--',
        flag: value?.flag || 'NORMAL',
      }));

  return {
    status: payload.status || 'complete',
    riskScore: risk > 1 ? Math.round(risk) : Math.round(risk * 100),
    riskLabel: payload.risk_level || payload.risk_label || (risk > 80 ? 'High' : risk > 50 ? 'Moderate' : 'Low'),
    conditions: entities.conditions || [],
    medications: entities.medications || [],
    labValues: labRows,
    anomalies: payload.anomalies || [],
    shap: normalizeShap(payload),
    citations: (payload.sources || payload.source_citations || []).map((item) => ({
      title: item.title || item.source_id || item.pattern || 'Evidence',
      source: item.source || item.url || '',
      snippet: item.snippet || item.excerpt || item.text || '',
    })),
    explanation: payload.explanation_full || payload.rag_explanation ||
                 payload.explanation_brief || payload.plain_language_summary ||
                 payload.explanation || '',
    differential: payload.differential_diagnosis || null,
    ensemble: payload.ensemble_details || null,
    history_comparison: payload.history_comparison || null,
    patient_history: payload.patient_history || [],
    qr_token: payload.qr_token || null,
    qr_available: payload.qr_available || false,
    reportType,
  };
}

function ReportAnalyzer() {
  const [reportType, setReportType] = useState('lab');
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState('');
  const [patientId, setPatientId] = useState('');
  const [phase, setPhase] = useState('idle');
  const [step, setStep] = useState(0);
  const [stepLabel, setStepLabel] = useState('');
  const [result, setResult] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [serverJobId, setServerJobId] = useState('');
  const timerRef = useRef(null);
  const wsRef = useRef(null);
  const pollRef = useRef(null);
  const jobId = useRef('JOB-' + Math.floor(Math.random() * 900 + 100)).current;

  const validateFile = (item) => {
    const ext = item.name.slice(item.name.lastIndexOf('.')).toLowerCase();
    if (!['.pdf', '.csv'].includes(ext)) { setFileError(`Invalid type "${ext}". Accepted: .pdf, .csv`); return false; }
    if (item.size === 0) { setFileError('File is empty.'); return false; }
    if (item.size > 20 * 1024 * 1024) { setFileError('File exceeds 20 MB limit.'); return false; }
    setFileError('');
    return true;
  };
  const handleFile = (item) => { if (validateFile(item)) setFile(item); };

  const finishResult = (payload) => {
    clearInterval(timerRef.current);
    clearTimeout(pollRef.current);
    setStep((REPORT_STEPS[reportType] || []).length);
    setResult(processPayload(payload, reportType));
    setPhase('done');
  };

  const fallbackPoll = (reportId) => {
    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE}/analyze/report/${reportId}`);
        const payload = await response.json();
        const status = (payload.status || '').toLowerCase();
        if (['pending', 'processing', 'progress', 'started'].includes(status)) {
          pollRef.current = setTimeout(poll, 2000);
          return;
        }
        finishResult(payload.result || payload);
      } catch (error) {
        pollRef.current = setTimeout(poll, 3000);
      }
    };
    pollRef.current = setTimeout(poll, 1500);
  };

  const connectWebSocket = (reportId) => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/${reportId}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const pct = payload.progress || 0;
        const steps = REPORT_STEPS[reportType] || [];
        setStep(Math.min(steps.length, Math.ceil((pct / 100) * steps.length)));
        if (payload.step) setStepLabel(payload.step);
        if (payload.status === 'SUCCESS' && payload.result) {
          ws.close();
          finishResult(payload.result);
        }
        if (payload.status === 'FAILURE' || payload.status === 'ERROR') {
          clearInterval(timerRef.current);
          ws.close();
          setFileError(payload.error || 'Job failed');
          setPhase('idle');
        }
      } catch (error) {
        console.error('WebSocket parse error:', error);
      }
    };
    ws.onerror = () => {
      console.warn('WebSocket failed; falling back to polling');
      try { ws.close(); } catch (error) {}
      fallbackPoll(reportId);
    };
  };

  const handleSubmit = async () => {
    if (!file) { setFileError('Please upload a report file.'); return; }
    setPhase('running');
    setStep(0);
    setResult(null);
    setElapsed(0);
    setStepLabel('');
    clearInterval(timerRef.current);
    timerRef.current = setInterval(() => setElapsed((value) => value + 1), 1000);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('explanation_mode', 'full');
      if (patientId) formData.append('patient_id', patientId);
      const response = await fetch(`${API_BASE}/analyze/report/${reportType}`, { method: 'POST', body: formData });
      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload.detail || `API ${response.status}`);
      }
      const payload = await response.json();
      const reportId = payload.job_id || payload.id || payload.task_id;
      setServerJobId(reportId);
      connectWebSocket(reportId);
    } catch (error) {
      clearInterval(timerRef.current);
      setFileError(error.message);
      setPhase('idle');
    }
  };

  const reset = () => {
    clearInterval(timerRef.current);
    clearTimeout(pollRef.current);
    if (wsRef.current) {
      try { wsRef.current.close(); } catch (error) {}
      wsRef.current = null;
    }
    setPhase('idle');
    setFile(null);
    setFileError('');
    setStep(0);
    setStepLabel('');
    setResult(null);
    setElapsed(0);
    setServerJobId('');
  };

  const steps = REPORT_STEPS[reportType] || [];
  const anomalyMap = {};
  result?.anomalies?.forEach((item) => {
    const key = (item.test || item.field || '').toLowerCase().replace(/_/g, ' ').trim();
    if (key) anomalyMap[key] = item;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h1 style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>Report Analyzer</h1>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '4px 0 0' }}>AI-powered analysis of lab panels, urinalysis, clinical notes, and discharge summaries</p>
      </div>

      {phase === 'idle' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <Card><SectionLabel>Report Type</SectionLabel><SegmentedControl options={[{ value: 'lab', label: 'Lab Panel' }, { value: 'clinical', label: 'Clinical Note' }, { value: 'discharge', label: 'Discharge Summary' }]} value={reportType} onChange={setReportType} /></Card>
          <Card><SectionLabel>Upload Report</SectionLabel><FileUploadZone accept=".pdf,.csv" acceptLabel="Accepts .pdf and .csv · Max 20 MB" onFile={handleFile} file={file} error={fileError} /></Card>
          <Card><PatientIdField value={patientId} onChange={setPatientId} /></Card>
          <Button onClick={handleSubmit} disabled={!file}>▤ Analyze Report</Button>
          <MedicalDisclaimer />
        </div>
      )}

      {phase === 'running' && (
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          <Card style={{ flex: '1', minWidth: '240px' }}><SectionLabel>Pipeline</SectionLabel><PipelineStepper steps={steps} currentStep={step} /></Card>
          <Card style={{ flex: '2', minWidth: '260px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '14px' }}><SectionLabel>Processing</SectionLabel><span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>⏱ {elapsed}s</span></div>
            <StatusBadge status="processing" size="lg" />
            {stepLabel && <div style={{ marginTop: '8px', fontSize: '12px', color: 'oklch(0.46 0.19 145)', fontFamily: 'var(--font-mono)' }}>→ {stepLabel}</div>}
            <div style={{ marginTop: '14px', fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{serverJobId || 'Initializing...'}</div>
            <div style={{ marginTop: '12px', padding: '12px', background: 'var(--bg-elevated)', borderRadius: '7px', border: '1px solid var(--border)', fontSize: '13px' }}>
              <strong style={{ color: 'var(--text-primary)' }}>{file?.name}</strong>
              <div style={{ marginTop: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>{reportType} · {(file?.size / 1024).toFixed(1)} KB</div>
            </div>
            <div style={{ marginTop: '14px' }}><Button variant="ghost" onClick={reset} style={{ fontSize: '12px', padding: '7px 14px' }}>Cancel</Button></div>
          </Card>
        </div>
      )}

      {phase === 'done' && result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', gap: '16px', alignItems: 'stretch', flexWrap: 'wrap' }}>
            <Card style={{ flex: 1, minWidth: '300px' }}>
              <ResultHeader status={result.status} jobId={jobId} elapsed={elapsed} />
              <div style={{ marginTop: '16px', display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
                <div style={{ flex: 1, minWidth: '160px', padding: '14px', background: 'var(--bg-elevated)', borderRadius: '8px', border: '1px solid var(--border)' }}>
                  <MiniLabel>Risk Score</MiniLabel>
                  <div style={{ fontSize: '32px', fontWeight: 900, fontFamily: 'var(--font-mono)', color: 'oklch(0.75 0.14 60)' }}>{result.riskScore}</div>
                  <div style={{ fontSize: '12px', color: 'oklch(0.75 0.14 60)', marginTop: '2px', fontWeight: 700 }}>{result.riskLabel}</div>
                  {result.ensemble?.model_scores && (
                    <div style={{ marginTop: '10px', fontSize: '10px', color: 'var(--text-muted)' }}>
                      {Object.entries(result.ensemble.model_scores).filter(([, value]) => value != null).map(([model, value]) => <div key={model}>▸ {model}: {Math.round(value)}</div>)}
                      <div style={{ marginTop: '4px', fontWeight: 800, color: result.ensemble.model_agreement === 'high' ? 'oklch(0.46 0.19 145)' : 'oklch(0.75 0.14 60)' }}>Agreement: {result.ensemble.model_agreement}</div>
                    </div>
                  )}
                </div>
                <div style={{ flex: 2, minWidth: '200px' }}>
                  <ConfidenceMeter value={result.riskScore} label="Overall Risk Score" />
                  <div style={{ marginTop: '14px' }}>
                    <MiniLabel>Active Conditions</MiniLabel>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>{result.conditions?.map((item, index) => <EntityChip key={index} label={item} type="condition" />)}</div>
                    <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>{result.medications?.map((item, index) => <EntityChip key={index} label={item?.name || item} type="medication" />)}</div>
                    <DrugInteractionChips medications={result.medications || []} />
                  </div>
                </div>
              </div>
            </Card>
            {window.ReportQRWidget && <ReportQRWidget reportId={serverJobId} patientId={patientId} />}
          </div>

          <Card><SectionLabel>Lab Values</SectionLabel><LabTable rows={result.labValues} anomalyMap={anomalyMap} /></Card>
          <PatientHistoryDiff comparison={result.history_comparison} history={result.patient_history} />
          <DifferentialDxBlock ddx={result.differential} />

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            <Card style={{ flex: 1, minWidth: '240px' }}>
              <SectionLabel>Anomalies Detected</SectionLabel>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {result.anomalies?.map((item, index) => (
                  <div key={index} style={{ display: 'flex', gap: '10px', padding: '10px', background: 'oklch(0.65 0.20 25 / 0.07)', borderRadius: '6px', border: '1px solid oklch(0.65 0.20 25 / 0.25)' }}>
                    <span style={{ color: 'oklch(0.65 0.20 25)', fontSize: '13px', flexShrink: 0 }}>▲</span>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                      <strong style={{ color: 'oklch(0.65 0.20 25)' }}>{item.test || item.field}</strong>: {item.value} {item.unit}
                      <span style={{ color: 'var(--text-muted)' }}> (ref: {item.reference})</span> - {item.severity}
                      {item.clinical_meaning && <p style={{ margin: '6px 0 0', color: 'var(--text-muted)', fontSize: '11px' }}>{item.clinical_meaning}</p>}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
            <Card style={{ flex: 1, minWidth: '240px' }}>
              <SectionLabel>SHAP Top Factors</SectionLabel>
              {result.shap?.length > 0 ? result.shap.map((item, index) => <ShapRow key={index} factor={item.factor} score={item.score} />) : <p style={{ color: 'var(--text-muted)', fontSize: '13px', fontStyle: 'italic', marginTop: '10px' }}>SHAP not applicable for this panel type.</p>}
            </Card>
          </div>

          {result.explanation && (
            <Card style={{ borderLeft: '4px solid oklch(0.46 0.19 145)', background: 'oklch(0.46 0.19 145 / 0.03)' }}>
              {result.riskScore > 80 && <div style={{ padding: '8px 12px', background: 'oklch(0.65 0.20 25 / 0.1)', color: 'oklch(0.65 0.20 25)', borderRadius: '6px', marginBottom: '12px', fontSize: '13px', fontWeight: 800, border: '1px solid oklch(0.65 0.20 25 / 0.3)' }}>HIGH URGENCY: Please review these findings with a medical professional as soon as possible.</div>}
              <ExplanationBlock brief={result.explanation} full={result.explanation} />
            </Card>
          )}

          <Card><CitationList citations={result.citations} /></Card>
          <BiasAuditDashboard />
          <MedicalDisclaimer />
          <Button variant="ghost" onClick={reset}>← New Report</Button>
        </div>
      )}
    </div>
  );
}

window.ReportAnalyzer = ReportAnalyzer;

}
