{

// report-analyzer.jsx — Report Analyzer page

const { useState, useRef } = React;
const API_BASE = '/api/v1';

const REPORT_STEPS = {
  lab:      [
    { label: 'OCR Extraction',       desc: 'Reading text from PDF/CSV' },
    { label: 'ClinicalBERT NER',     desc: 'Extracting lab values & conditions' },
    { label: 'XGBoost Scoring',      desc: 'Risk classification model' },
    { label: 'SHAP Attribution',     desc: 'Top contributing factors' },
    { label: 'LLM/RAG Synthesis',    desc: 'Clinical narrative generation' },
  ],
  clinical: [
    { label: 'OCR Extraction',       desc: 'Parsing clinical note structure' },
    { label: 'ClinicalBERT NER',     desc: 'Conditions, medications, procedures' },
    { label: 'XGBoost Scoring',      desc: 'Severity & risk assessment' },
    { label: 'SHAP Attribution',     desc: 'Feature importance ranking' },
    { label: 'LLM/RAG Synthesis',    desc: 'Summarization and recommendations' },
  ],
  discharge: [
    { label: 'OCR Extraction',       desc: 'Discharge summary parsing' },
    { label: 'ClinicalBERT NER',     desc: 'Diagnoses, discharge meds, follow-up' },
    { label: 'XGBoost Scoring',      desc: 'Readmission risk model' },
    { label: 'SHAP Attribution',     desc: 'Readmission risk factors' },
    { label: 'LLM/RAG Synthesis',    desc: 'Care gap identification' },
  ],
};

const MOCK_LAB_RESULT = {
  status: 'verified',
  riskScore: 68,
  riskLabel: 'Moderate-High',
  conditions: ['Type 2 Diabetes Mellitus', 'Dyslipidemia', 'Chronic Kidney Disease Stage 3'],
  medications: ['Metformin 1000mg BD', 'Atorvastatin 40mg OD', 'Lisinopril 10mg OD'],
  labValues: [
    { name: 'HbA1c', value: '8.4%', ref: '<7.0%', flag: 'HIGH' },
    { name: 'LDL-C', value: '3.8 mmol/L', ref: '<2.6 mmol/L', flag: 'HIGH' },
    { name: 'eGFR', value: '42 mL/min', ref: '>60 mL/min', flag: 'LOW' },
    { name: 'Serum Creatinine', value: '142 μmol/L', ref: '62–106 μmol/L', flag: 'HIGH' },
    { name: 'Fasting Glucose', value: '9.2 mmol/L', ref: '3.9–6.1 mmol/L', flag: 'HIGH' },
    { name: 'Hemoglobin', value: '11.8 g/dL', ref: '12.0–16.0 g/dL', flag: 'LOW' },
  ],
  anomalies: [
    'HbA1c significantly above target — review glycemic management',
    'eGFR decline suggests CKD progression — nephrology referral advised',
    'Mild normocytic anemia — rule out CKD-related erythropoietin deficiency',
  ],
  shap: [
    { factor: 'HbA1c elevation', score: 0.41 },
    { factor: 'eGFR decline', score: 0.33 },
    { factor: 'LDL above target', score: 0.15 },
    { factor: 'Anemia presence', score: 0.08 },
    { factor: 'Age factor', score: 0.03 },
  ],
  citations: [
    { title: 'KDIGO 2022 Clinical Practice Guideline for Diabetes in CKD', source: 'Kidney Int 102(5S):S1–S127', snippet: 'Target HbA1c 6.5–8.0% based on comorbidities and hypoglycemia risk.' },
    { title: 'ADA Standards of Medical Care in Diabetes 2024', source: 'Diabetes Care 47(Suppl 1)', snippet: 'Intensify glycemic management; consider GLP-1 RA or SGLT2i for cardio-renal benefit.' },
  ],
};

const FLAG_COLOR = { HIGH: 'oklch(0.65 0.20 25)', LOW: 'oklch(0.46 0.19 145)', NORMAL: 'oklch(0.46 0.19 145)' };

function LabTable({ rows }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {['Test', 'Result', 'Reference', 'Flag'].map(h => (
              <th key={h} style={{ padding: '9px 12px', textAlign: 'left', fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows?.map((r, i) => (
            <tr key={i} style={{ borderBottom: i < rows.length - 1 ? '1px solid var(--border)' : 'none' }}>
              <td style={{ padding: '9px 12px', fontWeight: 600, color: 'var(--text-primary)' }}>{r.name}</td>
              <td style={{ padding: '9px 12px', fontFamily: 'var(--font-mono)', color: FLAG_COLOR[r.flag] || 'var(--text-primary)', fontWeight: 700 }}>{r.value}</td>
              <td style={{ padding: '9px 12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>{r.ref}</td>
              <td style={{ padding: '9px 12px' }}>
                <span style={{
                  fontSize: '10px', fontWeight: 700, fontFamily: 'var(--font-mono)',
                  color: FLAG_COLOR[r.flag], background: `${FLAG_COLOR[r.flag]}18`,
                  padding: '2px 7px', borderRadius: '3px', border: `1px solid ${FLAG_COLOR[r.flag]}40`,
                }}>{r.flag}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ShapRow({ factor, score }) {
  const pct = Math.abs(score) * 100;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', width: '180px', flexShrink: 0 }}>{factor}</div>
      <div style={{ flex: 1, height: '7px', background: 'var(--bg-elevated)', borderRadius: '4px', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: 'oklch(0.46 0.19 145)', borderRadius: '4px', transition: 'width 0.9s ease' }} />
      </div>
      <div style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'oklch(0.46 0.19 145)', width: '38px', textAlign: 'right' }}>{score.toFixed(2)}</div>
    </div>
  );
}

function ReportAnalyzer() {
  const [reportType, setReportType] = useState('lab');
  const [file, setFile]             = useState(null);
  const [fileError, setFileError]   = useState('');
  const [patientId, setPatientId]   = useState('');
  const [phase, setPhase]           = useState('idle');
  const [step, setStep]             = useState(0);
  const [result, setResult]         = useState(null);
  const [elapsed, setElapsed]       = useState(0);
  const timerRef = useRef(null);
  const jobId = useRef('JOB-' + Math.floor(Math.random() * 900 + 100)).current;

  const ALLOWED_EXT = ['.pdf', '.csv'];
  const validateFile = (f) => {
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
    if (!ALLOWED_EXT.includes(ext)) { setFileError(`Invalid file type "${ext}". Accepted: .pdf, .csv`); return false; }
    if (f.size === 0) { setFileError('File is empty.'); return false; }
    if (f.size > 20 * 1024 * 1024) { setFileError('File exceeds 20 MB limit.'); return false; }
    setFileError(''); return true;
  };

  const handleFile = (f) => { if (validateFile(f)) setFile(f); };

  const handleSubmit = async () => {
    if (!file) { setFileError('Please upload a report file.'); return; }
    setPhase('running'); setStep(0); setResult(null); setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);

    const steps = REPORT_STEPS[reportType] || [];
    let s = 0;
    const advance = () => {
      s++; setStep(s);
      if (s < steps.length) setTimeout(advance, 1000 + Math.random() * 600);
    };
    setTimeout(advance, 1000);

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (patientId) formData.append('patient_id', patientId);

      const submitRes = await fetch(`${API_BASE || '/api/v1'}/analyze/report/${reportType}`, {
        method: 'POST',
        body: formData,
      });

      if (!submitRes.ok) {
        const errData = await submitRes.json().catch(() => ({}));
        throw new Error(errData.detail || `API returned ${submitRes.status}`);
      }

      const submitData = await submitRes.json();
      const reportId = submitData.id || submitData.task_id;
      console.log('[ReportAnalyzer] Submit response:', submitData);
      console.log('[ReportAnalyzer] Polling with reportId:', reportId);

      // Poll for result
      const poll = async () => {
        try {
          const pollRes = await fetch(`${API_BASE}/analyze/report/${reportId}`);
          const data = await pollRes.json();
          console.log('[ReportAnalyzer] Poll response:', data);
          
          const st = (data.status || '').toLowerCase();
          if (st === 'pending' || st === 'processing' || st === 'progress' || st === 'started') {
            setTimeout(poll, 2000);
            return;
          }

          // Done — map to display format
          clearInterval(timerRef.current);
          setStep(steps.length);
          
          const entities = data.extracted_entities || {};
          const risk     = data.risk_score || 0;
          
          setResult({
            status:      data.status || 'complete',
            riskScore:   risk,
            riskLabel:   risk > 80 ? 'High' : risk > 50 ? 'Moderate' : 'Low',
            conditions:  entities.conditions || [],
            medications: entities.medications || [],
            labValues:   Object.entries(entities.lab_values || {}).map(([k, v]) => ({
              name: k,
              value: v?.value ?? v ?? '--',
              ref: v?.ref || '--',
              flag: v?.flag || 'NORMAL'
            })),
            anomalies:   data.anomalies || [],
            shap:        (data.risk_factors || data.shap_values || []).map(f => ({ 
              factor: f?.feature || f?.factor || 'Unknown', 
              score: f?.shap ?? f?.score ?? 0 
            })),
            citations:   (data.sources || data.source_citations || []).map(s => ({
              title:  s.title || s.source_id || 'Evidence',
              source: s.source || s.url || '',
              snippet: s.snippet || s.excerpt || ''
            })),
          });
          setPhase('done');
        } catch (err) {
          console.error('Poll error:', err);
          setTimeout(poll, 3000);
        }
      };
      
      setTimeout(poll, 2000);
    } catch (err) {
      clearInterval(timerRef.current);
      setFileError(err.message);
      setPhase('idle');
    }
  };

  const reset = () => {
    clearInterval(timerRef.current);
    setPhase('idle'); setFile(null); setFileError(''); setStep(0); setResult(null); setElapsed(0);
  };

  const steps = REPORT_STEPS[reportType] || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h1 style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>Report Analyzer</h1>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '4px 0 0' }}>AI-powered analysis of lab panels, clinical notes, and discharge summaries</p>
      </div>

      {phase === 'idle' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <Card>
            <SectionLabel>Report Type</SectionLabel>
            <SegmentedControl
              options={[{ value: 'lab', label: 'Lab Panel' }, { value: 'clinical', label: 'Clinical Note' }, { value: 'discharge', label: 'Discharge Summary' }]}
              value={reportType} onChange={setReportType}
            />
          </Card>
          <Card>
            <SectionLabel>Upload Report</SectionLabel>
            <FileUploadZone accept=".pdf,.csv" acceptLabel="Accepts .pdf and .csv · Max 20 MB" onFile={handleFile} file={file} error={fileError} />
          </Card>
          <Card>
            <PatientIdField value={patientId} onChange={setPatientId} />
          </Card>
          <Button onClick={handleSubmit} disabled={!file}>▤ Analyze Report</Button>
          <MedicalDisclaimer />
        </div>
      )}

      {phase === 'running' && (
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          <Card style={{ flex: '1', minWidth: '240px' }}>
            <SectionLabel>Pipeline</SectionLabel>
            <PipelineStepper steps={steps} currentStep={step} />
          </Card>
          <Card style={{ flex: '2', minWidth: '260px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '14px' }}>
              <SectionLabel>Processing</SectionLabel>
              <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>⏱ {elapsed}s</span>
            </div>
            <StatusBadge status="processing" size="lg" />
            <div style={{ marginTop: '14px', fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{jobId}</div>
            <div style={{ marginTop: '12px', padding: '12px', background: 'var(--bg-elevated)', borderRadius: '7px', border: '1px solid var(--border)', fontSize: '13px', color: 'var(--text-secondary)' }}>
              <strong style={{ color: 'var(--text-primary)' }}>{file?.name}</strong>
              <div style={{ marginTop: '4px', fontSize: '12px', color: 'var(--text-muted)' }}>
                {reportType.charAt(0).toUpperCase() + reportType.slice(1)} · {(file?.size / 1024).toFixed(1)} KB
              </div>
            </div>
            <div style={{ marginTop: '14px' }}>
              <Button variant="ghost" onClick={reset} style={{ fontSize: '12px', padding: '7px 14px' }}>Cancel</Button>
            </div>
          </Card>
        </div>
      )}

      {phase === 'done' && result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <Card>
            <ResultHeader status={result.status} jobId={jobId} elapsed={elapsed} />
            <div style={{ marginTop: '16px', display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
              <div style={{ flex: 1, minWidth: '160px', padding: '14px', background: 'var(--bg-elevated)', borderRadius: '8px', border: '1px solid var(--border)' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '8px' }}>Risk Score</div>
                <div style={{ fontSize: '32px', fontWeight: 900, fontFamily: 'var(--font-mono)', color: 'oklch(0.75 0.14 60)' }}>{result.riskScore}</div>
                <div style={{ fontSize: '12px', color: 'oklch(0.75 0.14 60)', marginTop: '2px', fontWeight: 600 }}>{result.riskLabel}</div>
              </div>
              <div style={{ flex: 2, minWidth: '200px' }}>
                <ConfidenceMeter value={result.riskScore} label="Overall Risk Score" />
                <div style={{ marginTop: '14px' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '8px' }}>Active Conditions</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {result.conditions?.map((c, i) => <EntityChip key={i} label={c} type="condition" />)}
                  </div>
                  <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {result.medications?.map((m, i) => <EntityChip key={i} label={m} type="medication" />)}
                  </div>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <SectionLabel>Lab Values</SectionLabel>
            <LabTable rows={result.labValues} />
          </Card>

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            <Card style={{ flex: 1, minWidth: '240px' }}>
              <SectionLabel>Anomalies Detected</SectionLabel>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {result.anomalies?.map((a, i) => (
                  <div key={i} style={{ display: 'flex', gap: '10px', padding: '10px', background: 'oklch(0.65 0.20 25 / 0.07)', borderRadius: '6px', border: '1px solid oklch(0.65 0.20 25 / 0.25)' }}>
                    <span style={{ color: 'oklch(0.65 0.20 25)', fontSize: '13px', flexShrink: 0 }}>▲</span>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{a}</span>
                  </div>
                ))}
              </div>
            </Card>
            <Card style={{ flex: 1, minWidth: '240px' }}>
              <SectionLabel>SHAP Top Factors</SectionLabel>
              {result.shap?.map((s, i) => <ShapRow key={i} factor={s.factor} score={s.score} />)}
            </Card>
          </div>

          <Card>
            <CitationList citations={result.citations} />
          </Card>
          <MedicalDisclaimer />
          <Button variant="ghost" onClick={reset}>← New Report</Button>
        </div>
      )}
    </div>
  );
}

window.ReportAnalyzer = ReportAnalyzer;

}
