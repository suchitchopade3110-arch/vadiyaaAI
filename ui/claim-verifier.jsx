{

// claim-verifier.jsx — Claim Verifier page (wired to real API)

const { useState, useEffect, useRef } = React;

const API_BASE = `${window.location.origin}/api/v1`;

const CLAIM_STEPS = [
  { label: 'NLP Preprocessing',    desc: 'Tokenization, entity extraction' },
  { label: 'ClinicalBERT NER',     desc: 'Named entity recognition on medical terms' },
  { label: 'Evidence Retrieval',   desc: 'Searching PubMed & clinical databases' },
  { label: 'XGBoost Classification',desc: 'Scoring claim against evidence' },
  { label: 'SHAP Explanation',     desc: 'Computing feature attribution values' },
  { label: 'LLM Synthesis',        desc: 'Generating final verdict & reasoning' },
];

function ShapBar({ factor, score }) {
  const isPos = score >= 0;
  const color = isPos ? 'oklch(0.46 0.19 145)' : 'oklch(0.65 0.20 25)';
  const pct = Math.abs(score) * 100;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
      <div style={{ flex: 1, fontSize: '12px', color: 'var(--text-secondary)', textAlign: 'right', minWidth: '160px' }}>{factor}</div>
      <div style={{ width: '140px', height: '8px', background: 'var(--bg-elevated)', borderRadius: '4px', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: '4px', transition: 'width 0.8s ease' }} />
      </div>
      <div style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color, width: '42px' }}>
        {isPos ? '+' : ''}{score.toFixed(2)}
      </div>
    </div>
  );
}

function ClaimVerifier() {
  const [claim, setClaim]       = useState('');
  const [patientId, setPatientId] = useState('');
  const [phase, setPhase]       = useState('idle'); // idle | running | done
  const [step, setStep]         = useState(0);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState('');
  const [jobId, setJobId]       = useState('');
  const [elapsed, setElapsed]   = useState(0);
  const timerRef = useRef(null);
  const pollRef = useRef(null);

  const charCount = claim.length;
  const charValid = charCount >= 10 && charCount <= 5000;

  const handleSubmit = async () => {
    if (!charValid) { setError('Claim must be between 10 and 5000 characters.'); return; }
    setError('');
    setPhase('running');
    setStep(0);
    setResult(null);
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);

    // Animate pipeline steps while waiting
    let s = 0;
    const advanceStep = () => {
      s += 1;
      if (s < CLAIM_STEPS.length) {
        setStep(s);
        setTimeout(advanceStep, 1200 + Math.random() * 800);
      }
    };
    setTimeout(advanceStep, 1000);

    try {
      // Submit claim to real API
      const body = { claim_text: claim };
      if (patientId) body.patient_id = patientId;

      const submitRes = await fetch(`${API_BASE}/verify/claim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!submitRes.ok) {
        const errData = await submitRes.json().catch(() => ({}));
        throw new Error(errData.detail || `API returned ${submitRes.status}`);
      }

      const submitData = await submitRes.json();
      const claimId = submitData.job_id || submitData.id || submitData.task_id || submitData.claim_id;
      if (!claimId) throw new Error('No job ID returned from API');
      setJobId(claimId.slice(0, 8));

      // Poll for result
      const poll = async () => {
        try {
          const pollRes = await fetch(`${API_BASE}/verify/claim/${claimId}`);
          const data = await pollRes.json();
          
          if (data.status === 'PROCESSING' || data.status === 'PENDING' || data.status === 'processing' || data.status === 'pending') {
            pollRef.current = setTimeout(poll, 2000);
            return;
          }

          // Done — map to display format
          clearInterval(timerRef.current);
          setStep(CLAIM_STEPS.length);
          
          const analysisResult = data.analysis_result || {};
          const status = (data.status || '').toLowerCase();
          const displayStatus = status === 'verified' ? 'verified' : status === 'refuted' || status === 'contradicted' ? 'refuted' : 'uncertain';

          setResult({
            status: displayStatus,
            confidence: data.confidence || analysisResult.confidence?.score || 75,
            verdict_text: analysisResult.explanation || analysisResult.verdict || 'Analysis complete.',
            entities: (analysisResult.entities || []).map(e => ({ label: e.label || e, type: e.type || 'finding' })),
            shap: Object.entries(analysisResult.shap_values || {}).map(([k, v]) => ({ factor: k, score: v })).slice(0, 5),
            citations: (data.source_citations || data.sources || []).map(c => ({
              title: c.title || c.source_id || 'Source',
              source: c.url || c.source || '',
              snippet: c.excerpt || c.snippet || '',
            })),
          });
          setPhase('done');
        } catch (err) {
          console.error('Poll error:', err);
          pollRef.current = setTimeout(poll, 3000);
        }
      };
      
      setTimeout(poll, 2000);

    } catch (err) {
      clearInterval(timerRef.current);
      setError(err.message);
      setPhase('idle');
    }
  };

  const reset = () => {
    clearInterval(timerRef.current);
    clearTimeout(pollRef.current);
    setPhase('idle'); setStep(0); setResult(null); setError(''); setElapsed(0);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h1 style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>Claim Verifier</h1>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '4px 0 0' }}>Evidence-based fact-checking against clinical literature</p>
      </div>

      {phase === 'idle' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <Card>
            <SectionLabel>Medical Claim</SectionLabel>
            <textarea
              value={claim} onChange={e => setClaim(e.target.value)}
              placeholder="Enter a medical claim to verify, e.g. 'Aspirin reduces the risk of myocardial infarction by 25% in secondary prevention patients.'"
              style={{
                width: '100%', minHeight: '120px', background: 'var(--bg-elevated)',
                border: `1px solid ${error ? 'oklch(0.65 0.20 25)' : charCount > 0 && !charValid ? 'oklch(0.75 0.14 60)' : 'var(--border)'}`,
                borderRadius: '7px', padding: '12px', color: 'var(--text-primary)',
                fontSize: '14px', lineHeight: 1.6, resize: 'vertical',
                outline: 'none', fontFamily: 'inherit', boxSizing: 'border-box',
                transition: 'border-color 0.2s',
              }}
              onFocus={e => e.target.style.borderColor = 'oklch(0.46 0.19 145)'}
              onBlur={e => e.target.style.borderColor = 'var(--border)'}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '11px', color: charCount > 5000 ? 'oklch(0.65 0.20 25)' : 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              <span>{charCount < 10 && charCount > 0 ? `${10 - charCount} more chars needed` : ''}</span>
              <span>{charCount} / 5000</span>
            </div>
          </Card>
          <Card>
            <PatientIdField value={patientId} onChange={setPatientId} />
          </Card>
          <ApiErrorAlert message={error} />
          <div style={{ display: 'flex', gap: '10px' }}>
            <Button onClick={handleSubmit} disabled={!claim.trim()}>
              ◎ Analyze Claim
            </Button>
            <Button variant="ghost" onClick={() => { setClaim('Aspirin reduces the risk of myocardial infarction by approximately 25% in secondary prevention patients.'); }}>
              Try example
            </Button>
          </div>
          <MedicalDisclaimer />
        </div>
      )}

      {phase === 'running' && (
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          <Card style={{ flex: '1', minWidth: '260px' }}>
            <SectionLabel>Analysis Pipeline</SectionLabel>
            <PipelineStepper steps={CLAIM_STEPS} currentStep={step} />
          </Card>
          <Card style={{ flex: '2', minWidth: '280px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
              <SectionLabel>Job Status</SectionLabel>
              <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>⏱ {elapsed}s</span>
            </div>
            <StatusBadge status="processing" size="lg" />
            <div style={{ marginTop: '16px', fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              {jobId || '...'} · {CLAIM_STEPS[Math.min(step, CLAIM_STEPS.length-1)].label}
            </div>
            <div style={{ marginTop: '20px', padding: '14px', background: 'var(--bg-elevated)', borderRadius: '7px', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>Claim submitted:</div>
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6, fontStyle: 'italic' }}>"{claim.slice(0, 140)}{claim.length > 140 ? '…' : ''}"</div>
            </div>
            <div style={{ marginTop: '16px' }}>
              <Button variant="ghost" onClick={reset} style={{ fontSize: '12px', padding: '7px 14px' }}>Cancel Job</Button>
            </div>
          </Card>
        </div>
      )}

      {phase === 'done' && result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <Card>
            <ResultHeader status={result.status} jobId={jobId} elapsed={elapsed} />
            <div style={{ marginTop: '16px', fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.7, padding: '14px', background: 'var(--bg-elevated)', borderRadius: '7px', border: '1px solid var(--border)' }}>
              {result.verdict_text}
            </div>
            <div style={{ marginTop: '16px' }}>
              <ConfidenceMeter value={result.confidence} label="Verdict Confidence" />
            </div>
          </Card>

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            {result.entities.length > 0 && (
              <Card style={{ flex: 1, minWidth: '260px' }}>
                <SectionLabel>Extracted Entities</SectionLabel>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {result.entities?.map((e, i) => <EntityChip key={i} label={e.label} type={e.type} />)}
                </div>
              </Card>
            )}
            {result.shap.length > 0 && (
              <Card style={{ flex: 1, minWidth: '260px' }}>
                <SectionLabel>SHAP Feature Attribution</SectionLabel>
                {result.shap?.map((s, i) => <ShapBar key={i} factor={s.factor} score={s.score} />)}
              </Card>
            )}
          </div>

          {result.citations.length > 0 && (
            <Card>
              <CitationList citations={result.citations} />
            </Card>
          )}

          {result.status === 'uncertain' && (
            <UncertaintyBanner message="Confidence below threshold. Results should be interpreted with clinical judgment." />
          )}
          <MedicalDisclaimer />
          <div style={{ display: 'flex', gap: '10px' }}>
            <Button onClick={reset} variant="ghost">← New Claim</Button>
          </div>
        </div>
      )}
    </div>
  );
}

window.ClaimVerifier = ClaimVerifier;

}
