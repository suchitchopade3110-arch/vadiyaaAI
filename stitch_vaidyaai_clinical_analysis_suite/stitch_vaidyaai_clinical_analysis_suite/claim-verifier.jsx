{

// claim-verifier.jsx — Claim Verifier page

const { useState, useEffect, useRef } = React;

const CLAIM_STEPS = [
  { label: 'NLP Preprocessing',    desc: 'Tokenization, entity extraction' },
  { label: 'ClinicalBERT NER',     desc: 'Named entity recognition on medical terms' },
  { label: 'Evidence Retrieval',   desc: 'Searching PubMed & clinical databases' },
  { label: 'XGBoost Classification',desc: 'Scoring claim against evidence' },
  { label: 'SHAP Explanation',     desc: 'Computing feature attribution values' },
  { label: 'LLM Synthesis',        desc: 'Generating final verdict & reasoning' },
];

const MOCK_RESULT = {
  status: 'verified',
  confidence: 82,
  verdict_text: 'This claim is supported by multiple high-quality randomized controlled trials. Meta-analyses consistently show aspirin antiplatelet effect reduces major adverse cardiovascular events (MACE) in secondary prevention settings.',
  entities: [
    { label: 'Myocardial Infarction', type: 'condition' },
    { label: 'Aspirin 75–100mg', type: 'medication' },
    { label: 'Risk Reduction 22–25%', type: 'lab_value' },
    { label: 'Platelet Aggregation', type: 'finding' },
  ],
  shap: [
    { factor: 'RCT evidence count', score: 0.38 },
    { factor: 'Effect size consistency', score: 0.27 },
    { factor: 'Population specificity', score: 0.19 },
    { factor: 'Dose alignment', score: 0.10 },
    { factor: 'Recency of evidence', score: 0.06 },
  ],
  citations: [
    { title: 'Antithrombotic Trialists Collaboration (2009)', source: 'Lancet 373(9678):1849–1860', snippet: 'Aspirin reduced major vascular events by 22% in secondary prevention (RR 0.78, p<0.0001).' },
    { title: 'ARRIVE Trial — Gaziano et al. (2018)', source: 'Lancet 392(10152):1036–1046', snippet: 'Primary prevention findings; notable risk/benefit variation by baseline risk.' },
    { title: 'ACC/AHA Guideline on Primary Prevention (2019)', source: 'JACC 74(10):1376–1414', snippet: 'Class IIb recommendation for aspirin in selected high-risk patients aged 40–70.' },
  ],
};

const MOCK_REFUTED = {
  status: 'refuted',
  confidence: 91,
  verdict_text: 'This claim is not supported by peer-reviewed evidence. There is no credible clinical evidence that high-dose Vitamin C cures cancer. Cochrane reviews find no survival benefit. Some studies show potential adjunctive palliative effects but these are not curative.',
  entities: [
    { label: 'Cancer (malignant neoplasm)', type: 'condition' },
    { label: 'Vitamin C (ascorbic acid)', type: 'medication' },
    { label: 'High-dose IV infusion', type: 'finding' },
  ],
  shap: [
    { factor: 'Absence of RCT evidence', score: 0.52 },
    { factor: 'Cochrane meta-analysis', score: 0.28 },
    { factor: 'Mechanistic plausibility', score: -0.08 },
    { factor: 'Observational signal', score: 0.06 },
    { factor: 'Regulatory status', score: 0.14 },
  ],
  citations: [
    { title: 'Cochrane Review: Vitamin C for preventing and treating cancer (2021)', source: 'Cochrane Database Syst Rev', snippet: 'No reliable evidence that Vitamin C supplementation reduces cancer incidence or mortality.' },
    { title: 'Padayatty et al. (2006)', source: 'CMAJ 174(7):937–942', snippet: 'Case reports of tumour regression are anecdotal; controlled trials not supportive.' },
  ],
};

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
  const [jobId]                 = useState('JOB-' + Math.floor(Math.random()*900+100));
  const [elapsed, setElapsed]   = useState(0);
  const timerRef = useRef(null);

  const charCount = claim.length;
  const charValid = charCount >= 10 && charCount <= 5000;

  const handleSubmit = () => {
    if (!charValid) { setError('Claim must be between 10 and 5000 characters.'); return; }
    setError('');
    setPhase('running');
    setStep(0);
    setResult(null);
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);
    let s = 0;
    const advance = () => {
      s += 1;
      setStep(s);
      if (s < CLAIM_STEPS.length) setTimeout(advance, 900 + Math.random() * 600);
      else {
        clearInterval(timerRef.current);
        setTimeout(() => {
          const lc = claim.toLowerCase();
          const res = (lc.includes('vitamin c') || lc.includes('cure')) ? MOCK_REFUTED : MOCK_RESULT;
          setResult(res);
          setPhase('done');
        }, 400);
      }
    };
    setTimeout(advance, 900);
  };

  const reset = () => {
    clearInterval(timerRef.current);
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
              {jobId} · {CLAIM_STEPS[Math.min(step, CLAIM_STEPS.length-1)].label}
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
            <Card style={{ flex: 1, minWidth: '260px' }}>
              <SectionLabel>Extracted Entities</SectionLabel>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {result.entities.map((e, i) => <EntityChip key={i} label={e.label} type={e.type} />)}
              </div>
              <div style={{ marginTop: '14px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {['condition','medication','lab_value','finding'].map(t => {
                  const colors = { condition:'oklch(0.46 0.19 145)', medication:'oklch(0.46 0.19 145)', lab_value:'oklch(0.75 0.14 60)', finding:'oklch(0.72 0.13 300)' };
                  return <span key={t} style={{ fontSize: '10px', color: colors[t], display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: colors[t], display: 'inline-block' }} />
                    {t.replace('_',' ')}
                  </span>;
                })}
              </div>
            </Card>
            <Card style={{ flex: 1, minWidth: '260px' }}>
              <SectionLabel>SHAP Feature Attribution</SectionLabel>
              {result.shap.map((s, i) => <ShapBar key={i} factor={s.factor} score={s.score} />)}
            </Card>
          </div>

          <Card>
            <CitationList citations={result.citations} />
          </Card>

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
