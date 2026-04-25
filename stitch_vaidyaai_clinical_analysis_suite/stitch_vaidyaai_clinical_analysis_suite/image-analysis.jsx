{

// image-analysis.jsx — Image Analysis page

const { useState, useRef, useEffect } = React;

const IMAGE_STEPS = [
  { label: 'Normalization',      desc: 'Resize, denoise, DICOM windowing' },
  { label: 'Segmentation',       desc: 'ROI detection & organ masking' },
  { label: 'Classification',     desc: 'ResNet / EfficientNet inference' },
  { label: 'GradCAM',            desc: 'Gradient-weighted class activation map' },
  { label: 'LLM Explanation',    desc: 'Radiological narrative generation' },
];

const MOCK_XRAY_RESULT = {
  status: 'verified',
  type: 'xray',
  findings: [
    { label: 'Cardiomegaly', confidence: 87, severity: 'Moderate' },
    { label: 'Bilateral pleural effusion', confidence: 74, severity: 'Mild' },
    { label: 'Pulmonary vascular congestion', confidence: 81, severity: 'Moderate' },
  ],
  impression: 'Findings consistent with decompensated heart failure. Moderate cardiomegaly with cardiothoracic ratio ~0.58. Bilateral blunting of costophrenic angles indicating pleural effusion. Increased perihilar markings suggest pulmonary venous hypertension. Clinical correlation and echocardiography recommended.',
  roi: [
    { region: 'Cardiac silhouette', finding: 'Enlarged, CTR 0.58', x: '28%', y: '30%', w: '44%', h: '42%' },
    { region: 'Right base', finding: 'Blunting CP angle', x: '65%', y: '68%', w: '22%', h: '18%' },
    { region: 'Left base', finding: 'Blunting CP angle', x: '12%', y: '68%', w: '22%', h: '18%' },
  ],
  confidence: 81,
  uncertainty: false,
  citations: [
    { title: 'Cardiomegaly on Chest X-ray: Diagnostic Accuracy', source: 'Radiology 289(3):700–710', snippet: 'CTR >0.5 on PA film has sensitivity 88%, specificity 77% for cardiac enlargement.' },
    { title: 'AHA/ACC Guideline for Heart Failure (2022)', source: 'JACC 79(17):e263–e421', snippet: 'Imaging findings of congestion include pleural effusion and cephalization of pulmonary vasculature.' },
  ],
};

const MOCK_MRI_RESULT = {
  status: 'uncertain',
  type: 'mri',
  findings: [
    { label: 'T2 hyperintense lesion', confidence: 63, severity: 'Uncertain' },
    { label: 'Periventricular white matter change', confidence: 71, severity: 'Mild' },
  ],
  impression: 'Multiple T2/FLAIR hyperintense foci in periventricular and subcortical white matter. Differential includes demyelinating disease (MS), small vessel ischemic changes, or migraine-related changes. Enhancement study and CSF analysis recommended to narrow differential.',
  roi: [],
  confidence: 63,
  uncertainty: true,
  citations: [
    { title: 'McDonald Criteria for MS Diagnosis (2017 Revision)', source: 'Lancet Neurol 17(2):162–173', snippet: 'Dissemination in space criteria requires ≥1 T2 lesion in ≥2 of 4 CNS regions.' },
  ],
};

// Placeholder GradCAM canvas
function GradCAMViewer({ type }) {
  const canvasRef = useRef();
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    // Draw placeholder anatomy
    ctx.fillStyle = '#e8f5ee';
    ctx.fillRect(0, 0, w, h);
    // Subtle chest silhouette
    ctx.strokeStyle = 'rgba(50,120,80,0.15)';
    ctx.lineWidth = 1.5;
    if (type === 'xray') {
      // Ribcage outline
      for (let i = 0; i < 7; i++) {
        ctx.beginPath(); ctx.ellipse(w/2, h*0.35 + i*22, w*0.35 - i*4, 18, 0, 0, Math.PI); ctx.stroke();
      }
      // Spine
      ctx.beginPath(); ctx.moveTo(w/2, h*0.1); ctx.lineTo(w/2, h*0.85); ctx.stroke();
    }
    // GradCAM heat overlay — cardiac region
    const spots = type === 'xray'
      ? [{ x: w*0.5, y: h*0.42, r: 70, a: 0.55, color: [255, 80, 40] }, { x: w*0.72, y: h*0.72, r: 35, a: 0.4, color: [255, 180, 40] }, { x: w*0.28, y: h*0.72, r: 30, a: 0.35, color: [255, 180, 40] }]
      : [{ x: w*0.45, y: h*0.38, r: 45, a: 0.5, color: [80, 140, 255] }, { x: w*0.6, y: h*0.44, r: 30, a: 0.4, color: [80, 200, 255] }];
    spots.forEach(({ x, y, r, a, color }) => {
      const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, `rgba(${color.join(',')},${a})`);
      grad.addColorStop(1, `rgba(${color.join(',')},0)`);
      ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    });
    // Overlay label
    ctx.fillStyle = 'rgba(30,90,50,0.4)';
    ctx.font = '10px monospace';
    ctx.fillText('GradCAM Activation', 10, h - 12);
  }, [type]);
  return (
    <canvas ref={canvasRef} width={320} height={260}
      style={{ width: '100%', maxWidth: '320px', borderRadius: '8px', border: '1px solid var(--border)', display: 'block' }} />
  );
}

// Image preview with placeholder
function ImagePreview({ file }) {
  const [src, setSrc] = useState(null);
  useEffect(() => {
    if (!file) return;
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (['.jpg', '.jpeg', '.png'].includes(ext)) {
      const url = URL.createObjectURL(file);
      setSrc(url);
      return () => URL.revokeObjectURL(url);
    }
    setSrc(null);
  }, [file]);

  if (!file) return null;
  return (
    <div style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--bg-elevated)', maxWidth: '280px' }}>
      {src ? (
        <img src={src} alt="preview" style={{ width: '100%', display: 'block', maxHeight: '220px', objectFit: 'contain' }} />
      ) : (
        <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: '28px', marginBottom: '8px' }}>⬡</div>
          <div style={{ fontSize: '12px' }}>DICOM preview not available</div>
          <div style={{ fontSize: '11px', marginTop: '4px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{file.name}</div>
        </div>
      )}
    </div>
  );
}

function FindingRow({ finding }) {
  const sevColor = { Moderate: 'oklch(0.75 0.14 60)', Mild: 'oklch(0.46 0.19 145)', Severe: 'oklch(0.65 0.20 25)', Uncertain: 'oklch(0.55 0.01 260)' };
  const c = sevColor[finding.severity] || 'var(--text-secondary)';
  return (
    <div style={{ padding: '12px', background: 'var(--bg-elevated)', borderRadius: '7px', border: '1px solid var(--border)', marginBottom: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>{finding.label}</span>
        <span style={{ fontSize: '11px', fontWeight: 700, color: c, background: `${c}18`, padding: '2px 8px', borderRadius: '4px', border: `1px solid ${c}40` }}>{finding.severity}</span>
      </div>
      <ConfidenceMeter value={finding.confidence} label="Detection Confidence" />
    </div>
  );
}

function ImageAnalysis() {
  const [analysisType, setAnalysisType] = useState('xray');
  const [file, setFile]                 = useState(null);
  const [fileError, setFileError]       = useState('');
  const [patientId, setPatientId]       = useState('');
  const [phase, setPhase]               = useState('idle');
  const [step, setStep]                 = useState(0);
  const [result, setResult]             = useState(null);
  const [elapsed, setElapsed]           = useState(0);
  const timerRef = useRef(null);
  const jobId = useRef('JOB-' + Math.floor(Math.random() * 900 + 100)).current;

  const ALLOWED = ['.dcm', '.jpg', '.jpeg', '.png'];

  const handleFile = (f) => {
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
    if (!ALLOWED.includes(ext)) { setFileError(`Invalid type "${ext}". Accepted: DICOM, JPEG, PNG`); return; }
    if (f.size === 0) { setFileError('File is empty.'); return; }
    // Simulate DICOM magic byte check
    if (ext === '.dcm' && f.name.includes('invalid')) { setFileError('Invalid DICOM: magic bytes not found at offset 128.'); return; }
    setFileError(''); setFile(f);
  };

  const handleSubmit = () => {
    if (!file) { setFileError('Please upload an image file.'); return; }
    setPhase('running'); setStep(0); setResult(null); setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);
    let s = 0;
    const advance = () => {
      s++; setStep(s);
      if (s < IMAGE_STEPS.length) setTimeout(advance, 1100 + Math.random() * 700);
      else {
        clearInterval(timerRef.current);
        setTimeout(() => {
          setResult(analysisType === 'mri' ? MOCK_MRI_RESULT : MOCK_XRAY_RESULT);
          setPhase('done');
        }, 500);
      }
    };
    setTimeout(advance, 1000);
  };

  const reset = () => {
    clearInterval(timerRef.current);
    setPhase('idle'); setFile(null); setFileError(''); setStep(0); setResult(null); setElapsed(0);
  };

  const typeOptions = [
    { value: 'xray', label: 'X-Ray' }, { value: 'ct', label: 'CT' },
    { value: 'mri', label: 'MRI' }, { value: 'skin', label: 'Skin' },
    { value: 'pathology', label: 'Pathology' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h1 style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>Image Analysis</h1>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '4px 0 0' }}>AI-powered diagnostic imaging: X-ray, CT, MRI, dermatology, pathology</p>
      </div>

      {phase === 'idle' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <Card>
            <SectionLabel>Imaging Modality</SectionLabel>
            <SegmentedControl options={typeOptions} value={analysisType} onChange={setAnalysisType} />
          </Card>
          <Card>
            <SectionLabel>Upload Image</SectionLabel>
            <FileUploadZone accept=".dcm,.jpg,.jpeg,.png" acceptLabel="Accepts DICOM (.dcm), JPEG, PNG · Max 50 MB" onFile={handleFile} file={file} error={fileError} />
            {file && <div style={{ marginTop: '12px' }}><ImagePreview file={file} /></div>}
          </Card>
          <Card>
            <PatientIdField value={patientId} onChange={setPatientId} />
          </Card>
          <Button onClick={handleSubmit} disabled={!file}>⬡ Analyze Image</Button>
          <MedicalDisclaimer />
        </div>
      )}

      {phase === 'running' && (
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          <Card style={{ flex: '1', minWidth: '240px' }}>
            <SectionLabel>Pipeline</SectionLabel>
            <PipelineStepper steps={IMAGE_STEPS} currentStep={step} />
          </Card>
          <Card style={{ flex: '2', minWidth: '260px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '14px' }}>
              <SectionLabel>Processing</SectionLabel>
              <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>⏱ {elapsed}s</span>
            </div>
            <StatusBadge status="processing" size="lg" />
            <div style={{ marginTop: '14px', fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{jobId} · {analysisType.toUpperCase()}</div>
            <div style={{ marginTop: '12px' }}>
              <ImagePreview file={file} />
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
            {result.uncertainty && (
              <div style={{ marginTop: '12px' }}>
                <UncertaintyBanner message="Low confidence — findings require radiologist review before clinical use." />
              </div>
            )}
            <div style={{ marginTop: '16px', fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.7, padding: '14px', background: 'var(--bg-elevated)', borderRadius: '7px', border: '1px solid var(--border)' }}>
              <strong style={{ display: 'block', marginBottom: '6px', color: 'var(--text-primary)' }}>Radiological Impression</strong>
              {result.impression}
            </div>
            <div style={{ marginTop: '16px' }}>
              <ConfidenceMeter value={result.confidence} label="Overall Confidence" />
            </div>
          </Card>

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            <div style={{ flex: '1', minWidth: '260px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <Card>
                <SectionLabel>Classification Findings</SectionLabel>
                {result.findings.map((f, i) => <FindingRow key={i} finding={f} />)}
              </Card>
              {result.roi.length > 0 && (
                <Card>
                  <SectionLabel>Region of Interest (ROI)</SectionLabel>
                  {result.roi.map((r, i) => (
                    <div key={i} style={{ marginBottom: '8px', padding: '10px', background: 'var(--bg-elevated)', borderRadius: '6px', border: '1px solid var(--border)' }}>
                      <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>{r.region}</div>
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>{r.finding}</div>
                    </div>
                  ))}
                </Card>
              )}
            </div>
            <Card style={{ flex: '1', minWidth: '260px' }}>
              <SectionLabel>GradCAM Activation Map</SectionLabel>
              <GradCAMViewer type={result.type} />
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px', lineHeight: 1.5 }}>
                Heatmap shows regions of highest model attention. Red/orange = strong activation. Overlay is illustrative — clinical verification required.
              </p>
            </Card>
          </div>

          <Card>
            <CitationList citations={result.citations} />
          </Card>
          <MedicalDisclaimer />
          <Button variant="ghost" onClick={reset}>← New Image</Button>
        </div>
      )}
    </div>
  );
}

window.ImageAnalysis = ImageAnalysis;

}
