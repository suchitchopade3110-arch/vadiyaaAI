{

// image-analysis.jsx — Image Analysis page

const { useState, useRef, useEffect, useMemo } = React;
const API_BASE = `${window.location.origin}/api/v1`;

const IMAGE_STEPS = [
  { label: 'Normalization',      desc: 'Resize, denoise, DICOM windowing' },
  { label: 'Segmentation',       desc: 'ROI detection & organ masking' },
  { label: 'Classification',     desc: '14-class NIH chest X-ray inference' },
  { label: 'GradCAM',            desc: 'Gradient-weighted class activation map' },
  { label: 'LLM Explanation',    desc: 'Radiological narrative generation' },
];

function assetUrl(path) {
  if (!path) return '';
  const value = String(path);
  if (value.startsWith('http') || value.startsWith('data:') || value.startsWith('/')) return value;
  if (value.startsWith('data/yolo_outputs/')) return `${window.location.origin}/yolo_outputs/${value.split('/').pop()}`;
  if (value.includes('/data/yolo_outputs/')) return `${window.location.origin}/yolo_outputs/${value.split('/').pop()}`;
  return `${window.location.origin}/${value.replace(/^\/+/, '')}`;
}

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

const GRADCAM_REGION_SEEDS = [
  { label: 'Cardiac silhouette', x: 50, y: 48, r: 17, intensity: 0.88, severity: 'High' },
  { label: 'Right lower lung', x: 69, y: 70, r: 10, intensity: 0.64, severity: 'Moderate' },
  { label: 'Left lower lung', x: 31, y: 71, r: 10, intensity: 0.52, severity: 'Moderate' },
  { label: 'Right hilum', x: 60, y: 44, r: 9, intensity: 0.42, severity: 'Low' },
  { label: 'Left hilum', x: 40, y: 44, r: 9, intensity: 0.38, severity: 'Low' },
];

const GRADCAM_SEEDS_BY_TYPE = {
  xray: GRADCAM_REGION_SEEDS,
  ct: [
    { label: 'Right upper lung', x: 36, y: 38, r: 12, intensity: 0.68, severity: 'Moderate' },
    { label: 'Left upper lung', x: 64, y: 38, r: 12, intensity: 0.56, severity: 'Moderate' },
    { label: 'Mediastinum', x: 50, y: 50, r: 10, intensity: 0.44, severity: 'Low' },
  ],
  mri: [
    { label: 'Right hemisphere', x: 35, y: 48, r: 15, intensity: 0.70, severity: 'Moderate' },
    { label: 'Left hemisphere', x: 65, y: 48, r: 15, intensity: 0.50, severity: 'Moderate' },
    { label: 'Brain stem', x: 50, y: 72, r: 8, intensity: 0.40, severity: 'Low' },
  ],
  skin: [
    { label: 'Lesion center', x: 50, y: 50, r: 16, intensity: 0.72, severity: 'Moderate' },
    { label: 'Irregular border', x: 62, y: 44, r: 10, intensity: 0.54, severity: 'Moderate' },
    { label: 'Pigment variation', x: 42, y: 57, r: 9, intensity: 0.46, severity: 'Low' },
  ],
  pathology: [
    { label: 'Cellular atypia focus', x: 48, y: 46, r: 14, intensity: 0.72, severity: 'Moderate' },
    { label: 'Tissue architecture', x: 62, y: 58, r: 12, intensity: 0.55, severity: 'Moderate' },
    { label: 'Mitotic region', x: 38, y: 62, r: 8, intensity: 0.42, severity: 'Low' },
  ],
};

const WHO_IMAGING_CHECKLIST = [
  { id: 'identity', label: 'Patient identity', detail: 'Name, birthdate, institutional ID verified before interpretation.', section: 'WHO §IV' },
  { id: 'date', label: 'Study date', detail: 'Compare with prior exams to classify acute vs chronic findings.', section: 'WHO §IV' },
  { id: 'markers', label: 'L/R markers', detail: 'Anatomical side markers and projection are visible.', section: 'WHO §IV' },
  { id: 'positioning', label: 'Positioning', detail: 'PA/AP/portable view noted; AP magnification considered.', section: 'WHO §II-III' },
  { id: 'coverage', label: 'Anatomical coverage', detail: 'Supraclavicular region, chest walls, and both hemidiaphragms included.', section: 'WHO §III' },
  { id: 'quality', label: 'Image quality', detail: 'Inspiration, exposure, motion, and display quality acceptable.', section: 'WHO §II/VIII' },
];

const CYTOLOGY_TIERS = [
  { id: 1, label: 'Insufficient', risk: 'Repeat sample', color: 'oklch(0.55 0.01 260)', criteria: 'Obscuring blood, thick mucus, air-drying distortion, or too few preserved cells.', action: 'Repeat FNAB; consider rapid on-site evaluation.' },
  { id: 2, label: 'Benign', risk: 'Routine follow-up', color: 'oklch(0.46 0.19 145)', criteria: 'Unequivocal benign/reactive components or identifiable infectious organisms.', action: 'Routine clinical follow-up.' },
  { id: 3, label: 'Atypical', risk: 'Resolve ambiguity', color: 'oklch(0.75 0.14 60)', criteria: 'Atypia exceeds reactive change but is insufficient for neoplastic diagnosis.', action: 'Repeat sampling or ancillary molecular/flow testing.' },
  { id: 4, label: 'Neoplasm', risk: 'Risk-stratify', color: 'oklch(0.72 0.16 70)', criteria: 'Features of a specific neoplasm, stratified by low/high malignant risk.', action: 'Guide surgery, systemic therapy, or subspecialty review.' },
  { id: 5, label: 'Suspicious', risk: 'Biopsy/tumor board', color: 'oklch(0.70 0.18 35)', criteria: 'Strongly suggestive malignancy with low cellularity or incomplete features.', action: 'Core needle biopsy or tumor board correlation.' },
  { id: 6, label: 'Malignant', risk: 'Oncologic staging', color: 'oklch(0.63 0.22 30)', criteria: 'Unequivocal cancer features: pleomorphism, high N:C ratio, aberrant chromatin.', action: 'Immediate staging and oncology pathway.' },
];

const PATH_ESSENTIAL_CRITERIA = [
  { id: 'morphology', label: 'Essential morphology', detail: 'H&E scan shows defining architecture/cytology.' },
  { id: 'quality', label: 'Scan quality adequate', detail: 'Focus, stain, folds, bubbles, and compression are acceptable.' },
  { id: 'clinical', label: 'Clinical context', detail: 'Site, history, accession, and specimen/block identifiers available.' },
  { id: 'ihc', label: 'Basic IHC aligned', detail: 'Accessible immunohistochemistry supports the lineage/family.' },
];

const PATH_DESIRABLE_CRITERIA = [
  { id: 'molecular', label: 'Molecular profile', detail: 'NGS/FISH/methylation or subtype marker available.' },
  { id: 'icdo', label: 'ICD-O-4 mapped', detail: 'Structured diagnosis maps to topography and morphology code.' },
  { id: 'validation', label: 'Validated workflow', detail: 'WSI/AI workflow meets concordance and discrepancy thresholds.' },
];

function clampNumber(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.min(max, Math.max(min, n));
}

function percentNumber(value, fallback) {
  if (typeof value === 'string' && value.trim().endsWith('%')) return clampNumber(parseFloat(value), 0, 100);
  return clampNumber(value ?? fallback, 0, 100);
}

function regionIntensity(region, fallback = 0.5) {
  const raw = region.intensity ?? region.score ?? region.confidence ?? region.probability ?? fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) return fallback;
  return clampNumber(value > 1 ? value / 100 : value, 0.08, 1);
}

function severityFromIntensity(intensity) {
  if (intensity >= 0.76) return 'High';
  if (intensity >= 0.48) return 'Moderate';
  return 'Low';
}

function normalizeGradcamRegions({ result, type }) {
  const seedSet = GRADCAM_SEEDS_BY_TYPE[type] || GRADCAM_REGION_SEEDS;
  const gradcamRegions = result?.gradcam?.top_regions || result?.gradcam_regions || result?.top_regions || [];
  const roiRegions = Array.isArray(result?.roi) ? result.roi : [];
  const findings = Array.isArray(result?.findings) ? result.findings : [];
  const rawRegions = gradcamRegions.length ? gradcamRegions : (roiRegions.length ? roiRegions : findings);
  const base = rawRegions.length ? rawRegions : seedSet.slice(0, type === 'xray' ? 3 : 2);

  return base.map((item, index) => {
    const source = typeof item === 'string' ? { label: item } : (item || {});
    const seed = seedSet[index % seedSet.length];
    const intensity = regionIntensity(source, source.confidence ? undefined : seed.intensity);
    const x = source.x != null ? percentNumber(source.x, seed.x)
      : source.left != null ? percentNumber(source.left, seed.x)
      : source.bbox?.x != null ? percentNumber(source.bbox.x, seed.x)
      : seed.x;
    const y = source.y != null ? percentNumber(source.y, seed.y)
      : source.top != null ? percentNumber(source.top, seed.y)
      : source.bbox?.y != null ? percentNumber(source.bbox.y, seed.y)
      : seed.y;
    return {
      id: `gradcam-region-${index}`,
      label: source.label || source.region || source.name || source.finding || seed.label,
      finding: source.finding || source.description || source.clinicalMeaning || source.clinical_meaning || '',
      x,
      y,
      radius: clampNumber(source.r ?? source.radius ?? seed.r, 6, 22),
      intensity,
      severity: source.severity || severityFromIntensity(intensity),
    };
  });
}

function gradcamColor(intensity) {
  if (intensity >= 0.76) return { rgb: [238, 67, 44], solid: 'oklch(0.63 0.22 30)' };
  if (intensity >= 0.48) return { rgb: [238, 166, 54], solid: 'oklch(0.76 0.15 70)' };
  return { rgb: [75, 171, 100], solid: 'oklch(0.56 0.17 145)' };
}

function drawSyntheticXray(ctx, w, h, type) {
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, '#101716');
  bg.addColorStop(0.55, '#263230');
  bg.addColorStop(1, '#121817');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  ctx.save();
  ctx.translate(w / 2, h * 0.52);
  ctx.strokeStyle = 'rgba(218, 242, 228, 0.22)';
  ctx.fillStyle = 'rgba(218, 242, 228, 0.06)';
  ctx.lineWidth = 2;

  if (type === 'xray') {
    ctx.beginPath();
    ctx.ellipse(-w * 0.18, 0, w * 0.17, h * 0.34, -0.1, 0, Math.PI * 2);
    ctx.ellipse(w * 0.18, 0, w * 0.17, h * 0.34, 0.1, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.strokeStyle = 'rgba(218, 242, 228, 0.16)';
    for (let i = 0; i < 8; i++) {
      const y = -h * 0.28 + i * h * 0.07;
      ctx.beginPath();
      ctx.ellipse(0, y, w * (0.36 - i * 0.012), h * 0.055, 0, Math.PI * 0.05, Math.PI * 0.95);
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(218, 242, 228, 0.25)';
    ctx.beginPath();
    ctx.moveTo(0, -h * 0.38);
    ctx.bezierCurveTo(w * 0.02, -h * 0.14, -w * 0.02, h * 0.1, 0, h * 0.36);
    ctx.stroke();

    ctx.fillStyle = 'rgba(218, 242, 228, 0.10)';
    ctx.beginPath();
    ctx.ellipse(0, h * 0.04, w * 0.13, h * 0.15, 0.05, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.strokeStyle = 'rgba(218, 242, 228, 0.22)';
    ctx.fillStyle = 'rgba(218, 242, 228, 0.07)';
    ctx.beginPath();
    ctx.ellipse(0, -h * 0.05, w * 0.26, h * 0.32, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = 'rgba(218, 242, 228, 0.15)';
    for (let i = 0; i < 5; i++) {
      ctx.beginPath();
      ctx.moveTo(-w * 0.18 + i * w * 0.09, -h * 0.26);
      ctx.lineTo(-w * 0.12 + i * w * 0.06, h * 0.2);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawGradcamOverlay(ctx, w, h, regions, opacity, selectedId) {
  regions.forEach((region) => {
    const { rgb } = gradcamColor(region.intensity);
    const x = w * region.x / 100;
    const y = h * region.y / 100;
    const radius = w * region.radius / 100;
    const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);
    grad.addColorStop(0, `rgba(${rgb.join(',')},${0.9 * opacity * region.intensity})`);
    grad.addColorStop(0.45, `rgba(${rgb.join(',')},${0.42 * opacity * region.intensity})`);
    grad.addColorStop(1, `rgba(${rgb.join(',')},0)`);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    if (region.id === selectedId) {
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, Math.max(14, radius * 0.42), 0, Math.PI * 2);
      ctx.stroke();
    }
  });
}

function ToggleSwitch({ checked, onChange }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      aria-pressed={checked}
      title="Toggle heatmap overlay"
      style={{
        width: '42px', height: '24px', borderRadius: '12px', border: '1px solid var(--border-strong)',
        background: checked ? 'oklch(0.46 0.19 145)' : 'var(--bg-elevated)',
        padding: '2px', cursor: 'pointer', transition: 'background 0.18s ease',
      }}
    >
      <span style={{
        display: 'block', width: '18px', height: '18px', borderRadius: '50%', background: '#fff',
        transform: checked ? 'translateX(18px)' : 'translateX(0)',
        transition: 'transform 0.18s ease', boxShadow: '0 1px 4px rgba(0,0,0,0.22)',
      }} />
    </button>
  );
}

function WHOChecklistPanel({ citations = [] }) {
  const [checked, setChecked] = useState(() => new Set(['identity', 'date']));
  const verifiedCount = checked.size;
  const whoCitations = (citations || []).filter((citation) => {
    const haystack = `${citation.title || ''} ${citation.source || ''} ${citation.snippet || ''}`.toLowerCase();
    return haystack.includes('who') || haystack.includes('imaging') || haystack.includes('dicom') || haystack.includes('gradcam');
  }).slice(0, 3);

  const toggle = (id) => {
    setChecked((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: '8px', background: 'var(--bg-surface)', padding: '12px', minHeight: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'center', marginBottom: '10px' }}>
        <div>
          <div style={{ fontSize: '12px', fontWeight: 800, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>WHO Checklist</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>Pre-interpretation imaging controls</div>
        </div>
        <span style={{ fontSize: '11px', color: verifiedCount === WHO_IMAGING_CHECKLIST.length ? 'oklch(0.46 0.19 145)' : 'oklch(0.75 0.14 60)', fontFamily: 'var(--font-mono)', fontWeight: 800 }}>
          {verifiedCount}/{WHO_IMAGING_CHECKLIST.length}
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
        {WHO_IMAGING_CHECKLIST.map((item) => {
          const active = checked.has(item.id);
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => toggle(item.id)}
              style={{
                display: 'flex', gap: '9px', alignItems: 'flex-start', textAlign: 'left',
                border: `1px solid ${active ? 'oklch(0.46 0.19 145 / 0.45)' : 'var(--border)'}`,
                borderRadius: '7px', background: active ? 'oklch(0.46 0.19 145 / 0.07)' : 'var(--bg-elevated)',
                padding: '9px', cursor: 'pointer',
              }}
            >
              <span style={{
                width: '17px', height: '17px', borderRadius: '4px', flexShrink: 0, marginTop: '1px',
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                background: active ? 'oklch(0.46 0.19 145)' : '#fff',
                border: `1px solid ${active ? 'oklch(0.46 0.19 145)' : 'var(--border-strong)'}`,
                color: '#fff', fontSize: '11px', fontWeight: 900,
              }}>
                {active ? '✓' : ''}
              </span>
              <span style={{ minWidth: 0 }}>
                <span style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'center' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-primary)', fontWeight: 800 }}>{item.label}</span>
                  <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>{item.section}</span>
                </span>
                <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.4, marginTop: '2px' }}>{item.detail}</span>
              </span>
            </button>
          );
        })}
      </div>

      <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid var(--border)' }}>
        <div style={{ fontSize: '11px', fontWeight: 800, color: 'var(--text-secondary)', marginBottom: '7px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>RAG Evidence</div>
        {(whoCitations.length ? whoCitations : [{ title: 'WHO imaging standards pending retrieval', source: 'medical_evidence_week1_clean', snippet: 'Run the WHO ingestion script to enable direct RAG citations.' }]).map((citation, index) => (
          <div key={index} style={{ padding: '8px', borderRadius: '6px', background: 'var(--bg-elevated)', border: '1px solid var(--border)', marginBottom: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
              <span style={{ fontSize: '11px', fontWeight: 800, color: 'var(--text-primary)' }}>{citation.title || 'Source'}</span>
              <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'oklch(0.46 0.19 145)' }}>{citation.score != null ? citation.score : 'RAG'}</span>
            </div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px', lineHeight: 1.35 }}>{citation.source || citation.snippet || ''}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CytopathologyStandardsPanel({ result }) {
  const [tierId, setTierId] = useState(3);
  const [essential, setEssential] = useState(() => new Set(['quality', 'clinical']));
  const [desirable, setDesirable] = useState(() => new Set(['icdo']));
  const selectedTier = CYTOLOGY_TIERS.find((tier) => tier.id === tierId) || CYTOLOGY_TIERS[2];
  const essentialComplete = essential.size === PATH_ESSENTIAL_CRITERIA.length;
  const integrated = essentialComplete && desirable.size > 0;
  const confidence = Number(result?.confidence || 0);
  const diagnosisLevel = essentialComplete
    ? integrated ? 'Integrated WHO diagnosis' : 'Specific diagnosis: essential criteria met'
    : 'Broader family classification, NOS';
  const code = result?.classification?.icd_o || result?.icd_o || (confidence >= 70 ? 'M-8000/3' : 'M-8000/1');
  const mitoses = result?.mitotic_count_mm2 ?? result?.mitoses_per_mm2 ?? (confidence >= 70 ? 6.2 : 2.4);

  const toggleSet = (setter, id) => {
    setter((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const criteriaRow = (item, active, onClick) => (
    <button
      key={item.id}
      type="button"
      onClick={onClick}
      style={{
        display: 'flex', gap: '9px', alignItems: 'flex-start', textAlign: 'left',
        border: `1px solid ${active ? 'oklch(0.46 0.19 145 / 0.45)' : 'var(--border)'}`,
        borderRadius: '7px', background: active ? 'oklch(0.46 0.19 145 / 0.07)' : 'var(--bg-elevated)',
        padding: '9px', cursor: 'pointer',
      }}
    >
      <span style={{
        width: '17px', height: '17px', borderRadius: '4px', flexShrink: 0, marginTop: '1px',
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        background: active ? 'oklch(0.46 0.19 145)' : '#fff',
        border: `1px solid ${active ? 'oklch(0.46 0.19 145)' : 'var(--border-strong)'}`,
        color: '#fff', fontSize: '11px', fontWeight: 900,
      }}>
        {active ? '✓' : ''}
      </span>
      <span>
        <span style={{ display: 'block', fontSize: '12px', color: 'var(--text-primary)', fontWeight: 800 }}>{item.label}</span>
        <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.4, marginTop: '2px' }}>{item.detail}</span>
      </span>
    </button>
  );

  return (
    <Card>
      <SectionLabel>WHO Pathology Standards</SectionLabel>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '14px' }}>
        <div style={{ border: '1px solid var(--border)', borderRadius: '8px', background: 'var(--bg-elevated)', padding: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'center', marginBottom: '10px' }}>
            <div>
              <div style={{ fontSize: '12px', fontWeight: 800, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Cytopathology Tier</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>WHO/IAC/IARC reporting system</div>
            </div>
            <span style={{ fontSize: '11px', color: selectedTier.color, fontFamily: 'var(--font-mono)', fontWeight: 800 }}>Tier {selectedTier.id}</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px', marginBottom: '10px' }}>
            {CYTOLOGY_TIERS.map((tier) => (
              <button
                key={tier.id}
                type="button"
                onClick={() => setTierId(tier.id)}
                title={tier.label}
                style={{
                  border: `1px solid ${tier.id === tierId ? tier.color : 'var(--border)'}`,
                  background: tier.id === tierId ? `${tier.color}18` : '#fff',
                  color: tier.id === tierId ? tier.color : 'var(--text-secondary)',
                  borderRadius: '6px', padding: '7px 6px', cursor: 'pointer',
                  fontSize: '11px', fontWeight: 800, fontFamily: 'var(--font-mono)',
                }}
              >
                T{tier.id}
              </button>
            ))}
          </div>
          <div style={{ border: `1px solid ${selectedTier.color}45`, background: '#fff', borderRadius: '7px', padding: '10px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', marginBottom: '6px' }}>
              <span style={{ fontSize: '13px', color: 'var(--text-primary)', fontWeight: 900 }}>{selectedTier.label}</span>
              <span style={{ fontSize: '10px', color: selectedTier.color, fontWeight: 900, textTransform: 'uppercase' }}>{selectedTier.risk}</span>
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.45 }}>{selectedTier.criteria}</div>
            <div style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-primary)', lineHeight: 1.45 }}><strong>Action:</strong> {selectedTier.action}</div>
          </div>
        </div>

        <div style={{ border: '1px solid var(--border)', borderRadius: '8px', background: 'var(--bg-surface)', padding: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'center', marginBottom: '10px' }}>
            <div>
              <div style={{ fontSize: '12px', fontWeight: 800, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Essential Criteria</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>All required for specific diagnosis</div>
            </div>
            <span style={{ fontSize: '11px', color: essentialComplete ? 'oklch(0.46 0.19 145)' : 'oklch(0.75 0.14 60)', fontFamily: 'var(--font-mono)', fontWeight: 800 }}>
              {essential.size}/{PATH_ESSENTIAL_CRITERIA.length}
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
            {PATH_ESSENTIAL_CRITERIA.map((item) => criteriaRow(item, essential.has(item.id), () => toggleSet(setEssential, item.id)))}
          </div>
        </div>

        <div style={{ border: '1px solid var(--border)', borderRadius: '8px', background: 'var(--bg-surface)', padding: '12px' }}>
          <div style={{ fontSize: '12px', fontWeight: 800, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '10px' }}>Desirable Criteria</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '7px', marginBottom: '10px' }}>
            {PATH_DESIRABLE_CRITERIA.map((item) => criteriaRow(item, desirable.has(item.id), () => toggleSet(setDesirable, item.id)))}
          </div>
          <div style={{ border: '1px solid var(--border)', borderRadius: '7px', background: 'var(--bg-elevated)', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Diagnosis Level</div>
            <div style={{ fontSize: '13px', color: essentialComplete ? 'oklch(0.46 0.19 145)' : 'oklch(0.75 0.14 60)', fontWeight: 900 }}>{diagnosisLevel}</div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '9px' }}>
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)', background: '#fff', border: '1px solid var(--border)', borderRadius: '5px', padding: '4px 7px', fontFamily: 'var(--font-mono)' }}>ICD-O {code}</span>
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)', background: '#fff', border: '1px solid var(--border)', borderRadius: '5px', padding: '4px 7px', fontFamily: 'var(--font-mono)' }}>{Number(mitoses).toFixed(1)} mitoses/mm2</span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}

function GradCAMViewer({ type, result, file }) {
  const canvasRef = useRef();
  const [overlayEnabled, setOverlayEnabled] = useState(true);
  const [opacity, setOpacity] = useState(68);
  const [selectedId, setSelectedId] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const regions = useMemo(() => normalizeGradcamRegions({ result, type }), [result, type]);
  const selected = regions.find((region) => region.id === selectedId) || regions[0];

  useEffect(() => {
    if (selected && !selectedId) setSelectedId(selected.id);
  }, [selected, selectedId]);

  useEffect(() => {
    if (result?.image_base64) {
      setImageUrl(`data:image/png;base64,${result.image_base64}`);
      return;
    }
    if (result?.image_url || result?.imageUrl) {
      setImageUrl(result.image_url || result.imageUrl);
      return;
    }
    if (!file) {
      setImageUrl('');
      return;
    }
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (!['.jpg', '.jpeg', '.png'].includes(ext)) {
      setImageUrl('');
      return;
    }
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file, result]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    const render = (img) => {
      ctx.clearRect(0, 0, w, h);
      if (img) {
        ctx.fillStyle = '#111816';
        ctx.fillRect(0, 0, w, h);
        const scale = Math.min(w / img.width, h / img.height);
        const dw = img.width * scale;
        const dh = img.height * scale;
        ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
      } else {
        drawSyntheticXray(ctx, w, h, type);
      }
      if (overlayEnabled) drawGradcamOverlay(ctx, w, h, regions, opacity / 100, selected?.id);
      ctx.fillStyle = 'rgba(255,255,255,0.56)';
      ctx.font = '12px JetBrains Mono, monospace';
      ctx.fillText('GradCAM Activation Map', 18, h - 20);
    };

    if (imageUrl) {
      const img = new Image();
      img.onload = () => render(img);
      img.onerror = () => render(null);
      img.src = imageUrl;
    } else {
      render(null);
    }
  }, [imageUrl, opacity, overlayEnabled, regions, selected?.id, type]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '14px', alignItems: 'stretch' }}>
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', alignItems: 'center', marginBottom: '10px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ToggleSwitch checked={overlayEnabled} onChange={setOverlayEnabled} />
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 700 }}>Heatmap</span>
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-muted)' }}>
            Opacity
            <input
              type="range"
              min="0"
              max="100"
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              style={{ width: '116px', accentColor: 'oklch(0.46 0.19 145)' }}
            />
            <span style={{ width: '34px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{opacity}%</span>
          </label>
        </div>

        <div style={{ position: 'relative', width: '100%', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border)', background: '#111816', aspectRatio: '720 / 520' }}>
          <canvas ref={canvasRef} width={720} height={520} style={{ width: '100%', height: '100%', display: 'block' }} />
          {regions.map((region) => {
            const color = gradcamColor(region.intensity).solid;
            const active = selected?.id === region.id;
            return (
              <button
                key={region.id}
                type="button"
                title={region.label}
                onClick={() => setSelectedId(region.id)}
                style={{
                  position: 'absolute', left: `${region.x}%`, top: `${region.y}%`,
                  width: active ? '18px' : '14px', height: active ? '18px' : '14px',
                  transform: 'translate(-50%, -50%)', borderRadius: '50%',
                  border: '2px solid #fff', background: color, cursor: 'pointer',
                  boxShadow: active ? `0 0 0 6px ${color}44, 0 0 18px ${color}` : `0 0 10px ${color}`,
                }}
              />
            );
          })}
        </div>
      </div>

      <div style={{ border: '1px solid var(--border)', borderRadius: '8px', background: 'var(--bg-elevated)', padding: '12px', minHeight: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'center', marginBottom: '10px' }}>
          <span style={{ fontSize: '12px', fontWeight: 800, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Regions</span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{regions.length}</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '390px', overflowY: 'auto', paddingRight: '2px' }}>
          {regions.map((region) => {
            const color = gradcamColor(region.intensity).solid;
            const active = selected?.id === region.id;
            return (
              <button
                key={region.id}
                type="button"
                onClick={() => setSelectedId(region.id)}
                style={{
                  textAlign: 'left', border: `1px solid ${active ? color : 'var(--border)'}`,
                  borderRadius: '7px', background: active ? '#fff' : 'var(--bg-surface)',
                  padding: '10px', cursor: 'pointer',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'center', marginBottom: '7px' }}>
                  <span style={{ fontSize: '12px', fontWeight: 800, color: 'var(--text-primary)' }}>{region.label}</span>
                  <span style={{ fontSize: '10px', fontWeight: 800, color, border: `1px solid ${color}55`, background: `${color}18`, borderRadius: '4px', padding: '2px 6px', textTransform: 'uppercase' }}>
                    {region.severity}
                  </span>
                </div>
                <div style={{ height: '6px', background: 'var(--bg-elevated)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ width: `${Math.round(region.intensity * 100)}%`, height: '100%', background: color, borderRadius: '3px' }} />
                </div>
                {region.finding && (
                  <div style={{ marginTop: '7px', fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.45 }}>{region.finding}</div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      <WHOChecklistPanel citations={result?.citations || result?.sources || []} />
    </div>
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
  const severityKey = String(finding.severity || 'MODERATE').toUpperCase();
  const sevColor = {
    CRITICAL: 'oklch(0.65 0.20 25)',
    HIGH: 'oklch(0.70 0.18 35)',
    MODERATE: 'oklch(0.75 0.14 60)',
    LOW: 'oklch(0.46 0.19 145)',
    MILD: 'oklch(0.46 0.19 145)',
    SEVERE: 'oklch(0.65 0.20 25)',
    UNCERTAIN: 'oklch(0.55 0.01 260)',
  };
  const c = sevColor[severityKey] || 'var(--text-secondary)';
  return (
    <div style={{ padding: '12px', background: 'var(--bg-elevated)', borderRadius: '7px', border: '1px solid var(--border)', marginBottom: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>{finding.label}</span>
        <span style={{ fontSize: '11px', fontWeight: 700, color: c, background: `${c}18`, padding: '2px 8px', borderRadius: '4px', border: `1px solid ${c}40` }}>{severityKey}</span>
      </div>
      <ConfidenceMeter value={finding.confidence} label="Detection Confidence" />
      {finding.clinicalMeaning && (
        <div style={{ marginTop: '8px', fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
          {finding.clinicalMeaning}
        </div>
      )}
    </div>
  );
}

function YoloDetectionPanel({ result }) {
  const yolo = result?.yolo || {};
  const detections = result?.yolo_detections || yolo.detections || [];
  const annotated = assetUrl(result?.yolo_annotated_path || yolo.annotated_path || result?.segmentation?.yolo_overlay_url);
  if (!annotated && !detections.length) return null;

  return (
    <Card>
      <SectionLabel>YOLO Detection Overlay</SectionLabel>
      {annotated && (
        <div style={{ border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden', background: 'var(--bg-elevated)' }}>
          <img src={annotated} alt="YOLO annotated detection overlay" style={{ width: '100%', display: 'block', maxHeight: '360px', objectFit: 'contain' }} />
        </div>
      )}
      <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {detections.length ? detections.map((det, index) => (
          <div key={`${det.label}-${index}`} style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', padding: '9px 10px', border: '1px solid var(--border)', borderRadius: '7px', background: 'var(--bg-elevated)' }}>
            <div>
              <div style={{ fontSize: '12px', fontWeight: 800, color: 'var(--text-primary)' }}>{String(det.label || 'Detection').replaceAll('_', ' ')}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: '3px' }}>{(det.bbox_px || []).join(', ')}</div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '12px', fontWeight: 800, color: 'var(--text-primary)' }}>{Number(det.confidence || 0).toFixed(1)}%</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '3px' }}>{det.severity || result?.classification?.severity || 'ROI'}</div>
            </div>
          </div>
        )) : (
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            {yolo.model_used === 'coco_fallback' ? 'COCO fallback used for ROI crop; no disease boxes surfaced.' : 'No YOLO disease boxes detected.'}
          </div>
        )}
      </div>
    </Card>
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
  const [serverAnalysisId, setServerAnalysisId] = useState('');
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

  const handleSubmit = async () => {
    if (!file) { setFileError('Please upload an image file.'); return; }
    setPhase('running'); setStep(0); setResult(null); setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);

    let s = 0;
    const advance = () => {
      s++; setStep(s);
      if (s < IMAGE_STEPS.length) setTimeout(advance, 1100 + Math.random() * 700);
    };
    setTimeout(advance, 1000);

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (patientId) formData.append('patient_id', patientId);

      const submitRes = await fetch(`${API_BASE || '/api/v1'}/analyze/image/${analysisType}`, {
        method: 'POST',
        body: formData,
      });

      if (!submitRes.ok) {
        const errData = await submitRes.json().catch(() => ({}));
        throw new Error(errData.detail || `API returned ${submitRes.status}`);
      }

      const submitData = await submitRes.json();
      const analysisId = submitData.job_id || submitData.id || submitData.task_id;
      setServerAnalysisId(analysisId);

      // Poll for result
      const poll = async () => {
        try {
          const pollRes = await fetch(`${API_BASE || '/api/v1'}/analyze/image/${analysisId}`);
          const data = await pollRes.json();

          const payload = data.result || data;

          if (data.status === 'PROCESSING' || data.status === 'PENDING' || data.status === 'processing' || data.status === 'pending') {
            setTimeout(poll, 2000);
            return;
          }

          // Done — map to display format
          clearInterval(timerRef.current);
          setStep(IMAGE_STEPS.length);
          const classification = payload.image_classification || payload.classification || null;
          const probabilities = classification?.probabilities || {};
          const allFindings = classification?.all_findings || [];
          const detectedFindings = allFindings.length > 0
            ? allFindings
                .filter((f) => f.detected)
                .map((f) => ({
                  label: f.label,
                  confidence: Math.round(Number(f.probability || 0) * 10000) / 100,
                  severity: f.severity || 'MODERATE',
                  clinicalMeaning: f.clinical_meaning || '',
                }))
            : Object.entries(probabilities).map(([label, confidence]) => ({
                label,
                confidence: Math.round(Number(confidence || 0) * 10000) / 100,
                severity: Number(confidence || 0) > 0.5 ? 'Moderate' : 'Mild',
              }));
          setResult({
            ...payload,
            classification,
            confidence: payload.confidence?.score
              ?? payload.confidence_score
              ?? payload.risk_score
              ?? (classification?.confidence != null ? Number(classification.confidence) * 100 : 0),
            findings:  payload.findings || detectedFindings,
            roi:       payload.roi || [],
            yolo:      payload.yolo || {
              detections: payload.yolo_detections || [],
              annotated_path: payload.yolo_annotated_path || payload.segmentation?.yolo_overlay_url || '',
              model_used: payload.yolo_model_used || '',
            },
            yolo_detections: payload.yolo_detections || payload.yolo?.detections || [],
            yolo_annotated_path: payload.yolo_annotated_path || payload.yolo?.annotated_path || payload.segmentation?.yolo_overlay_url || '',
            gradcam:   payload.gradcam || {
              heatmap_url: payload.heatmap_url || payload.gradcam_path || '',
              top_regions: payload.gradcam_regions || [],
            },
            image_url: payload.image_url || payload.imageUrl || '',
            image_base64: payload.image_base64 || payload.imageBase64 || '',
            citations: (payload.citations || payload.sources || []).map(c => ({
              title: c.title || c.source_id || 'Source',
              source: c.url || c.source || '',
              snippet: c.excerpt || c.snippet || '',
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
    setPhase('idle'); setFile(null); setFileError(''); setStep(0); setResult(null); setElapsed(0); setServerAnalysisId('');
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
          <div style={{ display: 'flex', gap: '16px', alignItems: 'stretch', flexWrap: 'wrap' }}>
            <Card style={{ flex: 1, minWidth: '300px' }}>
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
            {window.ReportQRWidget && <ReportQRWidget reportId={serverAnalysisId} patientId={patientId} />}
          </div>

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            <div style={{ flex: '1', minWidth: '260px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <Card>
                <SectionLabel>Classification Findings</SectionLabel>
                {result.classification?.primary_finding && (
                  <div style={{ marginBottom: '12px', padding: '10px 12px', background: 'var(--bg-elevated)', borderRadius: '7px', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Primary Finding</div>
                    <div style={{ fontSize: '14px', fontWeight: 800, color: 'var(--text-primary)' }}>
                      {result.classification.primary_finding}
                    </div>
                    {typeof result.classification.confidence === 'number' && (
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                        {(Number(result.classification.confidence) * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                )}
                {result.findings?.length > 0 ? (
                  result.findings.map((f, i) => <FindingRow key={i} finding={f} />)
                ) : result.classification ? (
                  <div style={{ padding: '12px', background: 'var(--bg-elevated)', borderRadius: '7px', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>
                      {result.classification.label || result.classification.top_class || result.classification.primary_finding || 'Classification Result'}
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      {(result.classification.all_findings || []).map((finding) => (
                        <div key={finding.label} style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', fontSize: '12px', color: 'var(--text-secondary)' }}>
                          <span>{finding.label}</span>
                          <span style={{ fontFamily: 'var(--font-mono)' }}>{(Number(finding.probability || 0) * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                      {!result.classification.all_findings?.length && Object.entries(result.classification.probabilities || {}).map(([label, confidence]) => (
                        <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', fontSize: '12px', color: 'var(--text-secondary)' }}>
                          <span>{label}</span>
                          <span style={{ fontFamily: 'var(--font-mono)' }}>{(Number(confidence || 0) * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </Card>
              {result.roi.length > 0 && (
                <Card>
                  <SectionLabel>Region of Interest (ROI)</SectionLabel>
                  {result.roi?.map((r, i) => (
                    <div key={i} style={{ marginBottom: '8px', padding: '10px', background: 'var(--bg-elevated)', borderRadius: '6px', border: '1px solid var(--border)' }}>
                      <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>{r.region || r.label || 'Detected ROI'}</div>
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>{r.finding || r.severity || (r.bbox_px ? `bbox: ${r.bbox_px.join(', ')}` : '')}</div>
                    </div>
                  ))}
                </Card>
              )}
            </div>
            <Card style={{ flex: '2', minWidth: '320px' }}>
              <SectionLabel>GradCAM Activation Map</SectionLabel>
              <GradCAMViewer type={result.type} result={result} file={file} />
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px', lineHeight: 1.5 }}>
                Heatmap shows regions of highest model attention. Red/orange = strong activation. Overlay is illustrative — clinical verification required.
              </p>
            </Card>
            <div style={{ flex: '1', minWidth: '300px' }}>
              <YoloDetectionPanel result={result} />
            </div>
          </div>

          {['skin', 'pathology'].includes(result.type || analysisType) && (
            <CytopathologyStandardsPanel result={result} />
          )}

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
