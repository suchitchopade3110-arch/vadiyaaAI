{

// job-tracker.jsx — Job Tracker page

const { useState, useEffect, useRef } = React;

const ALL_JOBS = [
  { id: 'JOB-001', taskId: 'celery-a1b2c3d4', type: 'claim',  status: 'verified',   created: '2026-04-25 09:14:02', desc: 'Aspirin reduces MI risk by 25%', duration: '18s' },
  { id: 'JOB-002', taskId: 'celery-e5f6g7h8', type: 'report', status: 'processing',  created: '2026-04-25 09:08:41', desc: 'Lab panel — John D.', duration: '—' },
  { id: 'JOB-003', taskId: 'celery-i9j0k1l2', type: 'image',  status: 'verified',   created: '2026-04-25 08:55:17', desc: 'Chest X-ray — PT-20240125', duration: '31s' },
  { id: 'JOB-004', taskId: 'celery-m3n4o5p6', type: 'claim',  status: 'refuted',    created: '2026-04-25 08:30:55', desc: 'High-dose Vit C cures cancer', duration: '22s' },
  { id: 'JOB-005', taskId: 'celery-q7r8s9t0', type: 'image',  status: 'uncertain',  created: '2026-04-25 08:12:38', desc: 'MRI brain — PT-20240120', duration: '44s' },
  { id: 'JOB-006', taskId: 'celery-u1v2w3x4', type: 'report', status: 'failed',     created: '2026-04-25 07:50:22', desc: 'Discharge summary upload', duration: '3s' },
  { id: 'JOB-007', taskId: 'celery-y5z6a7b8', type: 'claim',  status: 'pending',    created: '2026-04-25 07:44:10', desc: 'Metformin lowers HbA1c in T2DM', duration: '—' },
  { id: 'JOB-008', taskId: 'celery-c9d0e1f2', type: 'image',  status: 'verified',   created: '2026-04-25 07:31:05', desc: 'Skin lesion dermoscopy', duration: '27s' },
];

const PIPELINE_ICONS = { claim: '◎', report: '▤', image: '⬡' };
const PIPELINE_LABELS = { claim: 'Claim', report: 'Report', image: 'Image' };
const TABS = ['all', 'claim', 'report', 'image'];

function PollingDrawer({ job, onClose }) {
  const [pollStep, setPollStep] = useState(0);
  const [logs, setLogs] = useState([]);
  const logsEndRef = useRef();

  const LOG_LINES = [
    `[09:08:41] Job ${job.id} accepted — Celery task ${job.taskId}`,
    `[09:08:42] OCR extraction started`,
    `[09:08:44] Extracted 1,842 tokens from PDF`,
    `[09:08:46] ClinicalBERT NER: 12 entities found`,
    `[09:08:49] XGBoost scoring: input features compiled`,
    `[09:08:51] SHAP explanation: 5 top features`,
    `[09:08:54] LLM/RAG synthesis: querying knowledge base`,
    `[09:08:57] Result ready — awaiting finalization`,
  ];

  useEffect(() => {
    if (job.status !== 'processing') return;
    let i = 0;
    const interval = setInterval(() => {
      if (i < LOG_LINES.length) {
        setLogs(l => [...l, LOG_LINES[i]]);
        i++;
      } else clearInterval(interval);
    }, 700);
    return () => clearInterval(interval);
  }, [job.id]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [logs]);

  return (
    <div style={{
      position: 'fixed', right: 0, top: 0, bottom: 0, width: '380px',
      background: 'var(--bg-surface)', borderLeft: '1px solid var(--border)',
      zIndex: 50, display: 'flex', flexDirection: 'column',
      boxShadow: '-8px 0 32px rgba(0,0,0,0.4)',
      animation: 'slide-in-right 0.25s ease',
    }}>
      {/* Header */}
      <div style={{ padding: '20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: '14px', fontWeight: 800, color: 'var(--text-primary)' }}>Job Details</div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px', fontFamily: 'var(--font-mono)' }}>{job.id}</div>
        </div>
        <button onClick={onClose} style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: '6px', width: '32px', height: '32px', cursor: 'pointer', color: 'var(--text-secondary)', fontSize: '16px' }}>✕</button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 12px', background: 'var(--bg-elevated)', borderRadius: '6px', border: '1px solid var(--border)' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Status</span>
            <StatusBadge status={job.status} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 12px', background: 'var(--bg-elevated)', borderRadius: '6px', border: '1px solid var(--border)' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Pipeline</span>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>{PIPELINE_ICONS[job.type]} {PIPELINE_LABELS[job.type]}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 12px', background: 'var(--bg-elevated)', borderRadius: '6px', border: '1px solid var(--border)' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Created</span>
            <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{job.created}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 12px', background: 'var(--bg-elevated)', borderRadius: '6px', border: '1px solid var(--border)' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Duration</span>
            <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{job.duration}</span>
          </div>
          <div style={{ padding: '10px 12px', background: 'var(--bg-elevated)', borderRadius: '6px', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>Celery Task ID</div>
            <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'oklch(0.46 0.19 145)', wordBreak: 'break-all' }}>{job.taskId}</div>
          </div>
        </div>

        {/* Description */}
        <div style={{ padding: '12px', background: 'var(--bg-elevated)', borderRadius: '6px', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 700 }}>Description</div>
          <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{job.desc}</div>
        </div>

        {/* Live logs for processing jobs */}
        {job.status === 'processing' && (
          <div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 700, marginBottom: '8px' }}>Live Logs</div>
            <div style={{ background: 'oklch(0.95 0.012 145)', borderRadius: '7px', border: '1px solid var(--border)', padding: '12px', maxHeight: '200px', overflowY: 'auto', fontFamily: 'var(--font-mono)', fontSize: '11px', lineHeight: 1.8 }}>
              {logs.map((line, i) => (
                <div key={i} style={{ color: i === logs.length - 1 ? 'oklch(0.46 0.19 145)' : 'var(--text-muted)' }}>{line}</div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        )}

        {/* Failure detail */}
        {job.status === 'failed' && (
          <div style={{ background: 'oklch(0.65 0.20 25 / 0.1)', border: '1px solid oklch(0.65 0.20 25 / 0.3)', borderRadius: '6px', padding: '12px' }}>
            <div style={{ fontSize: '11px', color: 'oklch(0.65 0.20 25)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '6px' }}>Failure Detail</div>
            <div style={{ fontSize: '12px', color: 'oklch(0.65 0.20 25)', fontFamily: 'var(--font-mono)', lineHeight: 1.6 }}>
              FileValidationError: MIME type mismatch.<br />
              Expected application/pdf, received application/octet-stream.<br />
              Stack: file_validator.py:87 → upload_handler.py:34
            </div>
          </div>
        )}
      </div>

      {/* Footer actions */}
      <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border)', display: 'flex', gap: '8px' }}>
        {(job.status === 'processing' || job.status === 'pending') && (
          <Button variant="ghost" style={{ fontSize: '12px', flex: 1, color: 'oklch(0.65 0.20 25)' }}>
            ✕ Cancel Job
          </Button>
        )}
        {job.status === 'failed' && (
          <Button style={{ fontSize: '12px', flex: 1 }}>↻ Retry Job</Button>
        )}
        {(job.status === 'verified' || job.status === 'refuted' || job.status === 'uncertain') && (
          <Button style={{ fontSize: '12px', flex: 1 }}>↗ View Result</Button>
        )}
      </div>
    </div>
  );
}

function JobTracker() {
  const [activeTab, setActiveTab] = useState('all');
  const [selectedJob, setSelectedJob] = useState(null);
  const [jobs, setJobs] = useState(ALL_JOBS);

  const filtered = activeTab === 'all' ? jobs : jobs.filter(j => j.type === activeTab);

  const tabCount = (tab) => tab === 'all' ? jobs.length : jobs.filter(j => j.type === tab).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>Job Tracker</h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '4px 0 0' }}>Monitor and manage all pipeline jobs</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '7px', height: '7px', borderRadius: '50%', background: 'oklch(0.46 0.19 145)', boxShadow: '0 0 8px oklch(0.46 0.19 145)', animation: 'pulse-dot 2s ease infinite' }} />
          <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 600 }}>Live polling</span>
        </div>
      </div>

      {/* Filter tabs */}
      <div style={{ display: 'flex', gap: '4px', borderBottom: '1px solid var(--border)', paddingBottom: '0' }}>
        {TABS.map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)} style={{
            padding: '9px 16px', background: 'transparent', border: 'none', cursor: 'pointer',
            fontSize: '13px', fontWeight: 600,
            color: activeTab === tab ? 'oklch(0.46 0.19 145)' : 'var(--text-muted)',
            borderBottom: `2px solid ${activeTab === tab ? 'oklch(0.46 0.19 145)' : 'transparent'}`,
            transition: 'all 0.18s', marginBottom: '-1px',
            display: 'flex', alignItems: 'center', gap: '6px',
          }}>
            <span style={{ textTransform: 'capitalize' }}>{tab === 'all' ? 'All' : PIPELINE_LABELS[tab]}</span>
            <span style={{
              fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 700,
              background: activeTab === tab ? 'oklch(0.46 0.19 145 / 0.12)' : 'var(--bg-elevated)',
              color: activeTab === tab ? 'oklch(0.46 0.19 145)' : 'var(--text-muted)',
              padding: '1px 6px', borderRadius: '10px',
            }}>{tabCount(tab)}</span>
          </button>
        ))}
      </div>

      {/* Jobs table */}
      <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: '10px', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-elevated)' }}>
              {['Job ID', 'Pipeline', 'Description', 'Status', 'Created', 'Duration', ''].map((h, i) => (
                <th key={i} style={{ padding: '11px 14px', textAlign: 'left', fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((job, i) => (
              <TrackerRow key={job.id} job={job} last={i === filtered.length - 1} onClick={() => setSelectedJob(job)} onCancel={() => setJobs(j => j.filter(jj => jj.id !== job.id))} />
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>No jobs found</div>
        )}
      </div>

      {selectedJob && <PollingDrawer job={selectedJob} onClose={() => setSelectedJob(null)} />}
    </div>
  );
}

function TrackerRow({ job, last, onClick, onCancel }) {
  const [hov, setHov] = useState(false);
  return (
    <tr onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{ borderBottom: last ? 'none' : '1px solid var(--border)', background: hov ? 'var(--bg-elevated)' : 'transparent', transition: 'background 0.15s', cursor: 'pointer' }}
      onClick={onClick}>
      <td style={{ padding: '11px 14px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'oklch(0.46 0.19 145)', fontWeight: 600 }}>{job.id}</td>
      <td style={{ padding: '11px 14px', fontSize: '12px', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
        {PIPELINE_ICONS[job.type]} <span style={{ fontWeight: 600 }}>{PIPELINE_LABELS[job.type]}</span>
      </td>
      <td style={{ padding: '11px 14px', fontSize: '13px', color: 'var(--text-secondary)', maxWidth: '220px' }}>
        <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{job.desc}</div>
      </td>
      <td style={{ padding: '11px 14px' }}><StatusBadge status={job.status} /></td>
      <td style={{ padding: '11px 14px', fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>{job.created}</td>
      <td style={{ padding: '11px 14px', fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{job.duration}</td>
      <td style={{ padding: '11px 14px' }} onClick={e => e.stopPropagation()}>
        {(job.status === 'processing' || job.status === 'pending') && (
          <button onClick={onCancel} style={{
            background: 'oklch(0.65 0.20 25 / 0.1)', border: '1px solid oklch(0.65 0.20 25 / 0.3)',
            borderRadius: '5px', padding: '4px 10px', cursor: 'pointer',
            fontSize: '11px', color: 'oklch(0.65 0.20 25)', fontWeight: 600,
          }}>Cancel</button>
        )}
        {job.status === 'failed' && (
          <button style={{
            background: 'oklch(0.46 0.19 145 / 0.10)', border: '1px solid oklch(0.46 0.19 145 / 0.25)',
            borderRadius: '5px', padding: '4px 10px', cursor: 'pointer',
            fontSize: '11px', color: 'oklch(0.46 0.19 145)', fontWeight: 600,
          }}>Retry</button>
        )}
      </td>
    </tr>
  );
}

window.JobTracker = JobTracker;

}
