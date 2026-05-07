{

// dashboard.jsx — Dashboard page component (wired to real API)

const { useState, useEffect } = React;

const API_BASE = `${window.location.origin}/api/v1`;

const PIPELINE_ICONS = { claim: '◎', report: '▤', image: '⬡' };
const PIPELINE_LABELS = { claim: 'Claim', report: 'Report', image: 'Image' };

// Map backend status values to frontend display statuses
function mapStatus(pipeline, statusVal) {
  if (!statusVal) return 'pending';
  const s = statusVal.toLowerCase();
  if (s === 'verified' || s === 'success') return 'verified';
  if (s === 'processing' || s === 'started') return 'processing';
  if (s === 'pending') return 'pending';
  if (s === 'refuted' || s === 'contradicted') return 'refuted';
  if (s === 'uncertain' || s === 'insufficient_evidence') return 'uncertain';
  if (s === 'failed' || s === 'failure') return 'failed';
  return 'pending';
}

function StatCard({ icon, label, value, color, sub }) {
  return (
    <div style={{
      background: 'var(--bg-surface)', border: '1px solid var(--border)',
      borderRadius: '10px', padding: '18px 20px', flex: 1, minWidth: 0,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '8px' }}>{label}</div>
          <div style={{ fontSize: '28px', fontWeight: 800, color: color || 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{value}</div>
          {sub && <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>{sub}</div>}
        </div>
        <div style={{
          width: '36px', height: '36px', borderRadius: '8px',
          background: `${color || 'oklch(0.46 0.19 145)'}18`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '18px', color: color || 'oklch(0.46 0.19 145)',
        }}>{icon}</div>
      </div>
    </div>
  );
}

function QuickAction({ icon, label, sub, onClick }) {
  const [hov, setHov] = useState(false);
  return (
    <button onClick={onClick}
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{
        background: hov ? 'var(--bg-hover)' : 'var(--bg-surface)',
        border: `1px solid ${hov ? 'oklch(0.46 0.19 145 / 0.35)' : 'var(--border)'}`,
        borderRadius: '10px', padding: '16px', cursor: 'pointer',
        textAlign: 'left', flex: 1, transition: 'all 0.18s ease',
        transform: hov ? 'translateY(-1px)' : 'none',
      }}>
      <div style={{ fontSize: '20px', marginBottom: '8px' }}>{icon}</div>
      <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>{label}</div>
      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '3px' }}>{sub}</div>
    </button>
  );
}

function Dashboard({ onNavigate }) {
  const [health, setHealth] = useState('checking');
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Real health check
  useEffect(() => {
    fetch(`${window.location.origin}/health`)
      .then(res => res.json())
      .then(data => setHealth(data.status === 'ok' ? 'online' : 'offline'))
      .catch(() => setHealth('offline'));
  }, []);

  // Fetch real jobs from the API
  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/jobs?limit=20`)
      .then(res => {
        if (!res.ok) throw new Error(`API returned ${res.status}`);
        return res.json();
      })
      .then(data => {
        const mapped = (data.jobs || []).map(j => ({
          id: j.job_id ? j.job_id.slice(0, 8) : '—',
          fullId: j.job_id,
          type: j.pipeline.includes('/') ? j.pipeline.split('/')[0] : j.pipeline,
          status: mapStatus(j.pipeline, j.status),
          created: j.created_at ? new Date(j.created_at).toLocaleString() : '—',
          desc: `${j.pipeline} analysis`,
          celeryId: j.celery_task_id,
        }));
        setJobs(mapped);
        setLoading(false);
      })
      .catch(err => {
        console.warn('Jobs API unavailable, no jobs to show:', err.message);
        setJobs([]);
        setLoading(false);
      });
  }, []);

  const counts = {
    total:      jobs.length,
    verified:   jobs.filter(j => j.status === 'verified').length,
    processing: jobs.filter(j => j.status === 'processing').length,
    failed:     jobs.filter(j => j.status === 'failed').length,
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>Dashboard</h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', margin: '4px 0 0' }}>Overview of your AI medical analysis pipeline</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: '8px', height: '8px', borderRadius: '50%',
            background: health === 'online' ? 'oklch(0.46 0.19 145)' : health === 'offline' ? 'oklch(0.65 0.20 25)' : 'oklch(0.75 0.14 60)',
            boxShadow: health === 'online' ? '0 0 8px oklch(0.46 0.19 145)' : 'none',
            animation: health === 'checking' ? 'pulse-dot 1.2s ease infinite' : 'none',
          }} />
          <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 600 }}>
            API {health === 'checking' ? 'Checking…' : health === 'online' ? 'Online' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Stats row */}
      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
        <StatCard icon="◎" label="Total Jobs" value={counts.total} sub="from database" />
        <StatCard icon="✓" label="Verified" value={counts.verified} color="oklch(0.46 0.19 145)" sub="claims & reports" />
        <StatCard icon="↻" label="Processing" value={counts.processing} color="oklch(0.46 0.19 145)" sub="in pipeline" />
        <StatCard icon="!" label="Failed" value={counts.failed} color="oklch(0.65 0.20 25)" sub="need attention" />
      </div>

      {/* Quick actions */}
      <div>
        <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '12px' }}>Start New Analysis</div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <QuickAction icon="◎" label="Verify a Claim" sub="Evidence-based fact checking" onClick={() => onNavigate('claim')} />
          <QuickAction icon="▤" label="Analyze Report" sub="Lab, clinical, discharge" onClick={() => onNavigate('report')} />
          <QuickAction icon="⬡" label="Analyze Image" sub="X-ray, CT, MRI, Pathology" onClick={() => onNavigate('image')} />
        </div>
      </div>

      {/* Recent jobs table */}
      <div>
        <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '12px' }}>
          Recent Jobs {loading && <span style={{ fontWeight: 400, animation: 'fade-in-out 1.4s ease infinite' }}>loading…</span>}
        </div>
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: '10px', overflow: 'hidden',
        }}>
          {jobs.length === 0 && !loading ? (
            <div style={{ padding: '32px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', opacity: 0.3, marginBottom: '8px' }}>◫</div>
              <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>No jobs yet. Submit a claim, report, or image to get started.</div>
            </div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Job ID', 'Pipeline', 'Description', 'Status', 'Created', ''].map((h, i) => (
                    <th key={i} style={{
                      padding: '11px 16px', textAlign: 'left',
                      fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)',
                      textTransform: 'uppercase', letterSpacing: '0.06em',
                      background: 'var(--bg-elevated)',
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {jobs.map((job, i) => (
                  <JobRow key={job.fullId || i} job={job} last={i === jobs.length - 1} onNavigate={onNavigate} />
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <MedicalDisclaimer />
    </div>
  );
}

function JobRow({ job, last, onNavigate }) {
  const [hov, setHov] = useState(false);
  const pageMap = { claim: 'claim', report: 'report', image: 'image' };
  return (
    <tr
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{
        borderBottom: last ? 'none' : '1px solid var(--border)',
        background: hov ? 'var(--bg-elevated)' : 'transparent',
        transition: 'background 0.15s',
      }}>
      <td style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'oklch(0.46 0.19 145)', fontWeight: 600 }}>{job.id}</td>
      <td style={{ padding: '12px 16px' }}>
        <span style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span>{PIPELINE_ICONS[job.type] || '?'}</span>
          <span style={{ fontWeight: 600 }}>{PIPELINE_LABELS[job.type] || job.type}</span>
        </span>
      </td>
      <td style={{ padding: '12px 16px', fontSize: '13px', color: 'var(--text-secondary)', maxWidth: '240px' }}>
        <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{job.desc}</div>
      </td>
      <td style={{ padding: '12px 16px' }}><StatusBadge status={job.status} /></td>
      <td style={{ padding: '12px 16px', fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{job.created}</td>
      <td style={{ padding: '12px 16px' }}>
        <button onClick={() => onNavigate(pageMap[job.type] || 'jobs')}
          style={{
            background: 'transparent', border: '1px solid var(--border)',
            borderRadius: '5px', padding: '5px 10px', cursor: 'pointer',
            fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 600,
            transition: 'all 0.15s',
          }}>
          View →
        </button>
      </td>
    </tr>
  );
}

window.Dashboard = Dashboard;

}
