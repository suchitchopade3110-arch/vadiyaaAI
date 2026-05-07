{

// report-qr-widget.jsx - QR card for report/image result screens

const QR_API_BASE = window.location.origin;

function ReportQRWidget({ reportId, patientId }) {
  const { useEffect, useState } = React;
  const [qrUrl, setQrUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [refreshCount, setRefreshCount] = useState(0);

  useEffect(() => {
    if (!reportId) return undefined;

    let objectUrl = null;
    setLoading(true);
    setError('');
    setQrUrl(null);

    const params = patientId ? `?patient_id=${encodeURIComponent(patientId)}` : '';
    fetch(`${QR_API_BASE}/reports/${encodeURIComponent(reportId)}/qr${params}`)
      .then((response) => {
        if (!response.ok) throw new Error(`QR generation failed (${response.status})`);
        return response.blob();
      })
      .then((blob) => {
        objectUrl = URL.createObjectURL(blob);
        setQrUrl(objectUrl);
      })
      .catch((err) => setError(err.message || 'QR generation failed'))
      .finally(() => setLoading(false));

    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [reportId, patientId, refreshCount]);

  return (
    <div style={styles.wrapper}>
      <div style={styles.header}>
        <div>
          <div style={styles.title}>Download QR</div>
          <div style={styles.subtitle}>Preview first · 30 min · single use</div>
        </div>
        <button type="button" onClick={() => setRefreshCount((n) => n + 1)} style={styles.iconButton} title="Generate a new QR code">
          ↺
        </button>
      </div>

      <div style={styles.qrBox}>
        {loading && <div style={styles.placeholder}>Generating...</div>}
        {error && <div style={styles.error}>{error}</div>}
        {qrUrl && !loading && !error && <img src={qrUrl} alt="Report download QR code" style={styles.qrImg} />}
      </div>
    </div>
  );
}

const styles = {
  wrapper: {
    background: 'var(--bg-surface)',
    border: '1px solid var(--border)',
    borderRadius: '8px',
    padding: '16px',
    width: '240px',
    flexShrink: 0,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '10px',
    marginBottom: '12px',
  },
  title: {
    fontSize: '13px',
    fontWeight: 800,
    color: 'var(--text-primary)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  subtitle: {
    fontSize: '11px',
    color: 'var(--text-muted)',
    marginTop: '3px',
  },
  iconButton: {
    width: '30px',
    height: '30px',
    borderRadius: '6px',
    border: '1px solid var(--border)',
    background: 'var(--bg-elevated)',
    color: 'var(--text-secondary)',
    cursor: 'pointer',
    fontSize: '15px',
  },
  qrBox: {
    width: '204px',
    height: '204px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#fff',
    borderRadius: '8px',
    padding: '8px',
    overflow: 'hidden',
  },
  qrImg: {
    width: '100%',
    height: '100%',
    objectFit: 'contain',
  },
  placeholder: {
    color: '#64748b',
    fontSize: '12px',
  },
  error: {
    color: '#dc2626',
    fontSize: '12px',
    lineHeight: 1.4,
    textAlign: 'center',
  },
};

window.ReportQRWidget = ReportQRWidget;

}
