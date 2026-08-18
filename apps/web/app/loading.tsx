export default function Loading() {
  return (
    <div
      style={{
        minHeight: '60vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'transparent',
        fontFamily: "'Inter', 'Segoe UI', sans-serif",
        gap: 16,
      }}
    >
      <div
        style={{
          width: 48,
          height: 48,
          borderRadius: '50%',
          border: '3px solid rgba(96,165,250,0.2)',
          borderTop: '3px solid #60a5fa',
          animation: 'spin 0.8s linear infinite',
        }}
      />
      <p style={{ margin: 0, color: '#64748b', fontSize: 14 }}>Loading...</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
