import React from 'react';
import { FileLoader } from './components/FileLoader/FileLoader';
import { ProcessingConfig } from './components/ProcessingConfig/ProcessingConfig';
import { ResultsViewer } from './components/ResultsViewer/ResultsViewer';
import { StatusBar } from './components/StatusBar/StatusBar';

// ── Error Boundary ────────────────────────────────────────────────────────────
interface EBState { hasError: boolean; error: Error | null }
class ErrorBoundary extends React.Component<React.PropsWithChildren, EBState> {
  state: EBState = { hasError: false, error: null };
  static getDerivedStateFromError(error: Error): EBState {
    return { hasError: true, error };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 32, color: '#ef4444', fontFamily: 'monospace', background: '#0a0f1e', minHeight: '100vh' }}>
          <h2 style={{ color: '#f59e0b', marginBottom: 12 }}>⚠ Render Error</h2>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: 13 }}>{this.state.error?.message}</pre>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: 11, color: '#64748b', marginTop: 8 }}>{this.state.error?.stack}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

const SIDEBAR_WIDTH = 280;

const appStyles: Record<string, React.CSSProperties> = {
  root: {
    display: 'grid',
    gridTemplateRows: '1fr 40px',
    gridTemplateColumns: `${SIDEBAR_WIDTH}px 1fr`,
    gridTemplateAreas: `
      "sidebar main"
      "statusbar statusbar"
    `,
    height: '100vh',
    width: '100vw',
    overflow: 'hidden',
    background: '#0a0f1e',
  },
  sidebar: {
    gridArea: 'sidebar',
    display: 'flex',
    flexDirection: 'column',
    background: '#111827',
    borderRight: '1px solid #1e293b',
    overflow: 'hidden',
  },
  sidebarInner: {
    flex: 1,
    overflowY: 'auto',
    overflowX: 'hidden',
    padding: '12px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  sidebarSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  sidebarSectionTitle: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.6875rem',
    fontWeight: 700,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.1em',
    color: '#64748b',
    padding: '0 2px',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  sidebarDivider: {
    height: '1px',
    background: '#1e293b',
    margin: '2px 0',
  },
  main: {
    gridArea: 'main',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    background: '#0a0f1e',
    minWidth: 0,
  },
  mainHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 16px',
    background: '#111827',
    borderBottom: '1px solid #1e293b',
    flexShrink: 0,
  },
  mainHeaderTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  appLogo: {
    width: '24px',
    height: '24px',
    color: '#3b82f6',
  },
  appName: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.9375rem',
    fontWeight: 600,
    color: '#f1f5f9',
    letterSpacing: '0.04em',
  },
  appSubtitle: {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.75rem',
    color: '#64748b',
  },
  mainContent: {
    flex: 1,
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    minHeight: 0,
  },
  statusBar: {
    gridArea: 'statusbar',
    display: 'flex',
    height: '40px',
  },
};

export function App() {
  return (
    <ErrorBoundary>
    <div style={appStyles.root}>
      {/* ── Left Sidebar ── */}
      <aside style={appStyles.sidebar}>
        <div style={appStyles.sidebarInner}>
          {/* File Loader section */}
          <div style={appStyles.sidebarSection}>
            <div style={appStyles.sidebarSectionTitle}>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
                <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
              </svg>
              Data Source
            </div>
            <FileLoader />
          </div>

          <div style={appStyles.sidebarDivider} />

          {/* Processing Config section */}
          <div style={appStyles.sidebarSection}>
            <div style={appStyles.sidebarSectionTitle}>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
                <circle cx="12" cy="12" r="3" />
                <path d="M19.07 4.93A10 10 0 1 0 4.93 19.07M19.07 4.93l-2.12 2.12" />
              </svg>
              Pipeline Config
            </div>
            <ProcessingConfig />
          </div>
        </div>
      </aside>

      {/* ── Main Results Area ── */}
      <main style={appStyles.main}>
        {/* Header */}
        <div style={appStyles.mainHeader}>
          <div style={appStyles.mainHeaderTitle}>
            <svg
              style={appStyles.appLogo}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <circle cx="12" cy="12" r="10" />
              <circle cx="12" cy="12" r="4" />
              <line x1="12" y1="2" x2="12" y2="8" />
              <line x1="12" y1="16" x2="12" y2="22" />
              <line x1="2" y1="12" x2="8" y2="12" />
              <line x1="16" y1="12" x2="22" y2="12" />
            </svg>
            <span style={appStyles.appName}>BOS Analysis</span>
            <span style={{ color: '#1e293b', fontSize: '1rem' }}>|</span>
            <span style={appStyles.appSubtitle}>
              Background-Oriented Schlieren · Scientific Analysis Suite
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.7rem',
                color: '#64748b',
                padding: '2px 8px',
                border: '1px solid #1e293b',
                borderRadius: '4px',
              }}
            >
              v0.1.0
            </span>
          </div>
        </div>

        {/* Results viewer fills remaining space */}
        <div style={appStyles.mainContent}>
          <ResultsViewer />
        </div>
      </main>

      {/* ── Status Bar ── */}
      <div style={appStyles.statusBar}>
        <StatusBar />
      </div>
    </div>
    </ErrorBoundary>
  );
}
