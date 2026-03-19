import { useEffect, useRef, useState } from 'react';
import { useAnalysisStore } from '../store/analysisStore';
import type { ProgressMessage } from '../types/analysis';

export function useWebSocket(jobId: string | null): { connected: boolean } {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const setProgress = useAnalysisStore((s) => s.setProgress);
  const setError = useAnalysisStore((s) => s.setError);

  useEffect(() => {
    if (!jobId) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
        setConnected(false);
      }
      return;
    }

    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${wsProto}//${window.location.host}/ws/${jobId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(event.data) as ProgressMessage;
        setProgress(msg.stage, msg.progress, msg.message);
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection error');
      setConnected(false);
    };

    ws.onclose = () => {
      setConnected(false);
    };

    return () => {
      ws.close();
      wsRef.current = null;
      setConnected(false);
    };
  }, [jobId, setProgress, setError]);

  return { connected };
}
