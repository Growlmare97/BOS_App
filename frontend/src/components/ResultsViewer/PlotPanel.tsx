import { useMemo } from 'react';
// Use factory pattern with the minified dist to avoid the 12 MB full bundle
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import type { Data, Layout, Config } from 'plotly.js';
const Plot = createPlotlyComponent(Plotly);
import type { FieldData, ColormapName, ResultField } from '../../types/analysis';

interface PlotPanelProps {
  fieldData: FieldData | null;
  colormap: ColormapName;
  showQuiver: boolean;
  field: ResultField;
  showColorbar?: boolean;
  onHover?: (x: number, y: number, value: number) => void;
}

// Map our ColormapName to Plotly colorscale names
function toPlotlyColorscale(name: ColormapName): string {
  switch (name) {
    case 'viridis':  return 'Viridis';
    case 'plasma':   return 'Plasma';
    case 'turbo':    return 'Turbo';
    case 'RdBu':     return 'RdBu';
    case 'seismic':  return 'RdBu';
    case 'inferno':  return 'Inferno';
    default:         return 'Viridis';
  }
}

export function PlotPanel({
  fieldData,
  colormap,
  showQuiver,
  field,
  showColorbar = true,
  onHover,
}: PlotPanelProps) {
  const isVelocity = field === 'u' || field === 'v' || field === 'vorticity';

  const traces = useMemo((): Data[] => {
    if (!fieldData) return [];

    const heatmap: Data = {
      type: 'heatmap',
      z: fieldData.data,
      zmin: fieldData.vmin,
      zmax: fieldData.vmax,
      colorscale: toPlotlyColorscale(colormap),
      showscale: showColorbar,
      colorbar: {
        title: {
          text: fieldData.unit,
          side: 'right',
          font: { color: '#94a3b8', size: 11, family: 'JetBrains Mono, monospace' },
        },
        tickfont: { color: '#94a3b8', size: 10, family: 'JetBrains Mono, monospace' },
        bgcolor: 'rgba(17,24,39,0.8)',
        bordercolor: '#1e293b',
        borderwidth: 1,
        outlinecolor: '#1e293b',
        thickness: 14,
        len: 0.85,
      },
      hoverongaps: false,
      hovertemplate: `x: %{x} px<br>y: %{y} px<br>value: %{z:.4f} ${fieldData.unit}<extra></extra>`,
    } as Data;

    const result: Data[] = [heatmap];

    // Quiver overlay for velocity fields
    if (showQuiver && isVelocity && fieldData.data.length > 0) {
      const rows = fieldData.shape[0];
      const cols = fieldData.shape[1];
      const step = Math.max(1, Math.floor(Math.min(rows, cols) / 20));

      const xArr: number[] = [];
      const yArr: number[] = [];

      for (let r = 0; r < rows; r += step) {
        for (let c = 0; c < cols; c += step) {
          xArr.push(c);
          yArr.push(r);
        }
      }

      const quiver: Data = {
        type: 'scatter',
        mode: 'markers',
        x: xArr,
        y: yArr,
        marker: {
          symbol: 'arrow',
          size: 8,
          color: 'rgba(59,130,246,0.7)',
        },
        hoverinfo: 'skip',
        showlegend: false,
      } as Data;

      result.push(quiver);
    }

    return result;
  }, [fieldData, colormap, showColorbar, showQuiver, isVelocity]);

  const layout = useMemo(
    (): Partial<Layout> => ({
      margin: { t: 10, b: 40, l: 50, r: showColorbar ? 80 : 10 },
      paper_bgcolor: '#0a0f1e',
      plot_bgcolor: '#0a0f1e',
      xaxis: {
        title: { text: 'x [px]', font: { color: '#64748b', size: 11 } },
        tickfont: { color: '#64748b', size: 10, family: 'JetBrains Mono, monospace' },
        gridcolor: '#1e293b',
        zerolinecolor: '#1e293b',
        showgrid: true,
        automargin: true,
        scaleanchor: 'y',
        scaleratio: 1,
      },
      yaxis: {
        title: { text: 'y [px]', font: { color: '#64748b', size: 11 } },
        tickfont: { color: '#64748b', size: 10, family: 'JetBrains Mono, monospace' },
        gridcolor: '#1e293b',
        zerolinecolor: '#1e293b',
        showgrid: true,
        automargin: true,
        autorange: 'reversed',
      },
      font: { color: '#f1f5f9' },
      dragmode: 'zoom',
    }),
    [showColorbar]
  );

  const plotConfig = useMemo(
    (): Partial<Config> => ({
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
      displaylogo: false,
      scrollZoom: true,
    }),
    []
  );

  if (!fieldData) {
    return null;
  }

  return (
    <Plot
      data={traces}
      layout={layout}
      config={plotConfig}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler
      onHover={(event: import('plotly.js').PlotHoverEvent) => {
        if (!onHover) return;
        const pt = event.points[0];
        if (pt && 'x' in pt && 'y' in pt && 'z' in pt) {
          onHover(
            pt.x as number,
            pt.y as number,
            pt.z as number
          );
        }
      }}
    />
  );
}
