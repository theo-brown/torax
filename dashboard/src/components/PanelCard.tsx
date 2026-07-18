import {useMemo} from 'react';
import {varLabel} from '../catalog';
import {getSeriesData, isAllZero, nearestTimeIndex} from '../data';
import {escapeHtml, labelToPlotlyHtml} from '../latex';
import {
  CHROME,
  FONT_FAMILY,
  SERIES_COLORS,
  runDashStyle,
  useTheme,
} from '../palette';
import {computeYDomain} from '../scale';
import type {PanelConfig, RunData} from '../types';
import {PlotlyChart} from './PlotlyChart';
import {VarLabel} from './VarLabel';

const SERIES_SLOTS = 8;
const CHART_HEIGHT = 230;

interface Props {
  panel: PanelConfig;
  runs: RunData[];
  /** Current slider time (master run's clock), or null with no runs. */
  masterTime: number | null;
}

export function PanelCard({panel, runs, masterTime}: Props) {
  const theme = useTheme();
  const chrome = CHROME[theme];
  const seriesColors = SERIES_COLORS[theme];

  const yDomain = useMemo(() => computeYDomain(runs, panel), [runs, panel]);

  const {traces, legendVars} = useMemo(() => {
    const traces: Record<string, unknown>[] = [];
    const legendVars: {name: string; slot: number}[] = [];
    panel.variables.forEach((variable, varIdx) => {
      if (!variable.on) return;
      const slot = varIdx % SERIES_SLOTS;
      let plotted = false;
      runs.forEach((run, runIdx) => {
        if (panel.suppressZero && isAllZero(run, panel.type, variable.name))
          return;
        const timeIdx =
          panel.type === 'spatial' && masterTime != null
            ? nearestTimeIndex(run, masterTime)
            : 0;
        const data = getSeriesData(run, panel.type, variable.name, timeIdx);
        if (!data) return;
        plotted = true;
        const name =
          labelToPlotlyHtml(varLabel(variable.name)) +
          (runs.length > 1 ? ` (${escapeHtml(run.label)})` : '');
        traces.push({
          x: data.x,
          y: data.y,
          type: 'scatter',
          mode: 'lines',
          name,
          line: {
            color: seriesColors[slot],
            width: 2,
            dash: runDashStyle(runIdx).plotly,
          },
          hovertemplate: '%{y:.4g}',
        });
      });
      if (plotted) legendVars.push({name: variable.name, slot});
    });
    return {traces, legendVars};
  }, [panel, runs, masterTime, seriesColors]);

  const layout = useMemo(() => {
    const isSpatial = panel.type === 'spatial';
    let xRange: [number, number] = [0, 1];
    if (!isSpatial) {
      let t0 = Infinity;
      let t1 = -Infinity;
      for (const run of runs) {
        t0 = Math.min(t0, run.time[0]);
        t1 = Math.max(t1, run.time[run.time.length - 1]);
      }
      if (isFinite(t0) && t0 !== t1) xRange = [t0, t1];
    }
    const tickFont = {family: FONT_FAMILY, size: 10, color: chrome.textMuted};
    return {
      height: CHART_HEIGHT,
      margin: {l: 50, r: 10, t: 8, b: 28},
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {family: FONT_FAMILY, size: 11, color: chrome.textSecondary},
      showlegend: false,
      hovermode: 'x unified',
      hoverlabel: {
        bgcolor: chrome.surface,
        bordercolor: chrome.border,
        font: {family: FONT_FAMILY, size: 11, color: chrome.textPrimary},
      },
      xaxis: {
        range: xRange,
        ...(isSpatial ? {tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1.0]} : {nticks: 6}),
        showgrid: false,
        zeroline: false,
        showline: true,
        linecolor: chrome.baseline,
        linewidth: 1,
        ticks: 'outside',
        tickcolor: chrome.baseline,
        tickfont: tickFont,
      },
      yaxis: {
        range: yDomain ?? undefined,
        showgrid: true,
        gridcolor: chrome.grid,
        gridwidth: 1,
        zeroline: false,
        showline: false,
        tickfont: tickFont,
      },
      shapes:
        !isSpatial && masterTime != null
          ? [
              {
                type: 'line',
                x0: masterTime,
                x1: masterTime,
                yref: 'paper',
                y0: 0,
                y1: 1,
                line: {color: chrome.textMuted, width: 1, dash: 'dot'},
              },
            ]
          : [],
    };
  }, [panel.type, runs, yDomain, masterTime, chrome]);

  // Legend chips: one per plotted variable; a single variable in a single run
  // needs no legend (the title already names it).
  const showLegend = legendVars.length > 1;

  return (
    <section className="panel-card">
      <header className="panel-header">
        <h3 className="panel-title">
          <VarLabel label={panel.title} />
          <span className="panel-axis-hint">
            {' vs '}
            <VarLabel label={panel.type === 'spatial' ? '$\\rho$' : '$t$'} />
          </span>
        </h3>
        {showLegend && (
          <div className="legend" role="list">
            {legendVars.map(item => (
              <span className="legend-chip" role="listitem" key={item.name}>
                <svg width="14" height="4" aria-hidden="true">
                  <line
                    x1="0"
                    x2="14"
                    y1="2"
                    y2="2"
                    stroke={`var(--series-${item.slot + 1})`}
                    strokeWidth={2}
                  />
                </svg>
                <VarLabel label={varLabel(item.name)} />
              </span>
            ))}
          </div>
        )}
      </header>
      {traces.length > 0 ? (
        <PlotlyChart data={traces} layout={layout} />
      ) : (
        <div className="chart-empty" style={{height: CHART_HEIGHT}}>
          No data for this panel
        </div>
      )}
    </section>
  );
}
