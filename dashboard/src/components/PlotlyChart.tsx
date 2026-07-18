import Plotly from 'plotly.js-basic-dist-min';
import {useEffect, useRef} from 'react';

interface Props {
  data: unknown[];
  layout: Record<string, unknown>;
}

const CONFIG = {displayModeBar: false, responsive: true};

/** Thin React wrapper around Plotly.react. */
export function PlotlyChart({data, layout}: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) void Plotly.react(ref.current, data, layout, CONFIG);
  }, [data, layout]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return <div ref={ref} className="plotly-chart" />;
}
