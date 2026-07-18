/** Resolved palette values for Plotly (which can't consume CSS variables),
 *  plus a hook tracking the active theme. Values match styles.css. */

import {useEffect, useState} from 'react';

export type ThemeName = 'light' | 'dark';

export const FONT_FAMILY =
  "'Roboto', system-ui, -apple-system, 'Segoe UI', sans-serif";

export const SERIES_COLORS: Record<ThemeName, string[]> = {
  light: [
    '#2a78d6',
    '#008300',
    '#e87ba4',
    '#eda100',
    '#1baf7a',
    '#eb6834',
    '#4a3aa7',
    '#e34948',
  ],
  dark: [
    '#3987e5',
    '#008300',
    '#d55181',
    '#c98500',
    '#199e70',
    '#d95926',
    '#9085e9',
    '#e66767',
  ],
};

export interface Chrome {
  surface: string;
  textPrimary: string;
  textSecondary: string;
  textMuted: string;
  grid: string;
  baseline: string;
  border: string;
}

export const CHROME: Record<ThemeName, Chrome> = {
  light: {
    surface: '#fcfcfb',
    textPrimary: '#0b0b0b',
    textSecondary: '#52514e',
    textMuted: '#898781',
    grid: '#e1e0d9',
    baseline: '#c3c2b7',
    border: 'rgba(11, 11, 11, 0.1)',
  },
  dark: {
    surface: '#1a1a19',
    textPrimary: '#ffffff',
    textSecondary: '#c3c2b7',
    textMuted: '#898781',
    grid: '#2c2c2a',
    baseline: '#383835',
    border: 'rgba(255, 255, 255, 0.1)',
  },
};

/** Dash patterns cycled across runs, matching plotruns_lib._DASH_PATTERNS.
 *  Variable identity stays in color; the run is carried by the dash. Each
 *  entry pairs the Plotly dash name with the equivalent SVG dash array used
 *  by the run-chip and legend keys, so the two renderings never drift. */
export const RUN_DASH_STYLES: readonly {
  plotly: string;
  svgDashArray: string;
}[] = [
  {plotly: 'solid', svgDashArray: ''},
  {plotly: 'dash', svgDashArray: '6 4'},
  {plotly: 'dot', svgDashArray: '2 3'},
  {plotly: 'dashdot', svgDashArray: '9 4 2 4'},
  {plotly: 'longdash', svgDashArray: '13 4'},
  {plotly: 'longdashdot', svgDashArray: '13 4 2 4'},
];

export function runDashStyle(runIndex: number) {
  return RUN_DASH_STYLES[runIndex % RUN_DASH_STYLES.length];
}

function currentTheme(): ThemeName {
  const override = document.documentElement.dataset.theme;
  if (override === 'dark' || override === 'light') return override;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/** Active theme, reacting to both the OS setting and the app's toggle. */
export function useTheme(): ThemeName {
  const [theme, setTheme] = useState<ThemeName>(currentTheme);
  useEffect(() => {
    const update = () => setTheme(currentTheme());
    const mq = matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', update);
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });
    return () => {
      mq.removeEventListener('change', update);
      observer.disconnect();
    };
  }, []);
  return theme;
}
