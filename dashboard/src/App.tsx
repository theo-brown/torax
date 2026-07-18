import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from 'react';
import {parseRun} from './data';
import {runDashStyle} from './palette';
import {defaultPanels, validatePanels} from './presets';
import {PanelCard} from './components/PanelCard';
import {SettingsModal} from './components/SettingsModal';
import {TimeSlider} from './components/TimeSlider';
import type {PanelConfig, RunData, SliderMode} from './types';

const PANELS_STORAGE_KEY = 'torax-dashboard-panels-v1';
const THEME_STORAGE_KEY = 'torax-dashboard-theme';
const SLIDER_STORAGE_KEY = 'torax-dashboard-slider-mode';

type Theme = 'auto' | 'light' | 'dark';

declare global {
  interface Window {
    /** Exported run documents embedded in the page (e.g. a shared artifact
     *  build); loaded automatically on startup. */
    TORAX_EMBEDDED_RUNS?: unknown[];
  }
}

/** localStorage access can throw in sandboxed contexts; degrade gracefully. */
function storageGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function storageSet(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Storage unavailable: settings simply don't persist.
  }
}

function loadStoredPanels(): PanelConfig[] {
  try {
    const raw = storageGet(PANELS_STORAGE_KEY);
    if (!raw) return defaultPanels();
    return validatePanels(JSON.parse(raw)) ?? defaultPanels();
  } catch {
    return defaultPanels();
  }
}

export function App() {
  const [runs, setRuns] = useState<RunData[]>([]);
  const [panels, setPanels] = useState<PanelConfig[]>(loadStoredPanels);
  const [timeIndex, setTimeIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [theme, setTheme] = useState<Theme>(
    () => (storageGet(THEME_STORAGE_KEY) as Theme) || 'auto',
  );
  const [sliderMode, setSliderMode] = useState<SliderMode>(() =>
    storageGet(SLIDER_STORAGE_KEY) === 'steps' ? 'steps' : 'plasma',
  );
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    storageSet(SLIDER_STORAGE_KEY, sliderMode);
  }, [sliderMode]);

  // Load any runs embedded in the page itself.
  useEffect(() => {
    const docs = window.TORAX_EMBEDDED_RUNS;
    if (!docs || docs.length === 0) return;
    const loaded: RunData[] = [];
    for (const doc of docs) {
      try {
        loaded.push(parseRun(doc, 'embedded run'));
      } catch {
        // Ignore malformed embedded documents.
      }
    }
    if (loaded.length > 0) setRuns(prev => (prev.length > 0 ? prev : loaded));
  }, []);

  // Persist panel configuration.
  useEffect(() => {
    storageSet(PANELS_STORAGE_KEY, JSON.stringify(panels));
  }, [panels]);

  // Apply theme override.
  useEffect(() => {
    storageSet(THEME_STORAGE_KEY, theme);
    if (theme === 'auto') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
  }, [theme]);

  // Slider stop values, using the first run as the master clock (as in
  // plotruns_lib): either the simulation's own timesteps, or the same number
  // of stops spaced linearly in plasma time.
  const masterTimeArray = runs[0]?.time ?? [];
  const sliderTimes = useMemo(() => {
    if (sliderMode === 'steps' || masterTimeArray.length < 2) {
      return masterTimeArray;
    }
    const t0 = masterTimeArray[0];
    const t1 = masterTimeArray[masterTimeArray.length - 1];
    const n = masterTimeArray.length;
    return Array.from({length: n}, (_, i) => t0 + ((t1 - t0) * i) / (n - 1));
  }, [sliderMode, masterTimeArray]);
  const clampedIndex = Math.min(timeIndex, Math.max(0, sliderTimes.length - 1));
  const masterTime = sliderTimes.length > 0 ? sliderTimes[clampedIndex] : null;

  // Playback.
  useEffect(() => {
    if (!playing || sliderTimes.length < 2) return;
    const id = setInterval(() => {
      setTimeIndex(i => (i + 1) % sliderTimes.length);
    }, 150);
    return () => clearInterval(id);
  }, [playing, sliderTimes.length]);

  const loadFiles = useCallback(async (files: Iterable<File>) => {
    const errors: string[] = [];
    const loaded: RunData[] = [];
    for (const file of files) {
      try {
        const text = await file.text();
        loaded.push(
          parseRun(JSON.parse(text), file.name.replace(/\.json$/i, '')),
        );
      } catch (e) {
        errors.push(
          `${file.name}: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    }
    if (loaded.length > 0) {
      setRuns(prev => [...prev, ...loaded]);
      setTimeIndex(0);
    }
    setError(errors.length > 0 ? errors.join(' · ') : null);
  }, []);

  const onFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) void loadFiles(e.target.files);
    e.target.value = '';
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    void loadFiles(Array.from(e.dataTransfer.files));
  };

  const removeRun = (id: string) => {
    setRuns(prev => prev.filter(r => r.id !== id));
  };

  const visiblePanels = useMemo(() => panels.filter(p => p.visible), [panels]);
  const hasSpatial = visiblePanels.some(p => p.type === 'spatial');

  return (
    <div
      className={`app${dragging ? ' dragging' : ''}`}
      onDragOver={e => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={e => {
        if (e.target === e.currentTarget) setDragging(false);
      }}
      onDrop={onDrop}
    >
      <header className="app-header">
        <h1>TORAX dashboard</h1>
        <div className="header-actions">
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            multiple
            onChange={onFileInput}
            hidden
          />
          <button
            className="text-button"
            onClick={() => fileInputRef.current?.click()}
          >
            Open runs…
          </button>
          <button
            className="text-button"
            onClick={() => setSettingsOpen(true)}
            aria-haspopup="dialog"
          >
            ⚙ Settings
          </button>
          <button
            className="icon-button"
            onClick={() =>
              setTheme(t =>
                t === 'auto' ? 'dark' : t === 'dark' ? 'light' : 'auto',
              )
            }
            title={`Theme: ${theme} (click to change)`}
            aria-label={`Theme: ${theme}. Click to change.`}
          >
            {theme === 'auto' ? '◐' : theme === 'dark' ? '●' : '○'}
          </button>
        </div>
      </header>

      {error && (
        <div className="error-banner" role="alert">
          {error}
          <button
            className="icon-button small"
            onClick={() => setError(null)}
            aria-label="Dismiss"
          >
            ✕
          </button>
        </div>
      )}

      {runs.length > 0 && (
        <div className="run-chips">
          {runs.map((run, idx) => (
            <span className="run-chip" key={run.id}>
              <svg width="22" height="4" aria-hidden="true">
                <line
                  x1="0"
                  x2="22"
                  y1="2"
                  y2="2"
                  stroke="currentColor"
                  strokeWidth={2}
                  strokeDasharray={runDashStyle(idx).svgDashArray || undefined}
                />
              </svg>
              {run.label}
              <button
                className="icon-button small"
                onClick={() => removeRun(run.id)}
                aria-label={`Remove run ${run.label}`}
              >
                ✕
              </button>
            </span>
          ))}
        </div>
      )}

      {runs.length === 0 ? (
        <div className="empty-state">
          <h2>No runs loaded</h2>
          <p>
            Drop exported TORAX run files here, or use{' '}
            <strong>Open runs…</strong> above. Load several files to compare
            runs.
          </p>
          <p className="empty-hint">
            Export a run with:{' '}
            <code>plot_torax --outfile state_history.nc --export_json</code>
          </p>
          <button
            className="primary-button"
            onClick={() => fileInputRef.current?.click()}
          >
            Open runs…
          </button>
        </div>
      ) : (
        <main className="panel-grid">
          {visiblePanels.map(panel => (
            <PanelCard
              key={panel.id}
              panel={panel}
              runs={runs}
              masterTime={masterTime}
            />
          ))}
        </main>
      )}

      {runs.length > 0 && hasSpatial && sliderTimes.length > 1 && (
        <TimeSlider
          time={sliderTimes}
          index={clampedIndex}
          playing={playing}
          showStep={sliderMode === 'steps'}
          onChange={i => {
            setPlaying(false);
            setTimeIndex(i);
          }}
          onPlayToggle={() => setPlaying(p => !p)}
        />
      )}

      {settingsOpen && (
        <SettingsModal
          panels={panels}
          runs={runs}
          sliderMode={sliderMode}
          onChange={setPanels}
          onSliderModeChange={setSliderMode}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </div>
  );
}
