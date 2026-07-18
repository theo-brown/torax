import {useEffect, useRef, useState} from 'react';
import {varDescription, varLabel} from '../catalog';
import {availableVariables} from '../data';
import {
  CONFIG_FILE_FORMAT,
  PRESETS,
  validatePanels,
  type DashboardConfigFile,
} from '../presets';
import type {PanelConfig, PanelType, RunData, SliderMode} from '../types';
import {VarLabel} from './VarLabel';

interface Props {
  panels: PanelConfig[];
  runs: RunData[];
  sliderMode: SliderMode;
  onChange: (panels: PanelConfig[]) => void;
  onSliderModeChange: (mode: SliderMode) => void;
  onClose: () => void;
}

let addedPanelCounter = 0;

const TAB_LABELS = {plots: 'Plots', slider: 'Time slider'} as const;
type Tab = keyof typeof TAB_LABELS;

const SLIDER_MODES: {mode: SliderMode; label: string; hint: string}[] = [
  {
    mode: 'plasma',
    label: 'Plasma time',
    hint: 'slider stops evenly spaced in time',
  },
  {
    mode: 'steps',
    label: 'Simulation steps',
    hint: 'slider stops at the timesteps taken by the simulation',
  },
];

export function SettingsModal({
  panels,
  runs,
  sliderMode,
  onChange,
  onSliderModeChange,
  onClose,
}: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const configInputRef = useRef<HTMLInputElement>(null);
  const [tab, setTab] = useState<Tab>('plots');
  const [configError, setConfigError] = useState<string | null>(null);

  const loadPreset = (key: string) => {
    const preset = PRESETS.find(p => p.key === key);
    if (preset) {
      onChange(preset.panels());
      setConfigError(null);
    }
  };

  const saveConfigFile = () => {
    const doc: DashboardConfigFile = {
      format: CONFIG_FILE_FORMAT,
      sliderMode,
      panels,
    };
    const blob = new Blob([JSON.stringify(doc, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'torax-plot-config.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const loadConfigFile = async (file: File) => {
    try {
      const doc = JSON.parse(await file.text()) as Partial<DashboardConfigFile>;
      if (doc?.format !== CONFIG_FILE_FORMAT) {
        throw new Error(
          `Not a dashboard config file (expected format '${CONFIG_FILE_FORMAT}').`,
        );
      }
      const loaded = validatePanels(doc.panels);
      if (!loaded) throw new Error('Invalid panel list in config file.');
      onChange(loaded);
      if (doc.sliderMode === 'plasma' || doc.sliderMode === 'steps') {
        onSliderModeChange(doc.sliderMode);
      }
      setConfigError(null);
    } catch (e) {
      setConfigError(
        `${file.name}: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  };

  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    dialogRef.current?.focus();
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const spatialVars = availableVariables(runs, 'spatial');
  const timeVars = availableVariables(runs, 'time');

  const updatePanel = (id: string, update: Partial<PanelConfig>) => {
    onChange(panels.map(p => (p.id === id ? {...p, ...update} : p)));
  };

  const toggleVariable = (panel: PanelConfig, name: string) => {
    updatePanel(panel.id, {
      variables: panel.variables.map(v =>
        v.name === name ? {...v, on: !v.on} : v,
      ),
    });
  };

  const removeVariable = (panel: PanelConfig, name: string) => {
    updatePanel(panel.id, {
      variables: panel.variables.filter(v => v.name !== name),
    });
  };

  const addVariable = (panel: PanelConfig, name: string) => {
    if (!name || panel.variables.some(v => v.name === name)) return;
    updatePanel(panel.id, {
      variables: [...panel.variables, {name, on: true}],
    });
  };

  const addPanel = (type: PanelType) => {
    const panel: PanelConfig = {
      id: `custom-${Date.now()}-${addedPanelCounter++}`,
      title: type === 'spatial' ? 'New profile panel' : 'New time-series panel',
      type,
      variables: [],
      visible: true,
      yMinZero: true,
      upperPercentile: 100,
      lowerPercentile: 0,
      skipFirstTime: false,
      suppressZero: false,
    };
    onChange([...panels, panel]);
  };

  const removePanel = (id: string) => {
    onChange(panels.filter(p => p.id !== id));
  };

  const optionText = (name: string): string => {
    const description = varDescription(name);
    return description ? `${name} — ${description}` : name;
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Plot settings"
        tabIndex={-1}
        ref={dialogRef}
        onClick={e => e.stopPropagation()}
      >
        <header className="modal-header">
          <h2>Plot settings</h2>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Close settings"
          >
            ✕
          </button>
        </header>

        <div
          className="modal-tabs"
          role="tablist"
          aria-label="Settings sections"
        >
          {(Object.keys(TAB_LABELS) as Tab[]).map(t => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              className={`modal-tab${tab === t ? ' active' : ''}`}
              onClick={() => setTab(t)}
            >
              {TAB_LABELS[t]}
            </button>
          ))}
        </div>

        {tab === 'plots' && (
          <div className="settings-toolbar">
            <select
              value=""
              onChange={e => loadPreset(e.target.value)}
              aria-label="Load preset"
            >
              <option value="" disabled>
                Load preset…
              </option>
              {PRESETS.map(preset => (
                <option key={preset.key} value={preset.key}>
                  {preset.name}
                </option>
              ))}
            </select>
            <span className="modal-footer-spacer" />
            <button className="text-button" onClick={saveConfigFile}>
              Save to file…
            </button>
            <button
              className="text-button"
              onClick={() => configInputRef.current?.click()}
            >
              Load from file…
            </button>
            <input
              ref={configInputRef}
              type="file"
              accept=".json,application/json"
              hidden
              onChange={e => {
                const file = e.target.files?.[0];
                if (file) void loadConfigFile(file);
                e.target.value = '';
              }}
            />
          </div>
        )}

        {tab === 'plots' && configError && (
          <p className="modal-note config-error" role="alert">
            {configError}
          </p>
        )}

        {tab === 'plots' && runs.length === 0 && (
          <p className="modal-note">
            Load a run to see all variables available in your data. You can
            still arrange panels now.
          </p>
        )}

        {tab === 'slider' ? (
          <div className="modal-body" role="tabpanel">
            <fieldset className="settings-slider">
              <legend>Slider stops</legend>
              {SLIDER_MODES.map(({mode, label, hint}) => (
                <label key={mode}>
                  <input
                    type="radio"
                    name="slider-mode"
                    checked={sliderMode === mode}
                    onChange={() => onSliderModeChange(mode)}
                  />
                  <span className="settings-var-label">{label}</span>
                  <span className="settings-var-name">{hint}</span>
                </label>
              ))}
            </fieldset>
            <p className="modal-note">
              The slider drives the profile panels and the time marker on
              time-series panels, using the first loaded run as the clock.
            </p>
          </div>
        ) : (
          <div className="modal-body" role="tabpanel">
            {panels.map(panel => {
              const pool = panel.type === 'spatial' ? spatialVars : timeVars;
              const addable = pool.filter(
                name => !panel.variables.some(v => v.name === name),
              );
              return (
                <details className="settings-panel" key={panel.id}>
                  <summary>
                    <input
                      type="checkbox"
                      checked={panel.visible}
                      onChange={e =>
                        updatePanel(panel.id, {visible: e.target.checked})
                      }
                      onClick={e => e.stopPropagation()}
                      aria-label={`Show panel ${panel.title}`}
                    />
                    <span className="settings-panel-title">
                      <VarLabel label={panel.title} />
                    </span>
                    <span className="settings-panel-meta">
                      {panel.type === 'spatial' ? 'profile' : 'time series'} ·{' '}
                      {panel.variables.filter(v => v.on).length} shown
                    </span>
                  </summary>
                  <div className="settings-panel-body">
                    <label className="settings-field">
                      Title
                      <input
                        type="text"
                        value={panel.title}
                        onChange={e =>
                          updatePanel(panel.id, {title: e.target.value})
                        }
                      />
                    </label>
                    <ul className="settings-var-list">
                      {panel.variables.map(variable => (
                        <li key={variable.name}>
                          <label>
                            <input
                              type="checkbox"
                              checked={variable.on}
                              onChange={() =>
                                toggleVariable(panel, variable.name)
                              }
                            />
                            <span className="settings-var-label">
                              <VarLabel label={varLabel(variable.name)} />
                            </span>
                            <span className="settings-var-name">
                              {variable.name}
                              {varDescription(variable.name)
                                ? ` — ${varDescription(variable.name)}`
                                : ''}
                            </span>
                          </label>
                          <button
                            className="icon-button small"
                            onClick={() => removeVariable(panel, variable.name)}
                            aria-label={`Remove ${variable.name}`}
                            title="Remove from panel"
                          >
                            ✕
                          </button>
                        </li>
                      ))}
                    </ul>
                    <div className="settings-actions">
                      <select
                        value=""
                        onChange={e => addVariable(panel, e.target.value)}
                        disabled={addable.length === 0}
                        aria-label={`Add variable to ${panel.title}`}
                      >
                        <option value="" disabled>
                          {addable.length === 0
                            ? runs.length === 0
                              ? 'Load a run to add variables'
                              : 'No more variables'
                            : 'Add variable…'}
                        </option>
                        {addable.map(name => (
                          <option key={name} value={name}>
                            {optionText(name)}
                          </option>
                        ))}
                      </select>
                      <button
                        className="text-button danger"
                        onClick={() => removePanel(panel.id)}
                      >
                        Remove panel
                      </button>
                    </div>
                  </div>
                </details>
              );
            })}
          </div>
        )}

        <footer className="modal-footer">
          {tab === 'plots' && (
            <>
              <button
                className="text-button"
                onClick={() => addPanel('spatial')}
              >
                + Profile panel
              </button>
              <button className="text-button" onClick={() => addPanel('time')}>
                + Time-series panel
              </button>
              <span className="modal-footer-spacer" />
              <button
                className="text-button"
                onClick={() => loadPreset('default')}
              >
                Reset to default
              </button>
            </>
          )}
          {tab === 'slider' && <span className="modal-footer-spacer" />}
          <button className="primary-button" onClick={onClose}>
            Done
          </button>
        </footer>
      </div>
    </div>
  );
}
