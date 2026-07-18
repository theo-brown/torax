import {useMemo} from 'react';

interface Props {
  /** Slider stop values: simulation timesteps or a linear time grid,
   *  depending on the slider-mode setting. */
  time: number[];
  index: number;
  playing: boolean;
  /** Show the step counter (only meaningful in simulation-steps mode). */
  showStep: boolean;
  onChange: (index: number) => void;
  onPlayToggle: () => void;
}

/** Must match the thumb size in styles.css so ticks align with stops. */
const THUMB_PX = 14;
/** Cap on tick marks so long runs don't render a solid bar. */
const MAX_TICKS = 80;
const LABEL_COUNT = 8;

function positionCss(fraction: number): string {
  return `calc(${THUMB_PX / 2}px + ${fraction} * (100% - ${THUMB_PX}px))`;
}

export function TimeSlider({
  time,
  index,
  playing,
  showStep,
  onChange,
  onPlayToggle,
}: Props) {
  const n = time.length;
  const clamped = Math.min(index, n - 1);

  // One tick per slider stop (thinned on long runs), plus a few labeled
  // major ticks showing the stop's time value.
  const {tickIndices, labelIndices} = useMemo(() => {
    const stride = Math.max(1, Math.ceil(n / MAX_TICKS));
    const tickIndices: number[] = [];
    for (let i = 0; i < n; i += stride) tickIndices.push(i);
    if (tickIndices[tickIndices.length - 1] !== n - 1) tickIndices.push(n - 1);
    const labels = Math.min(LABEL_COUNT, n);
    const labelIndices = new Set(
      Array.from({length: labels}, (_, k) =>
        Math.round((k * (n - 1)) / Math.max(1, labels - 1)),
      ),
    );
    return {tickIndices, labelIndices};
  }, [n]);

  return (
    <div className="time-slider">
      <button
        className="icon-button"
        onClick={onPlayToggle}
        aria-label={playing ? 'Pause' : 'Play'}
        title={playing ? 'Pause' : 'Play through time'}
      >
        {playing ? '⏸' : '▶'}
      </button>
      <div className="slider-track-area">
        <input
          className="time-input"
          type="range"
          min={0}
          max={n - 1}
          step={1}
          value={clamped}
          onChange={e => onChange(Number(e.target.value))}
          aria-label="Simulation time"
          aria-valuetext={`t = ${time[clamped]?.toFixed(3)} s`}
        />
        <div className="slider-ticks" aria-hidden="true">
          {tickIndices.map(i => (
            <span
              key={`t${i}`}
              className={`slider-tick${labelIndices.has(i) ? ' major' : ''}`}
              style={{left: positionCss(n > 1 ? i / (n - 1) : 0)}}
            />
          ))}
          {[...labelIndices].map(i => (
            <span
              key={`l${i}`}
              className="slider-tick-label"
              style={{left: positionCss(n > 1 ? i / (n - 1) : 0)}}
            >
              {time[i]?.toFixed(2)}s
            </span>
          ))}
        </div>
      </div>
      <span className="time-readout">
        t = {time[clamped]?.toFixed(3)} s
        {showStep && (
          <span className="time-step">
            step {clamped + 1}/{n}
          </span>
        )}
      </span>
    </div>
  );
}
