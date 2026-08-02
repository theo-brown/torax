"""Physics comparison plots: baseline vs JFNK vs psi-splitting.

Reads the trajectories the benchmark harness recorded (no JAX, no simulation),
so this is cheap to run while the solver agent is busy.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/tmp/bench_master'
OUT = '/home/user/torax/bench/figs'
os.makedirs(OUT, exist_ok=True)

# Validated categorical palette, assigned in fixed order and never cycled.
BASE, JFNK, PSI = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b8b7b2'
SURFACE = '#fcfcfb'

PROFILES = [
    ('T_i', 'Ion temperature', 'keV', 1.0),
    ('T_e', 'Electron temperature', 'keV', 1.0),
    ('n_e', 'Electron density', r'10$^{19}$ m$^{-3}$', 1e-19),
    ('psi', 'Poloidal flux', 'Wb', 1.0),
]


def load(tag):
  return np.load(os.path.join(D, f'{tag}.npz'))


def rho(n):
  """Uniform cell centres on the normalised radius."""
  return (np.arange(n) + 0.5) / n


def style(ax):
  ax.set_facecolor(SURFACE)
  for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
  for s in ('left', 'bottom'):
    ax.spines[s].set_color(MUTED)
  ax.tick_params(colors=INK2, labelsize=8, length=3)
  ax.grid(True, color=MUTED, lw=0.5, alpha=0.45)
  ax.set_axisbelow(True)


def figure_profiles(base_tag, jfnk_tag, psi_tag, fname, title, sub):
  b, j, p = load(base_tag), load(jfnk_tag), load(psi_tag)
  n = b['T_i'].shape[1]
  x = rho(n)
  fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), facecolor=SURFACE)
  fig.suptitle(title, x=0.012, ha='left', fontsize=15, color=INK, weight='bold')
  fig.text(0.012, 0.925, sub, ha='left', fontsize=10, color=INK2)

  for k, (name, label, unit, scale) in enumerate(PROFILES):
    # --- top: the profiles themselves, at the final time ---
    ax = axes[0, k]
    style(ax)
    for arr, colour, lab in ((b, BASE, 'baseline'), (j, JFNK, 'JFNK'),
                             (p, PSI, 'psi split')):
      ax.plot(x, arr[name][-1] * scale, color=colour, lw=2.0, label=lab,
              solid_capstyle='round')
    ax.set_title(f'{label}  [{unit}]', fontsize=10, color=INK, loc='left')
    if k == 0:
      ax.set_ylabel('value at t = 80 s', fontsize=9, color=INK2)
      ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc='best')

    # --- bottom: signed relative difference from the baseline ---
    ax = axes[1, k]
    style(ax)
    ref = b[name][-1]
    denom = max(np.abs(ref).max(), 1e-30)
    for arr, colour, lab in ((j, JFNK, 'JFNK'), (p, PSI, 'psi split')):
      ax.plot(x, (arr[name][-1] - ref) / denom, color=colour, lw=2.0,
              label=lab, solid_capstyle='round')
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.set_xlabel(r'normalised radius  $\hat\rho$', fontsize=9, color=INK2)
    if k == 0:
      ax.set_ylabel('relative difference\nfrom baseline', fontsize=9,
                    color=INK2)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax.yaxis.get_offset_text().set(color=INK2, size=8)

  fig.tight_layout(rect=(0, 0, 1, 0.90))
  path = os.path.join(OUT, fname)
  fig.savefig(path, dpi=150, facecolor=SURFACE)
  plt.close(fig)
  return path


def figure_error_growth(fname):
  """How the disagreement with the baseline evolves, and why n_rho=100 differs."""
  b, j, p = load('base'), load('il_jfnk_1'), load('ps_lie')
  b1, j1 = load('n100_base_1'), load('n100_jfnk_1')
  fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)
  fig.suptitle('Where the disagreement comes from', x=0.012, ha='left',
               fontsize=15, color=INK, weight='bold')
  fig.text(0.012, 0.90,
           'Left: at n_rho=25 both variants track the baseline to solver '
           'tolerance. Right: at n_rho=100 the baseline itself is '
           'unconverged, so trajectories separate regardless of method.',
           ha='left', fontsize=9.5, color=INK2)

  ax = axes[0]
  style(ax)
  for arr, colour, lab in ((j, JFNK, 'JFNK'), (p, PSI, 'psi split')):
    m = min(len(arr['t']), len(b['t']))
    worst = np.zeros(m)
    for name, *_ in PROFILES:
      ref = b[name][:m]
      d = np.abs(arr[name][:m] - ref).max(1) / np.abs(ref).max()
      worst = np.maximum(worst, d)
    ax.semilogy(b['t'][:m], np.maximum(worst, 1e-16), color=colour, lw=2.0,
                label=lab, solid_capstyle='round')
  ax.axhline(1e-4, color=MUTED, lw=1.2, ls='--')
  ax.text(2, 1.3e-4, 'accuracy target 1e-4', fontsize=8.5, color=INK2)
  ax.set_xlabel('time  [s]', fontsize=9, color=INK2)
  ax.set_ylabel('worst relative difference\nfrom baseline', fontsize=9,
                color=INK2)
  ax.set_title('n_rho = 25', fontsize=10, color=INK, loc='left')
  ax.legend(frameon=False, fontsize=9, labelcolor=INK2)

  ax = axes[1]
  style(ax)
  for arr, colour, lab in ((b1, BASE, 'baseline'), (j1, JFNK, 'JFNK')):
    unconv = (arr['err'] == 2).sum()
    ax.plot(arr['t'], arr['dt'], color=colour, lw=2.0,
            label=f'{lab}  ({unconv}/{len(arr["t"])} steps unconverged)',
            solid_capstyle='round')
  ax.set_xlabel('time  [s]', fontsize=9, color=INK2)
  ax.set_ylabel('step size dt  [s]', fontsize=9, color=INK2)
  ax.set_title('n_rho = 100 — dt backtracking', fontsize=10, color=INK,
               loc='left')
  ax.legend(frameon=False, fontsize=9, labelcolor=INK2)

  fig.tight_layout(rect=(0, 0, 1, 0.86))
  path = os.path.join(OUT, fname)
  fig.savefig(path, dpi=150, facecolor=SURFACE)
  plt.close(fig)
  return path


if __name__ == '__main__':
  p1 = figure_profiles(
      'base', 'il_jfnk_1', 'ps_lie', 'variants_n25.png',
      'Evolved profiles: baseline vs JFNK vs psi splitting',
      'test_iterhybrid_rampup, n_rho = 25, 80 s. Top row: the three solvers '
      'overlay. Bottom row: difference from baseline, note the axis exponent.')
  p2 = figure_error_growth('variants_error.png')
  for p in (p1, p2):
    print('WROTE', p)
