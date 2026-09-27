"""official_api.py with the structured Jacobian jitted with inline=AUTO|NEVER instead of XLA_LATE.

python official_api_patched.py INLINE MODE N_RHO T_FINAL
"""
import sys
import jax as _jax
from torax._src.solver import structured_jacobian as sj

INLINE = sys.argv.pop(1)


class _Shim:
  def __getattr__(self, name):
    return getattr(_jax, name)

  def jit(self, f, **kw):
    if INLINE == 'none':
      return f  # no jit wrapper at all
    kw['inline'] = getattr(_jax.Inline, INLINE.upper())
    return _jax.jit(f, **kw)


sj.jax = _Shim()
sys.argv = [sys.argv[0]] + sys.argv[1:]
exec(open('official_api.py').read())
