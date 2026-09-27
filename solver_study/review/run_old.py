"""python run_old.py WORKTREE SCRIPT ARGS...: runs SCRIPT with torax from WORKTREE."""
import runpy
import sys

worktree = sys.argv[1]
sys.meta_path[:] = [
    f for f in sys.meta_path if 'editable' not in getattr(f, '__module__', '')
]
sys.path.insert(0, worktree)
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
