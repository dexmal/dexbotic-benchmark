"""Launch upstream RoboDojo with parent-owned runtime compatibility hooks."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

from robodojo_runtime_compat import (
    apply_runtime_compat,
    wrap_create_eval_env,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBODOJO_ROOT = (PROJECT_ROOT / "robodojo").resolve()


def _argument_value(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _pop_argument(name: str) -> str | None:
    value = _argument_value(name)
    if value is None:
        return None
    index = sys.argv.index(name)
    del sys.argv[index : index + 2]
    return value


def _prioritize_runtime_paths() -> None:
    preferred = [str(ROBODOJO_ROOT), str(ROBODOJO_ROOT / "XPolicyLab")]
    for path in reversed(preferred):
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def main() -> None:
    device_id = _argument_value("--device_id")
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    # This parent-owned guard is resolved into --num_envs before launch. Keep
    # it in the logged command for provenance, then remove it because the
    # upstream argparse surface deliberately does not carry benchmark-local
    # safety switches.
    _pop_argument("--max_num_envs")

    _prioritize_runtime_paths()
    # Upstream resolves assets, configs, and eval_result relative to its repo
    # root. The pre-migration shell entrypoint also ran from this directory.
    os.chdir(ROBODOJO_ROOT)
    native_main = importlib.import_module("src.eval_client.main")
    apply_runtime_compat()
    native_main.create_eval_env = wrap_create_eval_env(native_main.create_eval_env)
    native_main.main()


if __name__ == "__main__":
    main()
