"""Command dispatch.

``evalgate <verb> [hydra overrides]``. Everything after the verb is passed to
hydra's compose API, so any config value can be overridden from the command
line without a bespoke flag:

    make eval ARGS="retriever=bm25 retriever.k=3"

Commands are resolved lazily so that a verb needing torch does not make
``evalgate config`` pay for the import.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

from evalgate.errors import EvalgateError

# verb -> (module, callable). Extended as milestones land.
COMMANDS: dict[str, tuple[str, str]] = {
    "ablate": ("evalgate.commands.ablate", "run"),
    "config": ("evalgate.commands.show_config", "run"),
    "eval": ("evalgate.commands.evaluate", "run"),
    "freeze": ("evalgate.commands.freeze", "run"),
    "gate-demo": ("evalgate.commands.gate_demo", "run"),
    "ingest": ("evalgate.commands.ingest", "run"),
    "gen-eval": ("evalgate.commands.gen_eval", "run"),
    "index": ("evalgate.commands.index", "run"),
    "judge": ("evalgate.commands.judge", "run"),
    "label": ("evalgate.commands.label", "run"),
    "report": ("evalgate.commands.report", "run"),
    "retrieve": ("evalgate.commands.retrieve", "run"),
}

_USAGE = """usage: evalgate <command> [key=value ...]

commands:
{commands}

Any config key can be overridden positionally, e.g.
  evalgate config retriever=hybrid_rerank retriever.k=10
"""


def _usage() -> str:
    listing = "\n".join(f"  {name}" for name in sorted(COMMANDS))
    return _USAGE.format(commands=listing)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a verb to its command module."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        sys.stdout.write(_usage())
        return 0

    verb, *rest = args
    if verb not in COMMANDS:
        sys.stderr.write(f"unknown command: {verb}\n\n{_usage()}")
        return 2

    module_name, attr = COMMANDS[verb]
    module = importlib.import_module(module_name)
    command = getattr(module, attr)
    try:
        result: int = command(rest)
    except EvalgateError as exc:
        # Errors we raise on purpose carry an actionable message; a traceback
        # would bury it. Anything else propagates with its stack intact.
        sys.stderr.write(f"error: {exc}\n")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130
    return result


if __name__ == "__main__":
    raise SystemExit(main())
