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

# verb -> (module, callable). Extended as milestones land.
COMMANDS: dict[str, tuple[str, str]] = {
    "config": ("evalgate.commands.show_config", "run"),
    "ingest": ("evalgate.commands.ingest", "run"),
    "index": ("evalgate.commands.index", "run"),
    "retrieve": ("evalgate.commands.retrieve", "run"),
}

_USAGE = """usage: evalgate <command> [key=value ...]

commands:
{commands}

Any config key can be overridden positionally, e.g.
  evalgate config retriever=hybrid_rerank retriever.k=16
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
    result: int = command(rest)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
