"""``evalgate retrieve``: run one query through the configured retriever."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from evalgate.config import load_config
from evalgate.pipeline import build_stack

console = Console()
QUERY_PREFIX = "query="
SNIPPET_CHARS = 160


def run(overrides: list[str]) -> int:
    """Retrieve for a query passed as ``query=...``."""
    queries = [arg[len(QUERY_PREFIX) :] for arg in overrides if arg.startswith(QUERY_PREFIX)]
    if not queries:
        console.print('[red]usage: evalgate retrieve query="..." [overrides][/red]')
        return 2

    cfg = load_config(overrides=[arg for arg in overrides if not arg.startswith(QUERY_PREFIX)])
    stack = build_stack(cfg)
    for query in queries:
        table = Table(title=query, show_edge=False)
        table.add_column("#", justify="right")
        table.add_column("chunk")
        table.add_column("score", justify="right")
        table.add_column("text")
        for result in stack.retriever.retrieve(query):
            snippet = " ".join(result.text.split())[:SNIPPET_CHARS]
            table.add_row(str(result.rank), result.chunk_id, f"{result.score:.4f}", snippet)
        console.print(table)
    return 0
