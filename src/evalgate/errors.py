"""The error base class.

Everything this codebase raises deliberately inherits from ``EvalgateError``,
so the CLI can tell "we detected a problem and are telling you about it" from
"something unexpected happened and you should see the traceback". A cassette
miss, a missing corpus and an unverifiable gold span are the first kind; they
deserve a sentence, not a stack trace.
"""

from __future__ import annotations


class EvalgateError(RuntimeError):
    """Base class for every error this codebase raises on purpose."""
