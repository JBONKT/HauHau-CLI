"""Argument coercion for tool calls.

JSON is looser than a schema: ``1e400`` parses to infinity, ``"false"`` is a
truthy string, and a list can arrive where a string was promised. Each of those
used to escape a tool as a raw ``TypeError``, ``ValueError``, ``OverflowError``
or ``AttributeError``, which surfaced as "tool crashed" - a message that tells
the model nothing and hides whether anything ran.

Every helper raises :class:`BadArgument` with a sentence the model can act on;
``Tool.run`` turns that into an ordinary tool failure.
"""

from __future__ import annotations

import math


class BadArgument(ValueError):
    """An argument the tool cannot use, described for the caller."""


def text(args: dict, key: str, default: str = "", *, required: bool = False) -> str:
    if key not in args or args[key] is None:
        if required:
            raise BadArgument(f"'{key}' is required")
        return default
    value = args[key]
    if not isinstance(value, str):
        raise BadArgument(f"'{key}' must be a string, got {type(value).__name__}")
    return value


def integer(
    args: dict,
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if key not in args or args[key] is None:
        value = default
    else:
        raw = args[key]
        if isinstance(raw, bool):
            value = int(raw)
        elif isinstance(raw, int):
            value = raw
        elif isinstance(raw, float):
            # 1e400 parses to inf and int(inf) raises OverflowError, which no
            # isinstance check catches.
            if not math.isfinite(raw):
                raise BadArgument(f"'{key}' must be a finite integer")
            value = int(raw)
        elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
            value = int(raw.strip())
        else:
            raise BadArgument(f"'{key}' must be an integer, got {type(raw).__name__}")
    if minimum is not None and value < minimum:
        raise BadArgument(f"'{key}' must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise BadArgument(f"'{key}' must be at most {maximum}, got {value}")
    return value


def number(
    args: dict,
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if key not in args or args[key] is None:
        value = float(default)
    else:
        raw = args[key]
        if isinstance(raw, bool):
            raise BadArgument(f"'{key}' must be a number, got a boolean")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise BadArgument(f"'{key}' must be a number, got {type(raw).__name__}") from exc
        if not math.isfinite(value):
            raise BadArgument(f"'{key}' must be a finite number")
    if minimum is not None and value < minimum:
        raise BadArgument(f"'{key}' must be at least {minimum}, got {value:g}")
    if maximum is not None and value > maximum:
        raise BadArgument(f"'{key}' must be at most {maximum}, got {value:g}")
    return value


def flag(args: dict, key: str) -> bool:
    """True only for an actual truth value - ``"false"`` is not user consent.

    ``args.get("replace_all")`` treated the string "false" (and "0", and "no")
    as permission to rewrite every occurrence, which is the destructive branch.
    """
    value = args.get(key)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return value is True or (value == 1 and not isinstance(value, bool))
