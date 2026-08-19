"""The merge grid: which models get built, and what each one is called.

Pure enumeration — no torch, no I/O. `merge.build` consumes this, and the eval
runner and analysis stage will consume the same functions so that the set of
model ids is defined in exactly one place.

Grid (spec §V / paper Table II):

    6 pairs + 4 triples + 1 quadruple = 11 combinations
    x 5 methods                       = 55 merged models
    + base + 4 specialists            =  5 control rows

The coefficient sweep is deliberately NOT part of the main grid. It is a
secondary analysis over pairs under the two best-performing methods, and which
two those are is not known until evaluation has run — so it is enumerated
separately and parameterised by method.
"""

from __future__ import annotations

from itertools import combinations

# Canonical task order. Fixed here so that a combination always produces the
# same id regardless of the order the caller happens to pass tasks in.
TASKS = ("sql", "math", "translation", "json")

# Control rows: the unmerged base plus each individual specialist. These anchor
# both ends of the retention scale in eq. (1) — the specialist scores are the
# DENOMINATOR, so these must be evaluated before any retention number exists.
CONTROLS = ("base",) + TASKS


def canonical(combo) -> tuple[str, ...]:
    """Sort a task subset into canonical order and validate it."""
    unknown = set(combo) - set(TASKS)
    if unknown:
        raise ValueError(f"unknown task(s): {sorted(unknown)}")
    if len(set(combo)) != len(combo):
        raise ValueError(f"duplicate task in combo: {combo}")
    return tuple(t for t in TASKS if t in set(combo))


def all_combos(arity: int | None = None) -> list[tuple[str, ...]]:
    """The 11 adapter combinations, or just those of a given arity."""
    arities = (arity,) if arity is not None else (2, 3, 4)
    out: list[tuple[str, ...]] = []
    for k in arities:
        out.extend(combinations(TASKS, k))
    return out


def model_id(method: str, combo, weights=None) -> str:
    """Stable directory name for a merged model.

    Equal-weight grid entries omit the weight suffix, so a sweep entry at
    0.5/0.5 is distinguishable from the grid entry it duplicates.
    """
    combo = canonical(combo)
    base = f"{method}__{'-'.join(combo)}"
    if weights is None:
        return base
    return base + "__w" + "-".join(f"{w:g}" for w in weights)


def equal_weights(combo) -> list[float]:
    """Uniform mixing — the default for every entry in the main grid."""
    return [1.0 / len(combo)] * len(combo)


def enumerate_grid(methods, arity: int | None = None) -> list[dict]:
    """The main grid: every method x every combination, at equal weights."""
    specs = []
    for method in methods:
        for combo in all_combos(arity):
            specs.append({
                "model_id": model_id(method, combo),
                "method": method,
                "combo": combo,
                "weights": equal_weights(combo),
            })
    return specs


def enumerate_sweep(methods, weight_pairs) -> list[dict]:
    """Coefficient sweep: pairs only, under the given (top-2) methods."""
    specs = []
    for method in methods:
        for combo in all_combos(arity=2):
            for w in weight_pairs:
                if len(w) != 2:
                    raise ValueError(f"sweep weights must be pairs, got {w}")
                specs.append({
                    "model_id": model_id(method, combo, weights=w),
                    "method": method,
                    "combo": combo,
                    "weights": list(w),
                })
    return specs
