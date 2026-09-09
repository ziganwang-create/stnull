# -*- coding: utf-8 -*-
"""stnull.spaces: target-space declarations.

Author: Zigan Wang.

WHAT THIS MODULE IS
    The enumeration of the normalisations an expression target can live in,
    and the coercion function that refuses to guess which one you used.

PLAIN-LANGUAGE TERMS
    * normalisation / "space": the arithmetic applied to raw molecule counts
      before anyone correlates anything, e.g. divide by the spot's total
      count, multiply by 10,000, take log(1 + x).  Two papers can report the
      same Pearson r in different spaces and be measuring different things.
    * library size: the total number of RNA molecules (UMIs) counted in one
      spot; "depth" for short.  Deeper spots have larger counts everywhere.
    * panel vs full denominator: dividing by the sum over only the audited
      gene panel versus the sum over the whole transcriptome.  The two make
      genuinely different targets (see WHY below).

WHO CALLS IT
    ``audit()`` (first line: ``coerce_space``), ``ceiling.apply_space`` (to
    build the measurement replicate in the same space as the target) and
    ``report`` (to print the declared space and to flag depth-carrying ones).

DATA LINEAGE (row numbers refer to docs/CODE_WALKTHROUGH.md (section 4, the variable lineage table))
    [L12] is this module's entire traffic.  The user's ``space=`` string
    (a REQUIRED argument of ``audit()``, or the CLI's ``--space`` flag via
    the [L16] rename table at cli.py:127) enters :func:`coerce_space` at
    audit.py:757 and leaves as a ``TargetSpace`` member ``t_space``.  That
    member is then consumed in four places: ``ceiling.r_tech`` replays the
    same transform on thinned counts (audit.py:1389 -> ceiling.apply_space,
    ceiling.py:72); membership in :data:`DEPTH_CARRYING` sets
    ``DepthResult.target_carries_depth`` (audit.py:1224); ``RunMeta.space``
    records it for the report (audit.py:1432); and ``compare()`` builds its
    comparability key from it (audit.py:1561, raising
    ``core.NotComparable`` on a mismatch).  Nothing in this module touches
    expression data; it only classifies the user's declaration.

WHY ``space`` HAS NO DEFAULT (measured, not stylistic)
    The same zero-pixel composition null scores 0.007-0.083 higher under a
    panel-row-sum denominator than under a full-transcriptome denominator.
    That gap is larger than most single-step "advances" in the H&E->expression
    literature, so a default would be stnull silently inventing comparability
    on the user's behalf.
"""
from __future__ import annotations

from enum import Enum

from .core import MissingSpace


class TargetSpace(str, Enum):
    """How ``y_true`` was normalised before it reached stnull.

    A closed list on purpose: every member below was seen in the audited
    literature, and anything else must be declared as ``CUSTOM`` with a
    note.  Subclassing ``str`` keeps JSON serialisation and equality with
    plain strings trivial (``TargetSpace("log1p:raw") == "log1p:raw"``).
    """

    LOG1P_CP10K_PANEL = "log1p_cp10k:panel"   # denominator = panel row sum
    LOG1P_CP10K_FULL = "log1p_cp10k:full"     # denominator = full-transcriptome lib
    LOG1P_RAW = "log1p:raw"                   # HEST convention, carries depth
    LOG10_MEDLIB = "log10:median_lib"         # HisToGene / scprep convention
    ZSCORE = "zscore:per_gene"
    CUSTOM = "custom"                         # requires space_note=...


#: Spaces whose target values are themselves monotone in sequencing depth.  In
#: these, a high depth null is the DEFINITION of the space rather than a defect
#: of the model, and the report says so on its own banner.  The other four
#: spaces divide by a library size, which removes the leading depth term.
#: Consumed at audit.py:1224 to set ``DepthResult.target_carries_depth``
#: (part of the [L12] flow).
DEPTH_CARRYING = {TargetSpace.LOG1P_RAW}

#: The error text shown whenever the declaration is missing or wrong.  It is
#: the one place the valid values are spelled out for the user, so the CLI,
#: the API and the docs all point here instead of repeating the list.
SPACE_HELP = """\
space= is required and has no default.  Valid values:
  'log1p_cp10k:panel'  log1p(1e4 * count / panel row sum)
  'log1p_cp10k:full'   log1p(1e4 * count / full-transcriptome library size)
  'log1p:raw'          log1p(count)            [target itself carries depth]
  'log10:median_lib'   log10(1 + count/lib * median lib)
  'zscore:per_gene'    per-gene standardisation of any of the above
  'custom'             anything else -- you must also pass space_note='...'
Why no default: the same composition null moves by 0.007-0.083 r between the
panel and full-transcriptome denominators, which is larger than most reported
year-on-year gains in this field."""


def coerce_space(space, space_note: str = "") -> TargetSpace:
    """Validate the declared target space.

    Lineage [L12]: the single entry point of this module.  Called by
    ``audit()`` at audit.py:757 before any array is touched, with ``space``
    fresh from the user (or from the CLI's ``--space`` flag through the
    [L16] rename).  The returned member is the ``t_space`` variable whose
    four consumers are listed in the module docstring.

    Parameters
    ----------
    space : str or TargetSpace
        One of the values listed in ``SPACE_HELP``.  ``None`` is an error, not
        a default.
    space_note : str
        Free-text description of the transform; REQUIRED with
        ``space='custom'``, because a custom space disables the measurement
        ceiling and the reader needs to know what was actually done.

    Returns
    -------
    TargetSpace

    Raises
    ------
    MissingSpace
        When the space is absent, unrecognised, or custom without a note.
    """
    # --- 1. Absent is an error, never a default ---------------------------
    # WHY: see the module docstring; a silent default would decide the
    # meaning of every downstream r on the user's behalf.
    if space is None:
        raise MissingSpace(SPACE_HELP)
    # --- 2. Coerce string -> enum member ----------------------------------
    # WHAT: accept either the enum member itself (idempotent re-entry) or
    # its string value; anything else fails against the closed list above.
    if isinstance(space, TargetSpace):
        sp = space
    else:
        try:
            sp = TargetSpace(str(space))
        except ValueError:
            raise MissingSpace(
                "unrecognised space=%r.\n%s" % (space, SPACE_HELP))
    # --- 3. CUSTOM must come with a description ---------------------------
    # WHY: 'custom' switches the ceiling replay off (ceiling.apply_space has
    # no recipe for it), so the note is the only record of what the target
    # actually is; it travels into RunMeta and the report.
    if sp is TargetSpace.CUSTOM and not space_note:
        raise MissingSpace(
            "space='custom' requires space_note='<how you transformed y>'")
    return sp
