# -*- coding: utf-8 -*-
"""stnull: audit spatial-expression predictions against zero-pixel nulls.

Author: Zigan Wang.

    from stnull import audit
    rep = audit(y_true=Y, y_pred=P, space="log1p_cp10k:panel",
                section=sec, lib_size=lib, coords=xy, labels=lab)
    rep.to_html("audit.html")
    print(rep.summary())

The package never transforms your data, never trains a model, never reads an
image, and never prints a value without the permutation floor of the same
readout rule next to it.

PLAIN LANGUAGE
    A "zero-pixel null" is a competitor model that never sees the histology
    image: it predicts expression from sequencing depth, tissue composition
    or spatial neighbours alone.  If a published image model does not beat
    these nulls, the image was not the source of its score.  This package
    computes those nulls and the permutation floors that price pure chance.

MODULE MAP
    core     Stat, linear algebra, permutation / bootstrap / K-fold plans
    spaces   the target-space declaration that has no default
    nulls    the zero-pixel null models and the neighbourhood operators
    drg      the attribution ladder r_full -> DG -> CRG -> DRG
    ceiling  r_tech, the split-half measurement ceiling
    levers   what selection, smoothing and split granularity are worth
    audit    the orchestrator; the only module that knows about sections
    report   result containers, claim wording, renderers
    cli      a thin shell around audit() with no statistics of its own

DATA LINEAGE (row numbers refer to docs/CODE_WALKTHROUGH.md (section 4, the variable lineage table))
    This file computes nothing and holds no data; it is the public doorway.
    The names re-exported below are the two ends of the pipeline plus its
    error surface:
    * entry: ``audit`` receives the user tensors of [L01]-[L14]
      (y_true/y_pred matrices, section labels, lib_size, coords, labels,
      counts, ...), whether typed directly, mapped from CLI flags by the
      [L16] rename table in cli.py, or built by the example script [L17];
      ``check_inputs`` is the standalone validator behind [L52];
      ``nulls_only`` / ``ladder`` / ``perm_floor`` / ``compare`` are the
      thin wrappers [L53]-[L56] that re-route into the same ``audit`` core.
    * exit: ``AuditReport`` and its section containers (RunMeta, Headline,
      DepthResult, CompositionResult, SpatialResult, SelectionResult,
      CeilingResult, Claim) carry every result out, and every number inside
      them is a ``Stat``, the value/floor/ci triple of [L39]/[L40].
    * declarations and errors: ``TargetSpace`` is the [L12] space
      declaration; the exception classes are core.py's hierarchy, with
      ``NotComparable`` raised by ``compare`` [L56] and
      ``LibSizeLooksLikePanelSum`` guarding the [L04] depth input.
"""
# Entry points and the input validator: audit is the orchestrator that
# consumes [L01]-[L14]; nulls_only/ladder/perm_floor/compare are the
# wrappers [L53]-[L56]; check_inputs/InputReport are the [L52] pre-flight.
from .audit import (InputReport, audit, check_inputs, compare, ladder,
                    nulls_only, perm_floor)
# The number container ([L40]) and the full exception hierarchy, so users
# can catch StnullError once or handle each actionable failure separately.
from .core import (BadInput, InsufficientData, LibSizeLooksLikePanelSum,
                   MissingSpace, NotComparable, Stat, StnullError)
# Result containers: AuditReport is what audit() returns; the *Result
# dataclasses are its sections, all rendered by report.py.  VERSION is the
# single version string of the package (re-exported as __version__ below).
from .report import (VERSION, AuditReport, CeilingResult, Claim,
                     CompositionResult, DepthResult, Headline, RunMeta,
                     SelectionResult, SpatialResult, cite_text)
# The [L12] target-space enum, exported so users can pass members instead
# of strings.
from .spaces import TargetSpace

__version__ = VERSION
__all__ = [
    "audit", "nulls_only", "ladder", "perm_floor", "compare", "check_inputs",
    "AuditReport", "Stat", "TargetSpace", "RunMeta", "Headline", "DepthResult",
    "CompositionResult", "SpatialResult", "SelectionResult", "CeilingResult",
    "Claim", "InputReport", "cite_text", "StnullError", "MissingSpace",
    "NotComparable", "LibSizeLooksLikePanelSum", "InsufficientData", "BadInput",
    "__version__",
]
