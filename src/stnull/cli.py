# -*- coding: utf-8 -*-
"""stnull.cli: command line interface.

Author: Zigan Wang.

WHAT THIS MODULE IS
    A thin shell around :func:`stnull.audit.audit`.  It loads matrices and a
    per-spot table from disk, maps column names to keyword arguments, runs the
    audit and writes the requested output files.  It contains no statistics of
    its own, so the CLI and the Python API can never disagree about a number.

    python -m stnull.cli audit --help

SUBCOMMANDS
    audit   full audit of a model's predictions
    nulls   zero-pixel nulls only, no model
    levers  the protocol lever price list
    check   validate inputs only (seconds, no correlations computed)
    cite    print the Methods paragraph and the defaults

Exit codes: 0 ok | 2 strict validation failed | 3 degraded and --fail-on-degraded
            | 4 input unreadable

DATA LINEAGE (row numbers refer to docs/CODE_WALKTHROUGH.md (section 4, the variable lineage table))
    Inputs
        [L15] --true / --pred files on disk, loaded by ``_load_matrix``
              (cli.py:_load_matrix) into 2-D arrays; they become the two
              positional arguments of ``audit(Y, P, **kw)`` and from there
              the matrices ``Y`` [L01] and ``P`` [L02] inside audit.py.
        [L16] --obs, one per-spot table loaded by ``_load_obs``; ``_kwargs``
              picks single columns out of it and renames each one onto
              exactly one ``audit()`` keyword (--section-col -> section
              [L03], --lib-col -> lib_size [L04], --coord-cols -> coords
              [L06], --label-col / --comp-cols -> labels / composition
              [L07], --patient-col -> patient [L09], --train-col ->
              is_train [L10], --libpred-col -> lib_pred [L11],
              --libfull-col -> lib_size_full [L14], --counts -> counts
              [L08], --genes -> genes [L13], --space -> space [L12]).
    Outputs
        The AuditReport that ``audit()`` returns; ``_emit`` only forwards it
        to the report.py renderers (to_html / to_markdown / to_json /
        to_csv_dir) and prints ``rep.summary()``.  No tensor is created or
        transformed in this module: every statistic a CLI run prints was
        computed inside audit.py, which is why the CLI and the Python API
        can never disagree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .core import StnullError
from .report import VERSION, cite_text

EXIT_OK, EXIT_STRICT, EXIT_DEGRADED, EXIT_INPUT = 0, 2, 3, 4


# ------------------------------------------------------------------ loading
def _load_matrix(path):
    """Read a matrix from .npy / .npz (dense or sparse) / .csv / .tsv / .parquet.

    This is flow [L15]: the value returned here is handed unchanged to
    ``audit()`` as ``y_true`` [L01], ``y_pred`` [L02] or ``counts`` [L08];
    all coercion (densifying, float casting, shape checks) happens inside
    audit.py, not here.  Expected orientation: (n_spots, n_genes), one row
    per spot (a spot is one measured location on the tissue slide, a small
    circle covering a handful of cells).
    """
    # WHAT: dispatch on the file extension only, never sniff contents.
    # WHY: a wrong guess about the format would silently transpose or
    # re-type the matrix; failing loudly on an unknown suffix is safer.
    p = Path(path)
    if p.suffix == ".npy":
        # allow_pickle=False: a .npy from an untrusted source cannot run code.
        return np.load(p, allow_pickle=False)
    if p.suffix == ".npz":
        # WHAT: a .npz may be a numpy archive of dense arrays or a
        # scipy.sparse export; the two share an extension.
        # WHY: try the dense reading first, fall back to sparse only when
        # the archive holds no 2-D array (scipy stores 1-D index arrays).
        z = np.load(p, allow_pickle=False)
        keys = [k for k in z.files if z[k].ndim == 2]  # names of 2-D members
        if not keys:
            import scipy.sparse as sp
            return sp.load_npz(p)
        if len(keys) > 1:
            # WHY: with several 2-D members the choice would be arbitrary,
            # and an arbitrary choice of which matrix is y_true is exactly
            # the kind of silent error this package audits others for.
            raise StnullError("%s holds several 2-D arrays (%s); split it or pass "
                              "a .npy" % (p, ", ".join(keys)))
        return z[keys[0]]
    if p.suffix in (".csv", ".tsv"):
        # WHERE: a DataFrame return is fine, audit.py's _mat() takes
        # .values and reads gene names off the columns ([L13]).
        return pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    raise StnullError("unsupported matrix format: %s" % p.suffix)


def _load_obs(path):
    """Read the per-spot table that carries section, coords, labels and depth.

    This is the source table of flow [L16]: one row per spot, aligned with
    the rows of the --true matrix.  ``_kwargs`` below picks named columns
    out of it; nothing else reads it.  Only tabular formats are accepted
    because a bare array would lose the column names the flags refer to.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".csv", ".tsv"):
        return pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
    raise StnullError("--obs must be .parquet/.csv/.tsv")


# ------------------------------------------------------------------ argument
def _common(ap, with_pred=True):
    """Attach the shared flags.  Option values are restricted by `choices=` so
    that the CLI cannot select an estimator the API would reject.

    Flag layout mirrors the [L16] contract: file flags first, then one
    ``--*-col`` flag per optional per-spot column, then the protocol knobs
    whose defaults live in ``audit()``, then the output flags.
    """
    # --- required files and the declared target space -----------------
    # WHAT: the two matrices [L15] plus the obs table [L16]; --space [L12]
    # is required with no default because the audit refuses to guess which
    # normalisation y_true is in (a wrong space silently changes what the
    # depth null means).
    ap.add_argument("--true", required=True, help="y_true matrix (.npy/.npz/.csv)")
    if with_pred:
        ap.add_argument("--pred", help="y_pred matrix; omit for a nulls-only run")
    ap.add_argument("--obs", required=True, help="per-spot table (.parquet/.csv)")
    ap.add_argument("--space", required=True,
                    help="target space, e.g. log1p_cp10k:panel (no default)")
    ap.add_argument("--space-note", default="")
    # --- column pick-out flags (each one is a pure rename, see _kwargs) --
    ap.add_argument("--section-col", default="section")
    ap.add_argument("--lib-col")
    ap.add_argument("--coord-cols", help="e.g. arr_x,arr_y")
    ap.add_argument("--coord-kind", default="grid", choices=["grid", "micron"])
    ap.add_argument("--label-col")
    ap.add_argument("--comp-cols")
    ap.add_argument("--patient-col")
    ap.add_argument("--train-col")
    ap.add_argument("--libpred-col")
    ap.add_argument("--counts")
    ap.add_argument("--libfull-col")
    ap.add_argument("--genes")
    # --- protocol knobs ------------------------------------------------
    # WHAT: the statistical settings, forwarded verbatim to audit().
    # WHY choices=: an estimator name the API does not know would only be
    # rejected deep inside audit(); restricting it here fails at parse time
    # with a proper usage message instead.
    ap.add_argument("--top-n", default=None,
                    help="comma list, e.g. 10,50; default: 10,50,100,250 "
                         "trimmed to the panel size")
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--perm-kind", default="auto", choices=["auto", "block", "free"])
    ap.add_argument("--depth-proxy", default="observed",
                    choices=["observed", "from_pred", "given", "none"],
                    help="default 'observed' (validated against the oracle "
                         "ladder); 'from_pred' carries a permutation guard")
    ap.add_argument("--null-fit", default="cv_within_section",
                    choices=["cv_within_section", "train_only", "oracle"])
    ap.add_argument("--min-spots", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--lang", default="en", choices=["en", "zh"])
    # --- output flags (consumed only by _emit below) -------------------
    ap.add_argument("--out", help="audit.html / .md / .json")
    ap.add_argument("--csv-dir")
    ap.add_argument("--json")
    ap.add_argument("--no-strict", action="store_true")
    ap.add_argument("--fail-on-degraded", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap


def _kwargs(a, with_pred=True):
    """Turn parsed flags into the audit() keyword arguments (no defaults added
    here: every default lives in audit(), so both entry points share one).

    This function IS the [L16] CLI-to-API contract: every branch below is
    one rename of a file or a column onto one ``audit()`` keyword, and
    nothing is computed.  If a mapping ever did arithmetic, the CLI and the
    Python API could produce different numbers from the same data.
    """
    # WHAT: load the three on-disk objects.
    # WHERE: obs is the per-spot table from _load_obs [L16]; Y and P are
    # the matrix files from _load_matrix [L15] and become audit()'s
    # positional y_true [L01] and y_pred [L02].  P=None switches the audit
    # into nulls-only mode (audit.py handles that, not the CLI).
    obs = _load_obs(a.obs)          # (n_spots rows) per-spot metadata table
    Y = _load_matrix(a.true)        # (n_spots, n_genes) ground truth
    P = _load_matrix(a.pred) if (with_pred and getattr(a, "pred", None)) else None
    # WHAT: the always-present keywords, protocol knobs plus the section
    # labels [L03] (which tissue slice each spot belongs to; every r in the
    # audit is computed within one section).
    # WHY tuple(top_n): "10,50" -> (10, 50); None lets audit() apply its
    # own default trimmed to the panel size (panel = the set of genes being
    # scored).
    kw = dict(space=a.space, space_note=a.space_note,     # [L12] target space
              section=obs[a.section_col].values,          # [L03] section labels
              coord_kind=a.coord_kind,
              depth_proxy=a.depth_proxy, null_fit=a.null_fit,
              top_n=(tuple(int(x) for x in a.top_n.split(",") if x)
                     if a.top_n else None),
              n_perm=a.perm, n_boot=a.boot, perm_kind=a.perm_kind,
              min_spots=a.min_spots, seed=a.seed, n_jobs=a.jobs,
              strict=not a.no_strict, verbose=not a.quiet)
    # WHAT: optional per-spot columns, added only when the flag was given.
    # WHY: audit() treats a missing keyword as "input absent" and logs it
    # in the degradation ledger [L48]; passing a made-up default here would
    # hide that absence from the report.
    if a.lib_col:
        # [L04] library size: total UMI count per spot (UMI = one counted
        # RNA molecule), i.e. sequencing depth, the main known confounder.
        kw["lib_size"] = obs[a.lib_col].values
    if a.coord_cols:
        # [L06] spot positions, e.g. "arr_x,arr_y"; enables block
        # permutation and every spatial readout.
        cs = a.coord_cols.split(",")
        kw["coords"] = obs[cs].values               # (n_spots, 2)
    if a.label_col:
        # [L07] one categorical tissue label per spot (label branch).
        kw["labels"] = obs[a.label_col].values
    if a.comp_cols:
        # [L07] continuous branch: K columns of class proportions per spot.
        kw["composition"] = obs[a.comp_cols.split(",")].values
    if a.patient_col:
        # [L09] patient id per spot; enables the split-granularity lever.
        kw["patient"] = obs[a.patient_col].values
    if a.train_col:
        # [L10] the user's own train/test split as a boolean column.
        kw["is_train"] = obs[a.train_col].values
    if a.libpred_col:
        # [L11] a user-computed depth proxy, used when --depth-proxy=given.
        kw["lib_pred"] = obs[a.libpred_col].values
    if a.libfull_col:
        # [L14] full-transcriptome depth, needed only by the ceiling module.
        kw["lib_size_full"] = obs[a.libfull_col].values
    if a.counts:
        # [L08] raw UMI counts (a matrix file, not an obs column): the only
        # input that can price measurement noise in ceiling.r_tech.
        kw["counts"] = _load_matrix(a.counts)
    if a.genes:
        # [L13] gene names, one per whitespace-separated token; cosmetic
        # (they label the per_gene table rows), never used in a statistic.
        kw["genes"] = Path(a.genes).read_text(encoding="utf-8").split()
    return Y, P, kw


# ------------------------------------------------------------------- output
def _emit(rep, a):
    """Write the requested files, print the console summary, pick the exit code.

    WHERE: ``rep`` is the AuditReport built by audit.py (assembly at
    audit.py:1449); every renderer called here lives in report.py, so this
    function decides only WHICH renderers run, never what they say.
    """
    outs = []  # list of Path objects actually written, echoed at the end
    # WHAT: --out picks one main format by file extension.
    if a.out:
        p = Path(a.out)
        if p.suffix == ".html":
            outs.append(rep.to_html(p, lang=a.lang))
        elif p.suffix == ".md":
            rep.to_markdown(p, lang=a.lang)
            outs.append(p)
        elif p.suffix == ".json":
            rep.to_json(p)
            outs.append(p)
        else:
            raise StnullError("--out must end in .html, .md or .json")
    # WHAT: --json and --csv-dir are additive side outputs; --csv-dir
    # writes the per_section [L42], per_gene [L43], ladder [L44] and
    # claims tables via report.to_csv_dir.
    if a.json:
        rep.to_json(a.json)
        outs.append(Path(a.json))
    if a.csv_dir:
        outs.extend(rep.to_csv_dir(a.csv_dir))
    # WHAT: the console summary always prints, even when files were written.
    # WHY lang="en": the console renderer is ASCII-only by design (a cp936
    # Windows terminal garbles anything else); file renderers honour --lang.
    print(rep.summary(lang="en"))
    for p in outs:
        print("[wrote] %s" % p)
    # WHAT: turn a non-empty degradation ledger [L48] into a failing exit
    # code when the user asked for that (useful in CI pipelines).
    if a.fail_on_degraded and rep.caveats():
        print("[stnull] --fail-on-degraded: %d degradation(s)" % len(rep.caveats()))
        return EXIT_DEGRADED
    return EXIT_OK


def main(argv=None):
    """Entry point for the ``stnull`` console script; returns an exit code."""
    # WHY the local import: audit.py pulls in scipy and sklearn; deferring
    # the import keeps `stnull --version` and `stnull cite` fast.
    from .audit import audit as _run_audit, check_inputs as _run_check
    # --- build the parser: one sub-parser per subcommand ----------------
    # WHAT: audit/nulls/levers/check share the _common flag set; nulls
    # drops --pred (it forwards P=None, flow [L53]); cite takes nothing.
    ap = argparse.ArgumentParser(
        prog="stnull",
        description="Audit spatial-expression predictions against zero-pixel "
                    "nulls. Every number is printed with the permutation floor "
                    "of its own readout rule.")
    ap.add_argument("--version", action="version", version="stnull " + VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _common(sub.add_parser("audit", help="full audit of a model's predictions"))
    _common(sub.add_parser("nulls", help="zero-pixel nulls only, no model"),
            with_pred=False)
    _common(sub.add_parser("levers", help="protocol lever price list"))
    _common(sub.add_parser("check", help="validate inputs only (seconds)"))
    sub.add_parser("cite", help="print the methods paragraph and defaults")

    a = ap.parse_args(argv)
    # --- cite: no data touched, print the Methods paragraph and leave ---
    if a.cmd == "cite":
        print(cite_text(None))  # None = package defaults (report.py:cite_text)
        return EXIT_OK
    # --- load inputs; any read failure is exit code 4, not a traceback --
    # WHERE: _kwargs performs the [L15]/[L16] loading and renaming; the
    # nulls subcommand skips --pred so P stays None (nulls-only mode).
    try:
        Y, P, kw = _kwargs(a, with_pred=(a.cmd != "nulls"))
    except (OSError, KeyError, StnullError) as e:
        print("[stnull] cannot read inputs: %s" % e, file=sys.stderr)
        return EXIT_INPUT
    try:
        # --- check: structural validation only, no correlation computed --
        # WHERE: check_inputs is audit.py:140 and returns the InputReport
        # of flow [L52]; the same gate runs at the top of every audit(),
        # so `stnull check` predicts exactly what a full run would reject.
        if a.cmd == "check":
            rep = _run_check(
                y_true=Y, y_pred=P, section=kw["section"],
                lib_size=kw.get("lib_size"), coords=kw.get("coords"),
                labels=kw.get("labels"), composition=kw.get("composition"),
                counts=kw.get("counts"), patient=kw.get("patient"),
                is_train=kw.get("is_train"), lib_pred=kw.get("lib_pred"),
                genes=kw.get("genes"), min_spots=a.min_spots, space=a.space)
            print(rep)
            return EXIT_OK if rep.ok else EXIT_STRICT
        # --- audit / nulls / levers all run the same full audit ---------
        # WHAT: Y, P are positional (the [L01]/[L02] entry), kw carries
        # every other keyword; the subcommands differ only in what they
        # print afterwards, never in what was computed.
        rep = _run_audit(Y, P, **kw)
    except StnullError as e:
        print("[stnull] %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return EXIT_STRICT
    # --- levers: print only the protocol lever price list ---------------
    # WHERE: lever_price is SelectionResult.lever_price, assembled at
    # audit.py:1329-1353 from the selection lever [L45] and the leakage
    # lever [L46]; a nulls-only run has no selection block to print.
    if a.cmd == "levers":
        if rep.selection is None:
            print("[stnull] no y_pred -> no levers")
            return EXIT_OK
        print(rep.selection.lever_price.to_string(index=False))
        if a.csv_dir:
            d = Path(a.csv_dir)
            d.mkdir(parents=True, exist_ok=True)
            rep.selection.lever_price.to_csv(d / "levers.csv", index=False)
            print("[wrote] %s" % (d / "levers.csv"))
        return EXIT_OK
    return _emit(rep, a)


if __name__ == "__main__":
    sys.exit(main())
