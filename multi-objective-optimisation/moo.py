#!/usr/bin/env python3
"""
Multi-Objective FRI Schedule Optimiser  v2
==========================================
Accounts for extension fields (Fp^6 / Fp^8) and SHA3-256/384/512
with Keccak-rate-aware hash cost modelling.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import csv, time, math

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("  ⚠ matplotlib not found — plots will be skipped.\n")


# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CostModel:
    """
    Parameterised cost model for FRI over extension fields with SHA3.

    Extension fields
    ----------------
    FRI folding operates over Fp^d.  Every coset element in the proof
    is an extension-field element of size  d × ceil(field_bits/8) bytes.

    Hash function
    -------------
    SHA3-256 / SHA3-384 / SHA3-512 (Keccak-f[1600]).
    Cost is measured in Keccak permutations, whose count depends on
    the sponge rate:
        SHA3-256  rate 136 B   digest 32 B
        SHA3-384  rate 104 B   digest 48 B
        SHA3-512  rate  72 B   digest 64 B
    """
    # ── Base field ──
    field_bits: int = 64                # 64: Goldilocks; 31: BabyBear/M31

    # ── Extension field ──
    ext_degree: int = 8                 # 6 for Fp^6, 8 for Fp^8
    ext_mul_basemuls: int = 24          # base-field muls per ext-field mul
                                        # Fp^6 Karatsuba ≈ 18; Fp^8 tower ≈ 24

    # ── Hash function (Merkle + Fiat-Shamir) ──
    hash_name: str = 'SHA3-256'         # SHA3-256, SHA3-384, SHA3-512

    # ── Protocol ──
    blowup_log:     int = 3
    num_queries:    int = 30
    final_poly_log: int = 0             # log2(final poly degree + 1)

    # ── Search-space bounds ──
    max_arity_log: int = 10
    min_arity_log: int = 1
    max_rounds:    int = 6

    # ── Prover cost weights ──
    prover_basemul_weight:     float = 1.0     # per base-field multiply
    prover_keccak_perm_weight: float = 50.0    # per Keccak-f[1600]

    # ── Verifier cost weights ──
    verifier_basemul_weight:     float = 1.0
    verifier_keccak_perm_weight: float = 50.0

    # ──── derived ────

    @property
    def hash_bits(self) -> int:
        return {'SHA3-256': 256, 'SHA3-384': 384, 'SHA3-512': 512}[self.hash_name]

    @property
    def hash_bytes(self) -> int:
        return self.hash_bits // 8

    @property
    def keccak_rate_bytes(self) -> int:
        return {'SHA3-256': 136, 'SHA3-384': 104, 'SHA3-512': 72}[self.hash_name]

    @property
    def ext_elem_bytes(self) -> int:
        return self.ext_degree * math.ceil(self.field_bits / 8)

    def keccak_perms(self, input_bytes: int) -> int:
        """Keccak-f permutations to absorb input_bytes (incl. SHA3 padding)."""
        return input_bytes // self.keccak_rate_bytes + 1

    def leaf_hash_perms(self, coset_size: int) -> int:
        return self.keccak_perms(coset_size * self.ext_elem_bytes)

    def node_hash_perms(self) -> int:
        return self.keccak_perms(2 * self.hash_bytes)

    def fs_perms(self) -> int:
        """Keccak perms for one Fiat-Shamir challenge derivation."""
        return self.keccak_perms(2 * self.hash_bytes)


PROFILES = {
    'proof_size': (5.0, 0.5, 0.5),
    'prover':     (0.5, 5.0, 0.5),
    'verifier':   (0.5, 0.5, 5.0),
    'balanced':   (1.0, 1.0, 1.0),
    'on_chain':   (3.0, 0.2, 5.0),
}


# ═══════════════════════════════════════════════════════════════════════
#  DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Schedule:
    arities_log:      List[int]
    proof_size_bytes: float = 0.0
    prover_time:      float = 0.0
    verifier_time:    float = 0.0
    proof_breakdown:  Optional[Dict] = None

    @property
    def arities(self) -> List[int]:
        return [1 << a for a in self.arities_log]

    @property
    def rounds(self) -> int:
        return len(self.arities_log)

    @property
    def label(self) -> str:
        return str(self.arities)

    def objs(self) -> Tuple[float, float, float]:
        return (self.proof_size_bytes, self.prover_time, self.verifier_time)

    def dominates(self, other: 'Schedule') -> bool:
        s, o = self.objs(), other.objs()
        return all(a <= b for a, b in zip(s, o)) and any(a < b for a, b in zip(s, o))


# ═══════════════════════════════════════════════════════════════════════
#  COST EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate(arities_log: List[int], n: int, m: CostModel,
             breakdown: bool = False) -> Schedule:
    """
    Proof size per query at round i
    --------------------------------
      coset:  2^{a_i}  Fp^d elements  ×  ext_elem_bytes
      auth:   (n − S_{i+1})  hash digests

    Prover per round
    ----------------
      fold:   (D_i × a_i / 2) ext-mul  → ×ext_mul_basemuls base muls
      Merkle: D_{i+1} leaf hashes  +  (D_{i+1}−1) node hashes  (Keccak perms)

    Verifier per query per round
    ----------------------------
      fold:   2^{a_i} ext-mul  → ×ext_mul_basemuls base muls
      auth:   1 leaf hash  +  (n − S_{i+1}) node hashes

    Fiat-Shamir:  L hash invocations  (prover + verifier)
    """
    L = len(arities_log)
    q = m.num_queries

    S = [0] * (L + 1)
    for i in range(L):
        S[i + 1] = S[i] + arities_log[i]

    # ═══ PROOF SIZE (bytes) ═══
    roots_bytes    = L * m.hash_bytes
    finalpoly_bytes = (1 << m.final_poly_log) * m.ext_elem_bytes

    round_coset_bytes = []
    round_auth_bytes  = []
    for i in range(L):
        cs = 1 << arities_log[i]
        depth = n - S[i + 1]
        round_coset_bytes.append(q * cs * m.ext_elem_bytes)
        round_auth_bytes.append(q * depth * m.hash_bytes)

    proof_bytes = (roots_bytes + finalpoly_bytes
                   + sum(round_coset_bytes) + sum(round_auth_bytes))

    bd = None
    if breakdown:
        bd = dict(roots=roots_bytes, final_poly=finalpoly_bytes,
                  round_coset=list(round_coset_bytes),
                  round_auth=list(round_auth_bytes),
                  total=proof_bytes)

    # ═══ PROVER ═══
    prover = 0.0
    for i in range(L):
        dom_before = 1 << (n - S[i])
        dom_after  = 1 << (n - S[i + 1])
        a_i = arities_log[i]
        cs  = 1 << a_i

        # fold NTT: dom_before × a_i / 2 ext-muls
        prover += m.prover_basemul_weight * (dom_before * a_i / 2) * m.ext_mul_basemuls
        # Merkle: leaf hashes + internal node hashes
        prover += m.prover_keccak_perm_weight * dom_after * m.leaf_hash_perms(cs)
        prover += m.prover_keccak_perm_weight * max(0, dom_after - 1) * m.node_hash_perms()

    # Fiat-Shamir
    prover += m.prover_keccak_perm_weight * L * m.fs_perms()

    # ═══ VERIFIER ═══
    vperq = 0.0
    for i in range(L):
        cs    = 1 << arities_log[i]
        depth = n - S[i + 1]
        # fold check
        vperq += m.verifier_basemul_weight * cs * m.ext_mul_basemuls
        # Merkle verify
        vperq += m.verifier_keccak_perm_weight * m.leaf_hash_perms(cs)
        vperq += m.verifier_keccak_perm_weight * depth * m.node_hash_perms()

    verifier = q * vperq + m.verifier_keccak_perm_weight * L * m.fs_perms()

    return Schedule(list(arities_log), proof_bytes, prover, verifier, bd)


# ═══════════════════════════════════════════════════════════════════════
#  ENUMERATION
# ═══════════════════════════════════════════════════════════════════════

def enumerate_all(n: int, m: CostModel) -> List[Schedule]:
    out: List[Schedule] = []
    for L in range(1, m.max_rounds + 1):
        _rec(n, L, m.min_arity_log, m.max_arity_log, [], n, m, out)
    return out

def _rec(rem, left, lo, hi, pfx, n, m, out):
    if left == 1:
        if lo <= rem <= hi:
            out.append(evaluate(pfx + [rem], n, m))
        return
    for a in range(lo, min(hi, rem - (left - 1) * lo) + 1):
        _rec(rem - a, left - 1, lo, hi, pfx + [a], n, m, out)


# ═══════════════════════════════════════════════════════════════════════
#  PARETO FRONT  (numpy-accelerated)
# ═══════════════════════════════════════════════════════════════════════

def pareto_front(scheds: List[Schedule]) -> List[Schedule]:
    if not scheds:
        return []
    objs = np.array([s.objs() for s in scheds])
    N = objs.shape[0]
    is_pareto = np.ones(N, dtype=bool)

    for i in range(N):
        if not is_pareto[i]:
            continue
        diffs = objs - objs[i]                         # N × 3
        le = np.all(diffs <= 0, axis=1)
        lt = np.any(diffs < 0, axis=1)
        dominators = le & lt & is_pareto
        dominators[i] = False
        if np.any(dominators):
            is_pareto[i] = False
            continue
        # prune points dominated BY i
        ge = np.all(diffs >= 0, axis=1)
        gt = np.any(diffs > 0, axis=1)
        dominated = ge & gt & is_pareto
        dominated[i] = False
        is_pareto &= ~dominated

    return [scheds[i] for i in range(N) if is_pareto[i]]


def weighted_best(scheds, wp=1., wt=1., wv=1.):
    if not scheds:
        return None
    ps = [s.proof_size_bytes for s in scheds]
    pt = [s.prover_time      for s in scheds]
    vt = [s.verifier_time    for s in scheds]

    def nrm(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0.0

    best, best_sc = None, float('inf')
    for s in scheds:
        sc = (wp * nrm(s.proof_size_bytes, min(ps), max(ps)) +
              wt * nrm(s.prover_time,      min(pt), max(pt)) +
              wv * nrm(s.verifier_time,    min(vt), max(vt)))
        if sc < best_sc:
            best_sc, best = sc, s
    return best


# ═══════════════════════════════════════════════════════════════════════
#  BASELINES
# ═══════════════════════════════════════════════════════════════════════

def fixed_baseline(n: int, arity_log: int, m: CostModel) -> Schedule:
    full = n // arity_log
    rem  = n %  arity_log
    if rem == 0:
        a = [arity_log] * full
    elif rem >= m.min_arity_log:
        a = [arity_log] * full + [rem]
    elif full > 0:
        a = [arity_log] * (full - 1) + [arity_log + rem]
    else:
        a = [n]
    return evaluate(a, n, m, breakdown=True)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════

def run(k_range, m: CostModel, verbose=True) -> Dict:
    results = {}
    t0 = time.time()

    if verbose:
        print("=" * 115)
        print("  Multi-Objective FRI Schedule Optimiser  v2")
        print("=" * 115)
        print(f"  Field: {m.field_bits}-bit │ Extension: Fp^{m.ext_degree} "
              f"({m.ext_elem_bytes} B/elem) │ Hash: {m.hash_name} "
              f"({m.hash_bytes} B digest, {m.keccak_rate_bytes} B rate)")
        print(f"  Blowup: 2^{m.blowup_log} │ Queries: {m.num_queries} │ "
              f"Ext mul: {m.ext_mul_basemuls} base muls │ "
              f"Node hash: {m.node_hash_perms()} Keccak perms")
        print(f"  Arity: {1 << m.min_arity_log}…{1 << m.max_arity_log} │ "
              f"Max rounds: {m.max_rounds}")

    for k in k_range:
        n = k + m.blowup_log
        everything = enumerate_all(n, m)
        front      = pareto_front(everything)
        front_bd   = [evaluate(s.arities_log, n, m, breakdown=True) for s in front]
        bests      = {nm: weighted_best(front_bd, *w) for nm, w in PROFILES.items()}
        bases = {
            'binary':  fixed_baseline(n, 1, m),
            'arity-4': fixed_baseline(n, 2, m),
            'arity-8': fixed_baseline(n, 3, m),
        }
        results[k] = dict(all=everything, pareto=front_bd, bests=bests,
                          bases=bases, n=n)
        if verbose:
            _report(k, n, everything, front_bd, bests, bases)

    if verbose:
        print(f"\n  Completed in {time.time() - t0:.2f}s\n")
    return results


def _report(k, n, everything, front, bests, bases):
    print(f"\n{'─' * 115}")
    print(f"  k={k:>2}  │  domain 2^{n:>2}  │  "
          f"schedules {len(everything):>6}  │  Pareto front {len(front):>4}")
    print(f"  {'Profile':<14} {'Schedule':<40} {'Rnds':>4} "
          f"{'Proof KB':>10} {'Prover':>14} {'Verifier':>14}")
    print(f"  {'─'*14} {'─'*40} {'─'*4} {'─'*10} {'─'*14} {'─'*14}")
    for nm in PROFILES:
        s = bests.get(nm)
        if s:
            print(f"  {nm:<14} {s.label:<40} {s.rounds:>4} "
                  f"{s.proof_size_bytes/1024:>10.1f} "
                  f"{s.prover_time:>14.3e} {s.verifier_time:>14.3e}")
    print(f"  {'─'*14} {'─'*40} {'─'*4} {'─'*10} {'─'*14} {'─'*14}")
    for bn, s in bases.items():
        print(f"  {bn:<14} {s.label:<40} {s.rounds:>4} "
              f"{s.proof_size_bytes/1024:>10.1f} "
              f"{s.prover_time:>14.3e} {s.verifier_time:>14.3e}")


# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════

def print_summary(results: Dict, m: CostModel):
    k_vals = sorted(results.keys())

    print("\n" + "=" * 115)
    print(f"  SUMMARY — Fp^{m.ext_degree} / {m.hash_name} / "
          f"blowup 2^{m.blowup_log} / {m.num_queries} queries")
    print("=" * 115)

    for pname in PROFILES:
        print(f"\n  Profile: {pname}  (weights: proof={PROFILES[pname][0]}, "
              f"prover={PROFILES[pname][1]}, verifier={PROFILES[pname][2]})")
        print(f"  {'k':>4}  {'Schedule':<40} {'Rnds':>4}  "
              f"{'Proof KB':>9} {'Prover':>13} {'Verifier':>13}")
        print(f"  {'─'*4}  {'─'*40} {'─'*4}  {'─'*9} {'─'*13} {'─'*13}")
        for k in k_vals:
            s = results[k]['bests'].get(pname)
            if s:
                print(f"  {k:>4}  {s.label:<40} {s.rounds:>4}  "
                      f"{s.proof_size_bytes/1024:>9.1f} "
                      f"{s.prover_time:>13.3e} {s.verifier_time:>13.3e}")

    # ── Proof-size breakdown for balanced profile ──
    print(f"\n  {'─'*115}")
    print(f"  PROOF SIZE BREAKDOWN  (balanced profile)")
    print(f"  {'k':>4}  {'Schedule':<30} {'Roots':>7} {'FinPoly':>8} "
          f"{'Cosets':>10} {'Auth':>10} {'Total KB':>10}")
    print(f"  {'─'*4}  {'─'*30} {'─'*7} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    for k in k_vals:
        s = results[k]['bests'].get('balanced')
        if s and s.proof_breakdown:
            bd = s.proof_breakdown
            print(f"  {k:>4}  {s.label:<30} {bd['roots']:>7} "
                  f"{bd['final_poly']:>8} "
                  f"{sum(bd['round_coset']):>10} "
                  f"{sum(bd['round_auth']):>10} "
                  f"{bd['total']/1024:>10.1f}")


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def make_plots(results: Dict, m: CostModel, prefix='fri_moo'):
    if not HAS_MPL:
        print("  Skipping plots (matplotlib not installed).")
        return

    k_vals = sorted(results.keys())
    clr = dict(proof_size='#2176AE', prover='#D7263D', verifier='#3A8628',
               balanced='#7B2D8E', on_chain='#F5820D')
    getters = [
        ('Proof Size (KB)', lambda s: s.proof_size_bytes / 1024),
        ('Prover Time',     lambda s: s.prover_time),
        ('Verifier Time',   lambda s: s.verifier_time),
    ]
    subtitle = (f'Fp^{m.ext_degree} / {m.hash_name} / '
                f'blowup 2^{m.blowup_log} / {m.num_queries}q')

    # ── 1. Objective trajectories ──
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, (lab, g) in zip(axes, getters):
        for pn in PROFILES:
            vals = [g(results[k]['bests'][pn])
                    if results[k]['bests'].get(pn) else np.nan for k in k_vals]
            ax.plot(k_vals, vals, '-o', ms=4, label=pn, color=clr[pn])
        for bn, ls in [('binary','--'), ('arity-4','-.'), ('arity-8',':')]:
            vals = [g(results[k]['bases'][bn]) for k in k_vals]
            ax.plot(k_vals, vals, ls, color='gray', alpha=.55, label=f'{bn} fixed')
        ax.set_xlabel('k (log₂ trace)'); ax.set_ylabel(lab); ax.set_title(lab)
        if 'Time' in lab:
            ax.set_yscale('log')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=.25)
    plt.suptitle(f'Optimal FRI Objectives — {subtitle}', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{prefix}_profiles.png', dpi=150, bbox_inches='tight')
    print(f'    {prefix}_profiles.png')

    # ── 2. Arity heatmap per profile ──
    fig, axes = plt.subplots(len(PROFILES), 1,
                              figsize=(14, 2.8 * len(PROFILES)), sharex=True)
    if len(PROFILES) == 1:
        axes = [axes]
    for ax, pname in zip(axes, PROFILES):
        max_r = max((results[k]['bests'][pname].rounds
                     for k in k_vals if results[k]['bests'].get(pname)),
                    default=1)
        grid = np.full((max_r, len(k_vals)), np.nan)
        for j, k in enumerate(k_vals):
            s = results[k]['bests'].get(pname)
            if s:
                for r, a in enumerate(s.arities_log):
                    grid[r, j] = a
        im = ax.imshow(grid, aspect='auto', cmap='YlOrRd',
                        interpolation='nearest', vmin=1, vmax=m.max_arity_log,
                        extent=[k_vals[0]-0.5, k_vals[-1]+0.5,
                                max_r - 0.5, -0.5])
        for r in range(max_r):
            for j, k in enumerate(k_vals):
                if not np.isnan(grid[r, j]):
                    a = int(grid[r, j])
                    ax.text(k, r, f'{1 << a}', ha='center', va='center',
                            fontsize=7, fontweight='bold',
                            color='white' if a > 5 else 'black')
        ax.set_ylabel(f'{pname}\nRound')
        ax.set_yticks(range(max_r))
        plt.colorbar(im, ax=ax, label='log₂ arity', shrink=0.8)
    axes[-1].set_xlabel('k (log₂ trace)')
    plt.suptitle(f'Arity Schedule per Round — {subtitle}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{prefix}_arity_heatmap.png', dpi=150, bbox_inches='tight')
    print(f'    {prefix}_arity_heatmap.png')

    # ── 3. Proof-size breakdown stacked bar (balanced) ──
    fig, ax = plt.subplots(figsize=(15, 6))
    bot = np.zeros(len(k_vals))
    # Ensure breakdowns exist
    for ki, k in enumerate(k_vals):
        s = results[k]['bests'].get('balanced')
        if s and not s.proof_breakdown:
            s2 = evaluate(s.arities_log, results[k]['n'], m, breakdown=True)
            s.proof_breakdown = s2.proof_breakdown

    # roots
    vals = [results[k]['bests']['balanced'].proof_breakdown['roots'] / 1024
            if results[k]['bests'].get('balanced') and
               results[k]['bests']['balanced'].proof_breakdown else 0
            for k in k_vals]
    ax.bar(k_vals, vals, width=0.7, bottom=bot, label='Merkle roots',
           color='#264653')
    bot += vals

    # final poly
    vals = [results[k]['bests']['balanced'].proof_breakdown['final_poly'] / 1024
            if results[k]['bests'].get('balanced') and
               results[k]['bests']['balanced'].proof_breakdown else 0
            for k in k_vals]
    ax.bar(k_vals, vals, width=0.7, bottom=bot, label='Final poly',
           color='#2A9D8F')
    bot += vals

    # per-round coset + auth
    max_rr = max((results[k]['bests']['balanced'].rounds
                  for k in k_vals
                  if results[k]['bests'].get('balanced')), default=1)
    coset_cm = ['#E9C46A', '#F4A261', '#E76F51', '#D62828', '#6A0572', '#1B998B']
    auth_cm  = ['#FFF3B0', '#FDB863', '#E08214', '#B35806', '#7B3294', '#1B7837']
    for r in range(max_rr):
        c_kb, a_kb = [], []
        sched_labels = []
        for k in k_vals:
            s = results[k]['bests'].get('balanced')
            bd = s.proof_breakdown if s else None
            if bd and r < len(bd['round_coset']):
                c_kb.append(bd['round_coset'][r] / 1024)
                a_kb.append(bd['round_auth'][r] / 1024)
                sched_labels.append(s.arities[r] if r < s.rounds else '')
            else:
                c_kb.append(0)
                a_kb.append(0)
                sched_labels.append('')
        ci = r % len(coset_cm)
        ai = r % len(auth_cm)
        bars_c = ax.bar(k_vals, c_kb, width=0.7, bottom=bot,
                        label=f'R{r} coset', color=coset_cm[ci],
                        edgecolor='white', linewidth=0.3)
        # annotate coset bars with actual arity
        for rect, lbl, c in zip(bars_c, sched_labels, c_kb):
            if c > 0 and lbl:
                cx = rect.get_x() + rect.get_width() / 2
                cy = rect.get_y() + rect.get_height() / 2
                ax.text(cx, cy, str(lbl), ha='center', va='center',
                        fontsize=6, fontweight='bold')
        bot = [b + c for b, c in zip(bot, c_kb)]
        ax.bar(k_vals, a_kb, width=0.7, bottom=bot,
               label=f'R{r} auth', color=auth_cm[ai], alpha=0.7,
               edgecolor='white', linewidth=0.3)
        bot = [b + a for b, a in zip(bot, a_kb)]

    ax.set_xlabel('k (log₂ trace)')
    ax.set_ylabel('Proof Size (KB)')
    ax.set_title(f'Proof Size Breakdown (balanced) — {subtitle}')
    ax.legend(fontsize=6, ncol=4, loc='upper left')
    ax.grid(True, alpha=.25, axis='y')
    plt.tight_layout()
    plt.savefig(f'{prefix}_breakdown.png', dpi=150, bbox_inches='tight')
    print(f'    {prefix}_breakdown.png')

    # ── 4. Stacked arity bars with labels ──
    fig, ax = plt.subplots(figsize=(15, 6))
    for k in k_vals:
        s = results[k]['bests'].get('balanced')
        if not s:
            continue
        bot_a = 0
        for r, a in enumerate(s.arities_log):
            ax.bar(k, a, bottom=bot_a, width=.7, color=f'C{r}',
                   label=f'Round {r}' if k == k_vals[0] else '')
            ax.text(k, bot_a + a / 2, f'{1 << a}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')
            bot_a += a
    ax.set_xlabel('k (log₂ trace)')
    ax.set_ylabel('log₂(arity)')
    ax.set_title(f'Balanced — Arity per Round (labels = actual fold factor) — {subtitle}')
    h, l = ax.get_legend_handles_labels()
    by_lbl = dict(zip(l, h))
    ax.legend(by_lbl.values(), by_lbl.keys())
    ax.grid(True, alpha=.25, axis='y')
    plt.tight_layout()
    plt.savefig(f'{prefix}_schedules.png', dpi=150, bbox_inches='tight')
    print(f'    {prefix}_schedules.png')

    # ── 5. Pareto fronts ──
    for pk in [14, 18, 22]:
        if pk not in results or len(results[pk]['all']) < 5:
            continue
        A = np.array([s.objs() for s in results[pk]['all']])
        P = np.array([s.objs() for s in results[pk]['pareto']])
        A[:, 0] /= 1024; P[:, 0] /= 1024
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        pairs = [(0, 2, 'Proof KB', 'Verifier Time'),
                 (0, 1, 'Proof KB', 'Prover Time'),
                 (1, 2, 'Prover Time', 'Verifier Time')]
        for ax, (xi, yi, xl, yl) in zip(axes, pairs):
            ax.scatter(A[:, xi], A[:, yi], s=6, alpha=.10, c='gray', label='All')
            ax.scatter(P[:, xi], P[:, yi], s=30, c='red', zorder=5,
                       edgecolors='darkred', lw=.4, label='Pareto')
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.legend(fontsize=8); ax.grid(True, alpha=.25)
        plt.suptitle(f'Pareto Front  k={pk} — {subtitle}', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'{prefix}_pareto_k{pk}.png', dpi=150, bbox_inches='tight')
        print(f'    {prefix}_pareto_k{pk}.png')

    # ── 6. % improvement vs baselines ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, g) in zip(axes, getters):
        imp_b, imp_4, imp_8 = [], [], []
        for k in k_vals:
            opt = results[k]['bests'].get('balanced')
            b2  = results[k]['bases']['binary']
            b4  = results[k]['bases']['arity-4']
            b8  = results[k]['bases']['arity-8']
            vo = g(opt) if opt else None
            imp_b.append((1 - vo / g(b2)) * 100 if vo else 0)
            imp_4.append((1 - vo / g(b4)) * 100 if vo else 0)
            imp_8.append((1 - vo / g(b8)) * 100 if vo else 0)
        w_bar = 0.25
        x_arr = np.array(k_vals, dtype=float)
        ax.bar(x_arr - w_bar, imp_b, w_bar, label='vs Binary',
               color='steelblue', alpha=.85)
        ax.bar(x_arr,         imp_4, w_bar, label='vs Arity-4',
               color='coral', alpha=.85)
        ax.bar(x_arr + w_bar, imp_8, w_bar, label='vs Arity-8',
               color='seagreen', alpha=.85)
        ax.axhline(0, color='k', lw=.5)
        ax.set_xlabel('k'); ax.set_ylabel('Improvement %'); ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(True, alpha=.25, axis='y')
    plt.suptitle(f'Balanced vs Fixed-Arity Baselines — {subtitle}',
                 y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{prefix}_improvement.png', dpi=150, bbox_inches='tight')
    print(f'    {prefix}_improvement.png')

    plt.close('all')


# ═══════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════

def export_csv(results: Dict, m: CostModel, fname='fri_schedules.csv'):
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k', 'ext_degree', 'hash', 'profile', 'schedule',
                     'rounds', 'proof_KB', 'prover_time', 'verifier_time'])
        for k in sorted(results.keys()):
            for pn in PROFILES:
                s = results[k]['bests'].get(pn)
                if s:
                    w.writerow([k, m.ext_degree, m.hash_name, pn,
                                s.label, s.rounds,
                                f'{s.proof_size_bytes/1024:.2f}',
                                f'{s.prover_time:.4e}',
                                f'{s.verifier_time:.4e}'])
            for bn, s in results[k]['bases'].items():
                w.writerow([k, m.ext_degree, m.hash_name, f'base_{bn}',
                            s.label, s.rounds,
                            f'{s.proof_size_bytes/1024:.2f}',
                            f'{s.prover_time:.4e}',
                            f'{s.verifier_time:.4e}'])
    print(f'    {fname}')


def export_pareto_csv(results: Dict, m: CostModel,
                      fname='fri_pareto_all.csv'):
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k', 'ext_degree', 'hash', 'schedule', 'rounds',
                     'proof_KB', 'prover_time', 'verifier_time'])
        for k in sorted(results.keys()):
            for s in sorted(results[k]['pareto'],
                            key=lambda s: s.proof_size_bytes):
                w.writerow([k, m.ext_degree, m.hash_name,
                            s.label, s.rounds,
                            f'{s.proof_size_bytes/1024:.2f}',
                            f'{s.prover_time:.4e}',
                            f'{s.verifier_time:.4e}'])
    print(f'    {fname}')


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    model = CostModel(
        field_bits          = 64,       # Goldilocks
        ext_degree          = 8,        # Fp^8
        ext_mul_basemuls    = 24,       # tower-based Fp^8 mul
        hash_name           = 'SHA3-256',
        blowup_log          = 3,        # ×8 blowup
        num_queries         = 30,
        final_poly_log      = 0,
        max_arity_log       = 10,
        min_arity_log       = 1,
        max_rounds          = 6,
        prover_basemul_weight     = 1.0,
        prover_keccak_perm_weight = 50.0,
        verifier_basemul_weight     = 1.0,
        verifier_keccak_perm_weight = 50.0,
    )

    k_range = range(10, 24)              # 2^10 … 2^23

    results = run(k_range, model, verbose=True)
    print_summary(results, model)

    print('\n  Generating plots…')
    make_plots(results, model)

    print('\n  Exporting CSV…')
    export_csv(results, model)
    export_pareto_csv(results, model)

    # ── Quick comparison: swap hash to see impact ──
    print('\n' + '▓' * 115)
    print('  SHA3 variant comparison  (balanced profile, k=20)')
    print('▓' * 115)
    for hname in ['SHA3-256', 'SHA3-384', 'SHA3-512']:
        m2 = CostModel(
            field_bits=64, ext_degree=8, ext_mul_basemuls=24,
            hash_name=hname, blowup_log=3, num_queries=30,
            max_arity_log=10, min_arity_log=1, max_rounds=6,
            prover_basemul_weight=1.0, prover_keccak_perm_weight=50.0,
            verifier_basemul_weight=1.0, verifier_keccak_perm_weight=50.0,
        )
        n = 20 + m2.blowup_log
        everything = enumerate_all(n, m2)
        front = pareto_front(everything)
        front_bd = [evaluate(s.arities_log, n, m2, breakdown=True) for s in front]
        best = weighted_best(front_bd, 1, 1, 1)
        if best:
            print(f"  {hname:<12} {best.label:<35}  "
                  f"proof {best.proof_size_bytes/1024:>8.1f} KB  "
                  f"prover {best.prover_time:>12.3e}  "
                  f"verifier {best.verifier_time:>12.3e}")

    # ── Fp^6 vs Fp^8 ──
    print(f'\n  Extension degree comparison  (SHA3-256, balanced, k=20)')
    for ext_d, mul_c in [(6, 18), (8, 24)]:
        m2 = CostModel(
            field_bits=64, ext_degree=ext_d, ext_mul_basemuls=mul_c,
            hash_name='SHA3-256', blowup_log=3, num_queries=30,
            max_arity_log=10, min_arity_log=1, max_rounds=6,
            prover_basemul_weight=1.0, prover_keccak_perm_weight=50.0,
            verifier_basemul_weight=1.0, verifier_keccak_perm_weight=50.0,
        )
        n = 20 + m2.blowup_log
        everything = enumerate_all(n, m2)
        front = pareto_front(everything)
        front_bd = [evaluate(s.arities_log, n, m2, breakdown=True) for s in front]
        best = weighted_best(front_bd, 1, 1, 1)
        if best:
            print(f"  Fp^{ext_d:<3}       {best.label:<35}  "
                  f"proof {best.proof_size_bytes/1024:>8.1f} KB  "
                  f"prover {best.prover_time:>12.3e}  "
                  f"verifier {best.verifier_time:>12.3e}")

    print('\n  ✓ Done.\n')


if __name__ == '__main__':
    main()