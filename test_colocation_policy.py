"""Tests for the second-label selection policies (loop.py, F17/F18).

The repo's silent-defeat rule: after wiring a flag, verify the value arrives
where it is consumed AND that it changes what it should. Four of the five bugs
in that class on this branch were flags that were accepted and never forwarded.

These tests drive ``BOLoop._secondary_label_mask`` directly on a synthetic
library, so nothing docks and nothing fits a GP.
"""
import numpy as np
import pytest

import loop as L


def make_loop(policy="hash", fraction=0.25, floor=0, seed=42,
              n_library=400, n_evaluated=0, n_colocated=0, fp_bits=2048):
    """A BOLoop with only the attributes ``_secondary_label_mask`` reads.

    Built with ``__new__`` so no library, oracle or GP is constructed. Two
    chemical clusters, so 'diverse' and 'central' have something to disagree
    about.
    """
    rng = np.random.default_rng(0)
    fps = np.zeros((n_library, fp_bits), dtype=np.int8)
    # Two clusters with a SHARED 20-bit signature each plus 10 unique bits, so
    # within-cluster Tanimoto is 0.5 and across-cluster is exactly 0. Cluster A
    # is deliberately the LARGER one (indices < CUT), which is what gives
    # 'central' something to prefer: mean similarity to the batch is higher
    # inside the bigger cluster.
    CUT = int(n_library * 0.7)
    sig_a = np.arange(0, 20)
    sig_b = np.arange(fp_bits - 20, fp_bits)
    free_a = np.arange(20, (fp_bits // 2))
    free_b = np.arange((fp_bits // 2), fp_bits - 20)
    for i in range(n_library):
        sig, free = (sig_a, free_a) if i < CUT else (sig_b, free_b)
        fps[i, sig] = 1
        fps[i, rng.choice(free, size=10, replace=False)] = 1

    self = L.BOLoop.__new__(L.BOLoop)
    self.fingerprints = fps
    self.seed = seed
    self.hdhfr_fraction = fraction
    self.hdhfr_policy = policy
    self.hdhfr_floor = floor
    self.evaluated_indices = list(range(n_evaluated))
    hd_col = L.DOCKING_TASKS[1][0]
    Y = np.full((n_evaluated, L.N_OBJECTIVES), np.nan, dtype=np.float64)
    if n_colocated:
        Y[:n_colocated, hd_col] = -7.0
    self.Y_evaluated = Y
    return self


# --- the OFF path must be untouched -------------------------------------------

def test_hash_policy_is_bit_identical_to_the_historical_rule():
    """'hash' is the default and must reproduce the pre-F18 code exactly."""
    lp = make_loop(policy="hash", fraction=0.25, seed=42)
    idx = list(range(100, 140))
    got = lp._secondary_label_mask(idx)
    want = L._splitmix64_unit(np.asarray(idx, dtype=np.int64), 42) < 0.25
    assert np.array_equal(got, want)


def test_hash_policy_ignores_the_floor():
    """A per-molecule hash has no running total, so it cannot honour a floor.
    Documented behaviour, pinned so it is not mistaken for a bug later."""
    a = make_loop(policy="hash", fraction=0.01, floor=0)
    b = make_loop(policy="hash", fraction=0.01, floor=50)
    idx = list(range(200))
    assert np.array_equal(a._secondary_label_mask(idx),
                          b._secondary_label_mask(idx))


def test_default_policy_is_hash():
    assert L.SECONDARY_LABEL_POLICIES[0] == "hash"


# --- the floor closes the identifiability hole --------------------------------

@pytest.mark.parametrize("policy", ["uniform", "diverse", "central"])
def test_floor_is_met_on_the_first_batch(policy):
    """5% of a 10-molecule init rounds to 1, and a hash would often give 0.
    The floor must deliver 8, F17's advantage peak."""
    lp = make_loop(policy=policy, fraction=0.05, floor=8)
    mask = lp._secondary_label_mask(list(range(10)))
    assert mask.sum() == 8


@pytest.mark.parametrize("policy", ["uniform", "diverse", "central"])
def test_quota_never_reaches_zero_while_the_rate_is_positive(policy):
    """The quota rounds UP, so any positive rate yields at least one co-located
    molecule. This is why the floor is NOT about avoiding an unfittable model --
    the ceil already prevents that. The floor is about reaching F17's peak: at
    k=1 the learned task correlation has sd 0.693 across repeats and the ICM's
    advantage is not significant (p=0.277), so one co-located molecule is fitted
    but useless."""
    lp = make_loop(policy=policy, fraction=0.001, floor=0)
    assert lp._secondary_label_mask(list(range(10))).sum() == 1


@pytest.mark.parametrize("policy", ["uniform", "diverse", "central"])
def test_hash_by_contrast_can_reach_zero(policy):
    """The hole the quota form closes. Over 200 independent 10-molecule batches
    at a 5% per-molecule rate, the hash hands the ICM an unfittable design --
    zero observations on the second task -- in 119 of them. That is 59.5%, and
    it matches the binomial 0.95^10 = 0.599, so it is a property of the
    per-molecule design, not an unlucky seed."""
    zeros = 0
    for b in range(200):
        lp = make_loop(policy="hash", fraction=0.05, seed=b)
        if lp._secondary_label_mask(list(range(10 * b, 10 * b + 10))).sum() == 0:
            zeros += 1
    assert zeros == 119, f"expected 119/200 empty batches, got {zeros}"
    assert abs(zeros / 200 - 0.95 ** 10) < 0.05   # the binomial it should match


@pytest.mark.parametrize("policy", ["uniform", "diverse", "central"])
def test_floor_never_exceeds_the_batch(policy):
    lp = make_loop(policy=policy, fraction=0.05, floor=99)
    assert lp._secondary_label_mask(list(range(6))).sum() == 6


# --- the quota tracks the cumulative rate -------------------------------------

@pytest.mark.parametrize("policy", ["uniform", "diverse", "central"])
def test_quota_hits_the_cumulative_target_exactly(policy):
    """90 evaluated with 4 co-located, batch of 10, target 10% of 100 = 10,
    so this batch must supply exactly 6."""
    lp = make_loop(policy=policy, fraction=0.10, n_evaluated=90, n_colocated=4)
    assert lp._secondary_label_mask(list(range(200, 210))).sum() == 6


@pytest.mark.parametrize("policy", ["uniform", "diverse", "central"])
def test_quota_is_zero_once_the_rate_is_already_met(policy):
    lp = make_loop(policy=policy, fraction=0.10, n_evaluated=90, n_colocated=40)
    assert lp._secondary_label_mask(list(range(200, 210))).sum() == 0


def test_quota_policy_lands_in_f17s_band_over_a_whole_campaign():
    """The point of the whole change: a 5% target must hold the co-location rate
    inside F17's 3.6%-to-9.8% band at every iteration, not just on average."""
    lp = make_loop(policy="uniform", fraction=0.05, floor=8)
    hd_col = L.DOCKING_TASKS[1][0]
    nxt = 0
    rates = []
    for it in range(29):                       # 10 init + 28 x 10 = 290
        size = 10
        batch = list(range(nxt, nxt + size)); nxt += size
        mask = lp._secondary_label_mask(batch)
        rows = np.full((size, L.N_OBJECTIVES), np.nan)
        rows[mask, hd_col] = -7.0
        lp.Y_evaluated = np.vstack([lp.Y_evaluated, rows])
        lp.evaluated_indices.extend(batch)
        n = len(lp.evaluated_indices)
        rates.append(np.isfinite(lp.Y_evaluated[:, hd_col]).sum() / n)
    # The floor dominates early (8/10 = 80%), so judge from the point where the
    # rate becomes the binding constraint, i.e. once 5% of n exceeds the floor.
    late = [r for r, n in zip(rates, range(20, 300, 10)) if 0.05 * n >= 8]
    assert late, "no iteration where the rate binds"
    assert max(late) <= 0.098 + 1e-9, f"above F17's band: {max(late):.4f}"
    assert min(late) >= 0.036 - 1e-9, f"below F17's band: {min(late):.4f}"


def test_quota_policy_is_reproducible():
    a = make_loop(policy="uniform", fraction=0.10, n_evaluated=90, n_colocated=4)
    b = make_loop(policy="uniform", fraction=0.10, n_evaluated=90, n_colocated=4)
    idx = list(range(200, 210))
    assert np.array_equal(a._secondary_label_mask(idx), b._secondary_label_mask(idx))


def test_quota_policy_differs_between_seeds():
    """loop.py once had no --seed at all and six 'different' seeds gave six
    identical runs. Any new stochastic choice must be shown to move with it."""
    a = make_loop(policy="uniform", fraction=0.20, n_evaluated=90, seed=1)
    b = make_loop(policy="uniform", fraction=0.20, n_evaluated=90, seed=2)
    idx = list(range(200, 260))
    assert not np.array_equal(a._secondary_label_mask(idx),
                              b._secondary_label_mask(idx))


# --- the chemistry policies must actually differ -------------------------------

def test_diverse_and_central_choose_different_molecules():
    """If the two policies agreed, F18 would have compared a thing with itself."""
    idx = list(range(210, 310))              # spans both synthetic clusters
    dv = make_loop(policy="diverse", fraction=0.10, n_evaluated=90)._secondary_label_mask(idx)
    ct = make_loop(policy="central", fraction=0.10, n_evaluated=90)._secondary_label_mask(idx)
    # ceil(0.10 * (90 + 100)) = 19, none co-located yet, so both spend 19 docks.
    assert dv.sum() == ct.sum() == 19         # matched cost, as F18 requires
    assert not np.array_equal(dv, ct)


def test_diverse_spans_both_clusters_and_central_does_not():
    """The knob F18 varied, asserted on the synthetic library where the right
    answer is known: molecules < 280 are cluster A, >= 280 are cluster B."""
    idx = list(range(210, 310))
    dv = make_loop(policy="diverse", fraction=0.10, n_evaluated=90)._secondary_label_mask(idx)
    ct = make_loop(policy="central", fraction=0.10, n_evaluated=90)._secondary_label_mask(idx)
    chosen_dv = np.asarray(idx)[dv]
    chosen_ct = np.asarray(idx)[ct]
    assert (chosen_dv < 280).any() and (chosen_dv >= 280).any(), chosen_dv
    assert ((chosen_ct < 280).all() or (chosen_ct >= 280).all()), chosen_ct
    # The quantitative version of the same claim, and the knob F18 varied:
    # mean pairwise Tanimoto DISTANCE inside the chosen subset.
    S_dv = L._tanimoto(make_loop().fingerprints[chosen_dv],
                       make_loop().fingerprints[chosen_dv])
    S_ct = L._tanimoto(make_loop().fingerprints[chosen_ct],
                       make_loop().fingerprints[chosen_ct])
    iu_dv = np.triu_indices(len(chosen_dv), 1)
    iu_ct = np.triu_indices(len(chosen_ct), 1)
    assert (1 - S_dv[iu_dv]).mean() > (1 - S_ct[iu_ct]).mean()


def test_diverse_accumulates_coverage_against_already_labelled_molecules():
    """Coverage is cumulative: given cluster-A molecules already labelled, the
    next pick must come from cluster B."""
    lp = make_loop(policy="diverse", fraction=0.10, n_evaluated=90, n_colocated=9)
    # evaluated_indices 0..89 are cluster A (< 280); the first 9 carry hDHFR.
    mask = lp._secondary_label_mask(list(range(210, 310)))
    chosen = np.asarray(list(range(210, 310)))[mask]
    # Not ALL of them. Once one cluster-B molecule is picked, the remaining B
    # candidates sit at distance 0.5 from it, which is the same order as their
    # distance from the labelled A molecules, so later picks are nearly
    # indifferent. The claim is that coverage moves to the UNLABELLED cluster
    # first and more often than a policy blind to what is already labelled,
    # not that it ignores the labelled cluster entirely.
    # A boolean mask loses pick ORDER, so the claim is about composition, not
    # about which molecule came first.
    assert (chosen >= 280).any(), f"no cluster-B molecule chosen: {chosen}"
    # And the comparison that matters: a policy blind to what is already
    # labelled has no reason to prefer cluster B.
    blind = make_loop(policy="central", fraction=0.10, n_evaluated=90,
                      n_colocated=9)._secondary_label_mask(list(range(210, 310)))
    blind_chosen = np.asarray(list(range(210, 310)))[blind]
    assert (chosen >= 280).mean() > (blind_chosen >= 280).mean()


# --- validation ----------------------------------------------------------------

def test_tanimoto_matches_the_definition():
    a = np.zeros((1, 8)); a[0, :4] = 1
    b = np.zeros((1, 8)); b[0, 2:6] = 1
    assert L._tanimoto(a, b)[0, 0] == pytest.approx(2 / 6)


def test_tanimoto_of_a_row_with_itself_is_one():
    x = np.zeros((3, 16)); x[0, :5] = 1; x[1, 5:9] = 1; x[2, ::2] = 1
    assert np.allclose(np.diag(L._tanimoto(x, x)), 1.0)


def test_farthest_point_returns_distinct_positions():
    rng = np.random.default_rng(1)
    fps = (rng.random((40, 64)) > 0.7).astype(np.float32)
    sel = L._farthest_point(fps, 12)
    assert len(sel) == 12 and len(set(sel.tolist())) == 12


def test_farthest_point_caps_at_the_pool_size():
    fps = np.eye(5, 64, dtype=np.float32)
    assert len(L._farthest_point(fps, 99)) == 5
