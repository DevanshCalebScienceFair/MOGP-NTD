"""Hand-derived VALUE tests: hypervolume, GP posterior, task covariance, EHVI.

Why this file exists. The other suites pin these quantities against frozen
fixtures and against live library oracles (botorch / gpytorch). A fixture
captured from the code under test, or a library sharing a misreading with it,
cannot catch a wrong *value*. Every expected number below is instead derived
here, from the mathematics, in the comments beside it: areas and volumes by
arithmetic, the GP and ICM posteriors by an explicit kernel matrix and
``np.linalg.solve`` (Rasmussen & Williams eq. 2.25 / 2.26), the task covariance
by its definition, and the EHVI by a closed-form expectation. No expected value
is produced by calling the function under test, a library, or an archived fixture.

Fingerprints are chosen so the Tanimoto Gram is exact small fractions:

    a = [1,1,0,0]   b = [1,0,1,0]   c = [1,1,1,1]        (training set)
    k(x, y) = <x,y> / (|x| + |y| - <x,y>)  for 0/1 vectors

    k(a,b) = 1/(2+2-1) = 1/3     k(a,c) = 2/(2+4-2) = 1/2     k(b,c) = 1/2
    k(x,x) = 1

Run alone, every test here is far under a second.
"""

import math
from statistics import NormalDist

import numpy as np
import pytest

import evaluation
from mogp_core import gp, icm, qnehvi
from mogp_core.hypervolume import hypervolume

# ---------------------------------------------------------------------- #
# Shared fixtures: fingerprints and their hand-computed Tanimoto Grams
# ---------------------------------------------------------------------- #
X_TRAIN = np.array([[1, 1, 0, 0],      # a
                    [1, 0, 1, 0],      # b
                    [1, 1, 1, 1]],     # c
                   dtype=float)

# Query rows: q1 = [0,1,1,0], q2 = [1,0,0,0], q3 = a.
#   q1.a = 1,|q1|=2,|a|=2 -> 1/3    q1.b = 1 -> 1/3    q1.c = 2,|c|=4 -> 2/(2+4-2) = 1/2
#   q2.a = 1,|q2|=1,|a|=2 -> 1/2    q2.b = 1 -> 1/2    q2.c = 1 -> 1/(1+4-1)       = 1/4
#   q3 = a:  k(a,a) = 1, k(a,b) = 1/3, k(a,c) = 1/2
X_QUERY = np.array([[0, 1, 1, 0],
                    [1, 0, 0, 0],
                    [1, 1, 0, 0]], dtype=float)

S_TRAIN = np.array([[1.0, 1 / 3, 1 / 2],
                    [1 / 3, 1.0, 1 / 2],
                    [1 / 2, 1 / 2, 1.0]])
K_CROSS = np.array([[1 / 3, 1 / 3, 1 / 2],      # q1 vs (a, b, c)
                    [1 / 2, 1 / 2, 1 / 4],      # q2
                    [1.0, 1 / 3, 1 / 2]])       # q3
# Query-vs-query: q1.q2 = 0 -> 0 ; q1.q3 = 1 -> 1/3 ; q2.q3 = 1 -> 1/(1+2-1) = 1/2.
K_QQ = np.array([[1.0, 0.0, 1 / 3],
                 [0.0, 1.0, 1 / 2],
                 [1 / 3, 1 / 2, 1.0]])


def _inv_softplus(y):
    """Inverse of softplus(x) = log(1 + exp(x)):  x = log(exp(y) - 1)."""
    return math.log(math.expm1(y))


NOISE_FLOOR = 1e-4     # GreaterThan(1e-4): value = 1e-4 + softplus(raw)


# ====================================================================== #
# 1. Hypervolume, reference at the origin
# ====================================================================== #
# Each case: (name, points, expected volume, derivation).  Points are in the
# normalized maximization frame; the dominated region of p is the box [0, p].
HV_CASES = [
    # one box: 0.5 * 0.5
    ("one point 2-D", [[0.5, 0.5]], 0.25),
    # |A| = 1*0.5 = 0.5, |B| = 0.5*1 = 0.5, A n B = [0,.5]x[0,.5] = 0.25
    # union = 0.5 + 0.5 - 0.25
    ("two points 2-D", [[1.0, 0.5], [0.5, 1.0]], 0.75),
    # 0.5^3
    ("one point 3-D", [[0.5, 0.5, 0.5]], 0.125),
    # (0.25, 0.25) lies inside the box of (0.5, 0.5): changes nothing
    ("dominated extra point", [[0.5, 0.5], [0.25, 0.25]], 0.25),
    ("dominated extra point 3-D", [[0.5, 0.5, 0.5], [0.4, 0.5, 0.25]], 0.125),
    # a repeated point is the same box, not two
    ("duplicate point", [[0.5, 0.5], [0.5, 0.5]], 0.25),
    # (0.7, 0.0) sits on the reference boundary: zero width, zero area
    ("point on the reference boundary", [[0.5, 0.5], [0.7, 0.0]], 0.25),
    ("point on the boundary 3-D", [[0.5, 0.5, 0.5], [0.9, 0.9, 0.0]], 0.125),
    # 2-D staircase: strips of x. [0,.25]: height 1 -> .25 ; [.25,.5]: height .5
    # -> .125 ; [.5,1]: height .25 -> .125. Total 0.5
    ("three-point staircase", [[1.0, 0.25], [0.5, 0.5], [0.25, 1.0]], 0.5),
    # 3-D inclusion-exclusion, p = (1, .5, .25), q = (.5, .25, 1):
    #   |A| = 1*.5*.25 = 0.125            |B| = .5*.25*1 = 0.125
    #   A n B = [0, min(1,.5)] x [0, min(.5,.25)] x [0, min(.25,1)]
    #         = .5 * .25 * .25 = 0.03125
    #   |A u B| = 0.125 + 0.125 - 0.03125 = 0.21875
    # (p and q are mutually non-dominated: p1 > q1 but p3 < q3.)
    ("two points 3-D inclusion-exclusion",
     [[1.0, 0.5, 0.25], [0.5, 0.25, 1.0]], 0.21875),
]


@pytest.mark.parametrize("name,points,expected", HV_CASES,
                         ids=[c[0] for c in HV_CASES])
def test_core_hypervolume_matches_hand_value(name, points, expected):
    pts = np.array(points, dtype=float)
    assert hypervolume(pts, np.zeros(pts.shape[1])) == pytest.approx(expected, abs=1e-15)


def test_core_hypervolume_reference_is_subtracted():
    # Reference (0.5, 0.5), point (1, 1): box [.5,1]^2 = 0.5 * 0.5 = 0.25.
    assert hypervolume([[1.0, 1.0]], [0.5, 0.5]) == pytest.approx(0.25, abs=1e-15)
    # A point below the reference on one axis contributes nothing.
    assert hypervolume([[0.25, 1.0]], [0.5, 0.5]) == 0.0


def test_core_hypervolume_point_below_reference_on_two_axes_is_zero():
    # True volume is 0 (the point is outside the positive orthant on two axes).
    # The naive product (-1)(-1)(2) = +2 is the spurious "volume" a missing
    # clamp would report.
    assert hypervolume([[-1.0, -1.0, 2.0]], [0.0, 0.0, 0.0]) == 0.0


def test_core_hypervolume_empty_and_all_on_reference_are_zero():
    assert hypervolume(np.empty((0, 2)), [0.0, 0.0]) == 0.0
    assert hypervolume([[0.0, 0.0], [0.0, 0.7]], [0.0, 0.0]) == 0.0


# The evaluation path takes ORIGINAL units and the fixed signs. With bounds
# [0, 1] on every objective the map is u = y (sign +1) or u = 1 - y (sign -1),
# so the raw matrix that normalizes to a wanted unit-cube point is written down
# directly. Signs, in TASK_NAMES order: PfDHFR -1, hDHFR +1, hERG -1, Caco2 +1,
# Half_Life +1. Objectives not under test are all-NaN (inactive), which is how
# the module drops a column that carries no data.
SIGNS = np.array([-1.0, 1.0, -1.0, 1.0, 1.0])
UNIT_BOUNDS = np.array([[0.0, 1.0]] * 5)


def _raw_matrix(points, cols):
    """(n, 5) raw matrix whose normalized image on ``cols`` is ``points``."""
    pts = np.asarray(points, dtype=float)
    Y = np.full((pts.shape[0], 5), np.nan)
    for k, j in enumerate(cols):
        Y[:, j] = (1.0 - pts[:, k]) if SIGNS[j] < 0 else pts[:, k]
    return Y


# Column sets per dimension; the second set in each pair contains the sign -1
# objectives, so a flipped sign in the normalization changes the answer.
COLS_BY_DIM = {2: [(3, 4), (0, 2)], 3: [(1, 3, 4), (0, 2, 4)]}


def test_evaluation_signs_are_the_documented_layout():
    assert list(evaluation.OBJECTIVE_SIGNS) == SIGNS.tolist()


@pytest.mark.parametrize("name,points,expected", HV_CASES,
                         ids=[c[0] for c in HV_CASES])
def test_evaluation_compute_hypervolume_matches_hand_value(name, points, expected):
    pts = np.array(points, dtype=float)
    for cols in COLS_BY_DIM[pts.shape[1]]:
        Y = _raw_matrix(pts, cols)
        got = evaluation.compute_hypervolume(Y, bounds=UNIT_BOUNDS)
        assert got == pytest.approx(expected, abs=1e-15), (name, cols)


def test_evaluation_five_objective_volumes():
    # all five coordinates 0.5 -> 0.5^5
    all_half = _raw_matrix([[0.5] * 5], (0, 1, 2, 3, 4))
    assert evaluation.compute_hypervolume(all_half, bounds=UNIT_BOUNDS) == \
        pytest.approx(1 / 32, abs=1e-15)
    # u = (.25, 1, .5, 1, 1) -> .25 * .5 = 0.125. A flipped sign on objective 0
    # would read u0 = .75 and give 0.375.
    mixed = _raw_matrix([[0.25, 1.0, 0.5, 1.0, 1.0]], (0, 1, 2, 3, 4))
    assert evaluation.compute_hypervolume(mixed, bounds=UNIT_BOUNDS) == \
        pytest.approx(0.125, abs=1e-15)
    # the all-ones corner fills the cube
    corner = _raw_matrix([[1.0] * 5], (0, 1, 2, 3, 4))
    assert evaluation.compute_hypervolume(corner, bounds=UNIT_BOUNDS) == \
        pytest.approx(1.0, abs=1e-15)


def test_evaluation_rows_with_a_nan_in_an_active_column_are_ignored():
    # A failed dock (NaN) is dropped: the volume is that of the remaining row.
    Y = _raw_matrix([[0.5, 0.5], [1.0, 1.0]], (3, 4))
    Y[1, 3] = np.nan
    assert evaluation.compute_hypervolume(Y, bounds=UNIT_BOUNDS) == \
        pytest.approx(0.25, abs=1e-15)


def test_evaluation_empty_and_no_active_objective_are_zero():
    assert evaluation.compute_hypervolume(np.empty((0, 5)), bounds=UNIT_BOUNDS) == 0.0
    assert evaluation.compute_hypervolume(np.full((2, 5), np.nan),
                                          bounds=UNIT_BOUNDS) == 0.0


# ====================================================================== #
# 2. GP posterior (Rasmussen & Williams eq. 2.25 / 2.26)
# ====================================================================== #
# Model (documented in mogp_core/gp.py): per task t, kernel s_t * Tanimoto,
# constant mean mu_t, noise tau_t = sigma2 + d_t with ONE shared sigma2.
#
#   mean_*  = mu + K_*^T (K + tau I)^-1 (y - mu)               (R&W 2.25)
#   var_f   = k(x*,x*) - K_*^T (K + tau I)^-1 K_*              (R&W 2.26)
#   var_y   = var_f + tau
# with K = s S and K_* = s * (Tanimoto cross-Gram).
S_OUT = np.array([1.5, 0.75])         # s_t
MU = np.array([0.25, -0.5])           # mu_t
D_TASK = np.array([0.10, 0.20])       # d_t
SIGMA2 = 0.05                         # shared
TAU = SIGMA2 + D_TASK                 # [0.15, 0.25]
Y_NORM = np.array([[0.5, -1.0],       # exactly representable in float32, which
                   [1.0, 0.25],       # GPState stores Y_norm in
                   [-0.5, 0.75]])


def _gp_theta(s, mu, d, sigma2):
    return gp.theta_from_groups(
        raw_outputscale=[_inv_softplus(x) for x in s],
        raw_constant=list(mu),
        raw_task_noises=[_inv_softplus(x - NOISE_FLOOR) for x in d],
        raw_noise=_inv_softplus(sigma2 - NOISE_FLOOR))


def _rw_posterior(S, Kq, s, mu, tau, Y):
    """R&W 2.25 / 2.26 for each task, by an explicit solve. Prior variance of the
    scaled kernel at a query point is s * k(x*, x*) = s (Tanimoto self-sim is 1)."""
    M, T = Kq.shape[0], Y.shape[1]
    mean = np.zeros((M, T))
    var_f = np.zeros((M, T))
    for t in range(T):
        A = s[t] * S + tau[t] * np.eye(S.shape[0])
        alpha = np.linalg.solve(A, Y[:, t] - mu[t])
        for m in range(M):
            k_star = s[t] * Kq[m]
            mean[m, t] = mu[t] + k_star @ alpha
            var_f[m, t] = s[t] * 1.0 - k_star @ np.linalg.solve(A, k_star)
    return mean, var_f, var_f + tau[None, :]


def test_gp_tanimoto_gram_matches_hand_fractions():
    np.testing.assert_allclose(gp.gram(X_TRAIN, X_TRAIN), S_TRAIN, rtol=0, atol=1e-15)
    np.testing.assert_allclose(gp.gram(X_QUERY, X_TRAIN), K_CROSS, rtol=0, atol=1e-15)
    np.testing.assert_allclose(gp.gram(X_QUERY, X_QUERY), K_QQ, rtol=0, atol=1e-15)


def test_gp_parameterization_recovers_the_natural_values():
    s, mu, d, sigma2, tau = gp.natural(_gp_theta(S_OUT, MU, D_TASK, SIGMA2), 2)
    np.testing.assert_allclose(s, S_OUT, rtol=0, atol=1e-13)
    np.testing.assert_allclose(mu, MU, rtol=0, atol=0)
    np.testing.assert_allclose(d, D_TASK, rtol=0, atol=1e-13)
    assert sigma2 == pytest.approx(SIGMA2, abs=1e-13)
    np.testing.assert_allclose(tau, TAU, rtol=0, atol=1e-13)   # tau_t = sigma2 + d_t


def test_gp_posterior_single_training_point_by_pure_arithmetic():
    """N = 1, T = 1, solved on paper.

    s = 2, tau = 2 (sigma2 = 1, d = 1), mu = 1, y = 3, training x = a = [1,1,0,0],
    query x* = [1,0,0,0] with k(x*, a) = 1/2.
        K = s*1 + tau = 4              K_* = s * 1/2 = 1
        mean  = mu + K_* (y - mu) / K = 1 + 1 * 2 / 4          = 1.5
        var_f = s*1 - K_*^2 / K       = 2 - 1/4                = 1.75
        var_y = var_f + tau           = 1.75 + 2               = 3.75
    """
    theta = _gp_theta([2.0], [1.0], [1.0], 1.0)
    mean, var_f, var_y = gp.posterior(theta, X_TRAIN[:1], np.array([[3.0]]),
                                      np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert mean[0, 0] == pytest.approx(1.5, abs=1e-10)
    assert var_f[0, 0] == pytest.approx(1.75, abs=1e-10)
    assert var_y[0, 0] == pytest.approx(3.75, abs=1e-10)


def test_gp_posterior_matches_explicit_rw_solve_three_points_two_tasks():
    theta = _gp_theta(S_OUT, MU, D_TASK, SIGMA2)
    mean, var_f, var_y = gp.posterior(theta, X_TRAIN, Y_NORM, X_QUERY)
    e_mean, e_var_f, e_var_y = _rw_posterior(S_TRAIN, K_CROSS, S_OUT, MU, TAU, Y_NORM)
    np.testing.assert_allclose(mean, e_mean, rtol=0, atol=1e-10)
    np.testing.assert_allclose(var_f, e_var_f, rtol=0, atol=1e-10)
    np.testing.assert_allclose(var_y, e_var_y, rtol=0, atol=1e-10)
    # The tasks are independent given the shared noise: swapping the task order
    # of every input swaps the output columns and nothing else.
    theta_sw = _gp_theta(S_OUT[::-1], MU[::-1], D_TASK[::-1], SIGMA2)
    mean_sw, var_f_sw, _ = gp.posterior(theta_sw, X_TRAIN, Y_NORM[:, ::-1], X_QUERY)
    np.testing.assert_allclose(mean_sw, e_mean[:, ::-1], rtol=0, atol=1e-10)
    np.testing.assert_allclose(var_f_sw, e_var_f[:, ::-1], rtol=0, atol=1e-10)


def test_gp_posterior_variance_interpolation_limit():
    """At a training input with tiny noise the latent variance collapses toward 0
    and the mean toward the observation (R&W: noise-free interpolation). q3 = a."""
    theta = _gp_theta([1.0, 1.0], [0.0, 0.0], [1e-3, 1e-3], 1e-3)    # tau = 2e-3
    mean, var_f, _ = gp.posterior(theta, X_TRAIN, Y_NORM, X_QUERY)
    # Explicit-solve value at q3 (row 2), same derivation as above.
    e_mean, e_var_f, _ = _rw_posterior(S_TRAIN, K_CROSS, np.array([1.0, 1.0]),
                                       np.zeros(2), np.array([2e-3, 2e-3]), Y_NORM)
    np.testing.assert_allclose(mean, e_mean, rtol=0, atol=1e-10)
    np.testing.assert_allclose(var_f, e_var_f, rtol=0, atol=1e-10)
    assert var_f[2].max() < 2e-3                       # <= tau
    assert abs(mean[2, 0] - Y_NORM[0, 0]) < 0.02       # ~ y_a


def test_gp_predict_destandardizes_mean_linearly_and_variance_by_the_square():
    """predict(): mean * y_std + y_mean ; variance * y_std**2 (not y_std)."""
    theta = _gp_theta(S_OUT, MU, D_TASK, SIGMA2)
    observed = np.array([True, True, False, False, False])
    state = gp.GPState(theta, X_TRAIN, Y_NORM, observed)
    y_mean = np.array([10.0, -20.0, np.nan, np.nan, np.nan])
    y_std = np.array([2.0, 0.5, np.nan, np.nan, np.nan])
    e_mean, e_var_f, e_var_y = _rw_posterior(S_TRAIN, K_CROSS, S_OUT, MU, TAU, Y_NORM)

    mean, var = gp.predict(state, y_mean, y_std, X_QUERY)            # marginal var_y
    np.testing.assert_allclose(mean[:, :2], e_mean * y_std[:2] + y_mean[:2],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(var[:, :2], e_var_y * y_std[:2] ** 2, rtol=0, atol=1e-10)
    assert np.isnan(mean[:, 2:]).all() and np.isnan(var[:, 2:]).all()

    _, var_latent = gp.predict(state, y_mean, y_std, X_QUERY, observation_noise=False)
    np.testing.assert_allclose(var_latent[:, :2], e_var_f * y_std[:2] ** 2,
                               rtol=0, atol=1e-10)


# ====================================================================== #
# 3. ICM task covariance and the Kronecker covariance
# ====================================================================== #
# B = W W^T + diag(v), v = softplus(raw_var):
#   W = [[1], [2]] (K=2, R=1) ->  W W^T = [[1, 2], [2, 4]]
#   v = [0.5, 0.25]            ->  B = [[1.5, 2.0], [2.0, 4.25]]
W_ICM = np.array([[1.0], [2.0]])
V_ICM = np.array([0.5, 0.25])
B_EXPECTED = np.array([[1.5, 2.0],
                       [2.0, 4.25]])
TAU_ICM = np.array([0.05, 0.15])
SIGMA2_ICM = 0.10
D_ICM = TAU_ICM + SIGMA2_ICM          # d_t = tau_t + sigma2 = [0.15, 0.25]
C_ICM = np.array([0.2, -0.3])


def _icm_theta():
    return icm.theta_from_groups(
        W_ICM,
        [_inv_softplus(v) for v in V_ICM],
        [_inv_softplus(t - NOISE_FLOOR) for t in TAU_ICM],
        _inv_softplus(SIGMA2_ICM - NOISE_FLOOR),
        C_ICM)


def test_icm_task_covariance_is_WWt_plus_diag_v_entrywise():
    B = icm.task_covariance(W_ICM, [_inv_softplus(v) for v in V_ICM])
    # B[0,0] = 1*1 + 0.5 ; B[0,1] = B[1,0] = 1*2 ; B[1,1] = 2*2 + 0.25
    assert B[0, 0] == pytest.approx(1.5, abs=1e-13)
    assert B[0, 1] == pytest.approx(2.0, abs=1e-13)
    assert B[1, 0] == pytest.approx(2.0, abs=1e-13)
    assert B[1, 1] == pytest.approx(4.25, abs=1e-13)
    np.testing.assert_allclose(B, B_EXPECTED, rtol=0, atol=1e-13)


def test_icm_natural_parameters_recover_B_noise_and_mean():
    B, tau, sigma2, d, c = icm.natural(_icm_theta(), 2)
    np.testing.assert_allclose(B, B_EXPECTED, rtol=0, atol=1e-13)
    np.testing.assert_allclose(tau, TAU_ICM, rtol=0, atol=1e-13)
    assert sigma2 == pytest.approx(SIGMA2_ICM, abs=1e-13)
    np.testing.assert_allclose(d, D_ICM, rtol=0, atol=1e-13)    # d_t = tau_t + sigma2
    np.testing.assert_allclose(c, C_ICM, rtol=0, atol=0)


def test_icm_task_correlation_of_the_hand_B():
    # rho = B_01 / sqrt(B_00 B_11) = 2 / sqrt(1.5 * 4.25) = 2 / sqrt(6.375)
    rho = icm.task_correlation(B_EXPECTED)
    assert rho[0, 1] == pytest.approx(2.0 / math.sqrt(6.375), abs=1e-14)
    np.testing.assert_allclose(np.diag(rho), [1.0, 1.0], rtol=0, atol=1e-15)


def test_icm_kronecker_covariance_is_data_major_entrywise():
    """Sigma[i*K + t, j*K + s] = Kx[i,j] * B[t,s]  (+ noise on the diagonal).

    Data-major: the flat index is molecule * K + task, i.e. C-order reshape of an
    (N, K) array. The task-major alternative, kron(B, Kx), is permutation-similar
    (same loss, same determinant) but puts different numbers in these cells.
    """
    N, K = 3, 2
    K_data = icm.assemble_dense(S_TRAIN, B_EXPECTED)
    # cells worked by hand
    assert K_data[0, 1] == pytest.approx(1.0 * 2.0, abs=1e-14)          # (a,t0),(a,t1)
    assert K_data[1, 2] == pytest.approx((1 / 3) * 2.0, abs=1e-14)      # (a,t1),(b,t0)
    assert K_data[3, 5] == pytest.approx((1 / 2) * 4.25, abs=1e-14)     # (b,t1),(c,t1)
    for i in range(N):
        for j in range(N):
            for t in range(K):
                for s in range(K):
                    assert K_data[i * K + t, j * K + s] == pytest.approx(
                        S_TRAIN[i, j] * B_EXPECTED[t, s], abs=1e-14)
    # and it is NOT the task-major ordering
    assert not np.allclose(K_data, np.kron(B_EXPECTED, S_TRAIN))
    assert K_data[0, 1] != pytest.approx(np.kron(B_EXPECTED, S_TRAIN)[0, 1])

    # With the noise terms as the model writes them: + I_N (x) diag(tau) + sigma2 I.
    Sigma = icm.dense_covariance(S_TRAIN, B_EXPECTED, TAU_ICM, SIGMA2_ICM)
    expected = np.zeros((N * K, N * K))
    for i in range(N):
        for j in range(N):
            for t in range(K):
                for s in range(K):
                    expected[i * K + t, j * K + s] = S_TRAIN[i, j] * B_EXPECTED[t, s]
    for a in range(N * K):
        expected[a, a] += TAU_ICM[a % K] + SIGMA2_ICM
    np.testing.assert_allclose(Sigma, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(Sigma, np.kron(S_TRAIN, B_EXPECTED)
                               + np.diag(np.tile(D_ICM, N)), rtol=0, atol=1e-14)


def _icm_expected(S, Kq, Kqq, B, d, c, Y):
    """Exact GP conditioning on the dense data-major covariance, by loops."""
    N, M, K = S.shape[0], Kq.shape[0], B.shape[0]
    Sigma = np.zeros((N * K, N * K))
    Cross = np.zeros((M * K, N * K))
    Prior = np.zeros((M * K, M * K))
    for i in range(N):
        for j in range(N):
            for t in range(K):
                for s in range(K):
                    Sigma[i * K + t, j * K + s] = S[i, j] * B[t, s]
    for a in range(N * K):
        Sigma[a, a] += d[a % K]
    for m in range(M):
        for j in range(N):
            for t in range(K):
                for s in range(K):
                    Cross[m * K + t, j * K + s] = Kq[m, j] * B[t, s]
        for m2 in range(M):
            for t in range(K):
                for s in range(K):
                    Prior[m * K + t, m2 * K + s] = Kqq[m, m2] * B[t, s]
    r = (Y - c[None, :]).reshape(-1)                      # index i*K + t
    mean = c[None, :] + (Cross @ np.linalg.solve(Sigma, r)).reshape(M, K)
    cov = Prior - Cross @ np.linalg.solve(Sigma, Cross.T)
    var_latent = np.diag(cov).reshape(M, K)
    return mean, var_latent, var_latent + d[None, :], cov


Y_ICM = np.array([[0.5, -1.0],
                  [1.0, 0.25],
                  [-0.5, 0.75]])


def test_icm_posterior_matches_dense_conditioning():
    mean, var_latent, var_obs = icm.posterior(_icm_theta(), X_TRAIN, Y_ICM, X_QUERY)
    e_mean, e_lat, e_obs, _ = _icm_expected(S_TRAIN, K_CROSS, K_QQ, B_EXPECTED,
                                            D_ICM, C_ICM, Y_ICM)
    np.testing.assert_allclose(mean, e_mean, rtol=0, atol=1e-10)
    np.testing.assert_allclose(var_latent, e_lat, rtol=0, atol=1e-10)
    np.testing.assert_allclose(var_obs, e_obs, rtol=0, atol=1e-10)
    # The two tasks are genuinely coupled through B_01 = 2: the task-0 mean is
    # not what an independent GP on column 0 alone would give.
    solo_mean = _icm_expected(S_TRAIN, K_CROSS, K_QQ,
                                    np.diag(np.diag(B_EXPECTED)), D_ICM, C_ICM, Y_ICM)[0]
    assert np.abs(mean[:, 0] - solo_mean[:, 0]).max() > 1e-3


def test_icm_posterior_joint_covariance_and_blocks_match_dense_conditioning():
    theta = _icm_theta()
    _, _, _, e_cov = _icm_expected(S_TRAIN, K_CROSS, K_QQ, B_EXPECTED,
                                   D_ICM, C_ICM, Y_ICM)
    cov = icm.posterior_covariance(theta, X_TRAIN, Y_ICM, X_QUERY)
    np.testing.assert_allclose(cov, e_cov, rtol=0, atol=1e-10)
    blocks = icm.posterior_block_covariance(theta, X_TRAIN, Y_ICM, X_QUERY)
    for m in range(X_QUERY.shape[0]):
        np.testing.assert_allclose(blocks[m], e_cov[2 * m:2 * m + 2, 2 * m:2 * m + 2],
                                   rtol=0, atol=1e-10)


# ====================================================================== #
# 4. EHVI against a closed form
# ====================================================================== #
# A regime where the expectation has a closed form: TWO objectives, one baseline
# point, independent Gaussian docking posterior, known-exact second objective.
#
#   objective 0: docking, sign -1, bounds [-11, -5]  ->  u = (-5 - y) / 6
#   objective 1: a known (ADMET-like) column, sign +1, bounds [0, 1] -> u = y
#   reference = (0, 0).
#
# Baseline: ONE point (b1, b2) = (0.4, 0.5), i.e. docking y = -5 - 6*0.4 = -7.4
# with variance 0 (floored to 1e-9 by the module; sd 3e-5 kcal, 5e-6 in u).
#
# A candidate (y, a) with y = docking in u-units ~ N(m, s^2), known a, has
#   HVI(y, a) = area(box(y, a)) - area(box(y, a) n box(b1, b2))
#             = y a - min(y, b1) min(a, b2)
# so, by independence of the (single) random coordinate,
#   E[HVI] = a m - min(a, b2) E[min(y, b1)]
#   E[min(y, b)] = m - E[(y - b)+]
#   E[(y - b)+]  = (m - b) Phi(d) + s phi(d),   d = (m - b) / s      (Gaussian EI)
# valid while the clip of u to [0, 1] has negligible mass (all cases below keep
# u >= 6 sd away from 0 and 1; clip mass ~ 1e-9 or less).
BOUNDS_2D = np.array([[-11.0, -5.0], [0.0, 1.0]])
SIGNS_2D = np.array([-1.0, 1.0])
B1, B2 = 0.4, 0.5


def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _closed_form_ehvi(m, s, a):
    d = (m - B1) / s
    ei = (m - B1) * _Phi(d) + s * _phi(d)
    return a * m - min(a, B2) * (m - ei)


# Candidates, in u-units: (m, s, a).  raw docking mean = -5 - 6 m, raw sd = 6 s.
#   c1: m = .5, s = .05, a = .8    d = 2.   E = 0.8*.5 - .5*(.5 - EI),
#       EI = 0.1 Phi(2) + 0.05 phi(2) = 0.1*0.97724987 + 0.05*0.05399097
#          = 0.09772499 + 0.00269955 = 0.10042454
#       E  = 0.4 - 0.5*(0.5 - 0.10042454) = 0.4 - 0.19978773 = 0.20021227
#   c2: m = .1, s = .05, a = .3    y < .4 and a < .5: box inside the baseline's
#       (needs a 6-sigma draw to escape) -> E = 0 exactly
#   c3: m = .7, s = .05, a = .4    y > .4 by 6 sigma and a < b2:
#       HVI = a (y - b1)  ->  E = 0.4 * (0.7 - 0.4) = 0.12
CANDIDATES = [(0.5, 0.05, 0.8), (0.1, 0.05, 0.3), (0.7, 0.05, 0.4)]
HAND_EXPECTED = np.array([0.20021227, 0.0, 0.12])


def _ehvi_inputs():
    mu_b = np.array([[-5.0 - 6.0 * B1]])
    var_b = np.array([[0.0]])
    admet_b = np.array([[B2]])
    mu_c = np.array([[-5.0 - 6.0 * m] for m, _, _ in CANDIDATES])
    var_c = np.array([[(6.0 * s) ** 2] for _, s, _ in CANDIDATES])
    admet_c = np.array([[a] for _, _, a in CANDIDATES])
    return mu_b, var_b, admet_b, mu_c, var_c, admet_c


def _run_ehvi(n_mc, **kw):
    return qnehvi.qnehvi(*_ehvi_inputs(), BOUNDS_2D, estimator="exact", n_mc=n_mc,
                         signs=SIGNS_2D, dock_idx=(0,), lib_idx=(1,),
                         num_objectives=2, **kw)


def test_ehvi_closed_form_values_are_what_the_comments_say():
    cf = np.array([_closed_form_ehvi(*c) for c in CANDIDATES])
    np.testing.assert_allclose(cf, HAND_EXPECTED, rtol=0, atol=1e-7)
    # c3 needs no Phi: 0.4 * 0.3. The closed form carries the 6-sigma tail (~1e-9 * s).
    assert cf[2] == pytest.approx(0.12, abs=1e-9)


def test_ehvi_exact_estimator_matches_closed_form_within_mc_tolerance():
    """iid Monte Carlo. HVI is a Lipschitz function g of one standard normal z:
    |dg/dz| <= max slope * s <= 0.8 * 0.05 = L (slope in u is a - min(a,b2)*[y<b1]
    <= 0.8). Gaussian Poincare: Var g(Z) <= E g'(Z)^2 <= L^2, so the standard
    error of an n-sample mean is <= L / sqrt(n). Tolerance: 5 standard errors."""
    n_mc = 2048
    L = 0.8 * 0.05
    tol = 5.0 * L / math.sqrt(n_mc)                    # 4.4e-3
    got = _run_ehvi(n_mc, run_seed=3)
    np.testing.assert_allclose(got, HAND_EXPECTED, rtol=0, atol=tol)
    assert got[1] == 0.0                               # dominated: exactly zero


def test_ehvi_exact_estimator_quantile_quadrature_pins_the_value_tightly():
    """Deterministic variant, no sampling noise: feed the n equal-probability
    quantile midpoints z_t = Phi^-1((t + 1/2) / n) as the candidate's standard
    normals (and 0 for the baseline, so the baseline sits exactly at b1). The
    estimator then computes (1/n) sum g(z_t), a midpoint rule on the cells of
    equal Gaussian mass. g is L-Lipschitz in z, so each cell's error is at most
    L times the cell width; the widths sum to the z-range covered
    (about 2 * 3.8 + the two tail cells' ~0.3 each), giving
        |error| <= L * 8.2 / n = 0.04 * 8.2 / 4000 = 8.2e-5 < 1e-4."""
    n = 4000
    z = np.array([NormalDist().inv_cdf((t + 0.5) / n) for t in range(n)])
    base = np.zeros((n, 2, 1))                         # (n_mc, B + 1, #docking cols)
    base[:, 1, 0] = z                                  # row B is the candidate slot
    got = _run_ehvi(n, base_samples=base)
    np.testing.assert_allclose(got, HAND_EXPECTED, rtol=0, atol=1e-4)
    assert got[1] == 0.0
    assert got[2] == pytest.approx(0.12, abs=1e-4)


def test_ehvi_depends_on_the_sign_of_the_docking_objective():
    """If the minimized docking objective were normalized as a maximized one the
    baseline would sit at u = 0.6 instead of 0.4 and c1 would drop from 0.2002 to
    0.5*(...)+0.15 = 0.1502: far outside the quantile tolerance."""
    n = 4000
    z = np.array([NormalDist().inv_cdf((t + 0.5) / n) for t in range(n)])
    base = np.zeros((n, 2, 1))
    base[:, 1, 0] = z
    wrong = qnehvi.qnehvi(*_ehvi_inputs(), BOUNDS_2D, estimator="exact", n_mc=n,
                          signs=np.array([+1.0, +1.0]), dock_idx=(0,), lib_idx=(1,),
                          num_objectives=2, base_samples=base)
    assert abs(wrong[0] - HAND_EXPECTED[0]) > 0.04


# ====================================================================== #
# 5. The fixed reference point and the non-viable dock
# ====================================================================== #
DOCK_BOUNDS = np.array([[-11.0, -5.0],      # PfDHFR  (sign -1)
                        [-11.0, -5.0],      # hDHFR   (sign +1)
                        [0.0, 1.0],
                        [0.0, 1.0],
                        [0.0, 1.0]])


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_fixed_reference_point_is_the_all_zeros_corner(n):
    ref = evaluation.fixed_reference_point(n)
    assert ref.shape == (n,)
    assert (ref == 0.0).all() and not np.signbit(ref).any()


def test_module_fixed_reference_point_constant_is_all_zeros():
    assert evaluation.FIXED_REFERENCE_POINT.shape == (evaluation.N_OBJECTIVES,)
    assert (evaluation.FIXED_REFERENCE_POINT == 0.0).all()


def test_worst_value_of_each_docking_objective_is_the_opposite_bound():
    # PfDHFR is minimized (sign -1): worst = the TOP of the cube, -5.0.
    # hDHFR is maximized (sign +1): worst = the BOTTOM of the cube, -11.0.
    assert evaluation.worst_objective_value(0, bounds=DOCK_BOUNDS) == -5.0
    assert evaluation.worst_objective_value(1, bounds=DOCK_BOUNDS) == -11.0


def test_dock_viability_threshold_is_strictly_below_zero_and_finite():
    raw = np.array([-8.0, -0.001, 0.0, 0.001, 4.213, np.nan, np.inf, -np.inf])
    assert evaluation.dock_is_viable(raw).tolist() == \
        [True, True, False, False, False, False, False, False]


def test_nonviable_dock_normalizes_to_exactly_zero():
    """Expected values from the bounds [-11, -5], span 6:

        PfDHFR (sign -1): u = (hi - y) / 6 = (-5 - y) / 6
            viable   y = -8   -> 3/6 = 0.5
            no pose  y = +4.213 -> worst value -5 -> (-5 - -5)/6 = 0
        hDHFR  (sign +1): u = (y - lo) / 6 = (y + 11) / 6
            viable   y = -8   -> 3/6 = 0.5
            no pose  y = +10.73 -> worst value -11 -> (-11 + 11)/6 = 0
    Without the substitution the hDHFR clash would read (10.73 + 11)/6 = 3.62,
    clip to 1.0 and score as the BEST possible selectivity: the defect.
    """
    pf_raw = np.array([-8.0, 4.213, np.nan])
    h_raw = np.array([-8.0, 10.73, np.nan])
    pf_obj, pf_viable = evaluation.apply_dock_viability(pf_raw, 0, bounds=DOCK_BOUNDS)
    h_obj, h_viable = evaluation.apply_dock_viability(h_raw, 1, bounds=DOCK_BOUNDS)
    assert pf_viable.tolist() == [True, False, False]
    assert h_viable.tolist() == [True, False, False]
    assert pf_obj[0] == -8.0 and pf_obj[1] == -5.0 and np.isnan(pf_obj[2])
    assert h_obj[0] == -8.0 and h_obj[1] == -11.0 and np.isnan(h_obj[2])
    assert pf_raw[1] == 4.213 and h_raw[1] == 10.73             # input untouched

    Y = np.full((3, 5), 0.5)
    Y[:, 0], Y[:, 1] = pf_obj, h_obj
    U = evaluation.normalize(Y, bounds=DOCK_BOUNDS)
    assert U[0, 0] == 0.5 and U[0, 1] == 0.5
    for u in (U[1, 0], U[1, 1]):
        assert u == 0.0 and not np.signbit(u)                   # exactly +0.0
    assert np.isnan(U[2, 0]) and np.isnan(U[2, 1])

    # The unsubstituted hDHFR clash is the defect: it saturates at the top.
    clash = np.full((1, 5), 0.5)
    clash[0, 1] = 10.73
    assert evaluation.normalize(clash, bounds=DOCK_BOUNDS)[0, 1] == 1.0


def test_nonviable_dock_contributes_nothing_to_the_hypervolume():
    """Five objectives at u = 0.5 each (docking at -8 kcal/mol, the other three at
    raw 0.5 which is u = 0.5 under [0, 1] whatever the sign): HV = 0.5^5 = 1/32.
    A row whose PfDHFR found no pose is substituted to u = 0 on that axis, so its
    box has zero width there: it adds nothing, alone or beside the good row."""
    good = np.array([[-8.0, -8.0, 0.5, 0.5, 0.5]])
    assert evaluation.compute_hypervolume(good, bounds=DOCK_BOUNDS) == \
        pytest.approx(1 / 32, abs=1e-15)

    bad = good.copy()
    substituted, _ = evaluation.apply_dock_viability(np.array([3.0]), 0, bounds=DOCK_BOUNDS)
    bad[0, 0] = substituted[0]
    assert bad[0, 0] == -5.0
    assert evaluation.compute_hypervolume(bad, bounds=DOCK_BOUNDS) == 0.0
    assert evaluation.compute_hypervolume(np.vstack([good, bad]), bounds=DOCK_BOUNDS) == \
        pytest.approx(1 / 32, abs=1e-15)

    # The same hDHFR clash, unsubstituted, would have been the BEST value:
    # u1 = 1 instead of 0.5 -> HV = 0.5^4 * 1 = 1/16.
    clash = good.copy()
    clash[0, 1] = 3.0
    assert evaluation.compute_hypervolume(clash, bounds=DOCK_BOUNDS) == \
        pytest.approx(1 / 16, abs=1e-15)
