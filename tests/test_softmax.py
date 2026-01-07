import numpy as np

from stabilitycheck.softmax import neff as neff_from_weights, solve_lambda_for_neff


def test_neff_monotone_in_lambda():
    rng = np.random.default_rng(0)
    g = rng.normal(size=200)
    # Two lambdas: larger lambda => weights flatter => larger Neff
    lam_small = 0.2
    lam_large = 2.0
    from stabilitycheck.softmax import softmax_weights

    mu_s = softmax_weights(g, lam_small)
    mu_l = softmax_weights(g, lam_large)
    assert neff_from_weights(mu_l) >= neff_from_weights(mu_s)


def test_extremes_flat_and_spike():
    # Flat g: Neff is always n (uniform weights)
    g_flat = np.zeros(50)
    sol_flat = solve_lambda_for_neff(g_flat, K=10)
    assert sol_flat["neff"] == 50.0
    assert sol_flat["converged"] is False

    # Spike g: Neff can get close to 1 for small K
    g_spike = np.zeros(100)
    g_spike[0] = 10.0
    sol_spike = solve_lambda_for_neff(g_spike, K=2)
    assert sol_spike["neff"] >= 1.0
    assert sol_spike["neff"] <= 5.0


def test_root_finding_accuracy_typical():
    rng = np.random.default_rng(1)
    g = rng.normal(size=300)
    K = 30
    sol = solve_lambda_for_neff(g, K=K, tol_rel=1e-4, max_iter=200)
    rel_err = abs(sol["neff"] - K) / K
    assert rel_err < 1e-3
