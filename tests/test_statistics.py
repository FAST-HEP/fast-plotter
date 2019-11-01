import sys
import pytest
import fast_plotter.statistics as stats
# import numpy as np


@pytest.mark.skipif(sys.version_info < (3, 5), reason="Test stalls on python2 currently")
def test_mid_p_interval():
    assert stats.mid_p_interval(1., 1., is_upper=True) == pytest.approx(1)
    assert stats.mid_p_interval(2., 1., is_upper=True) == pytest.approx(0.84134475)
    assert stats.mid_p_interval(3., 1., is_upper=True) == pytest.approx(0.65424223)
    assert stats.mid_p_interval(4., 1., is_upper=True) == pytest.approx(0.53021317)

    assert stats.mid_p_interval(1., 1., is_upper=False) == pytest.approx(0.31731051)
    assert stats.mid_p_interval(2., 1., is_upper=False) == pytest.approx(0.15865525)
    assert stats.mid_p_interval(3., 1., is_upper=False) == pytest.approx(0.10616908)
    assert stats.mid_p_interval(4., 1., is_upper=False) == pytest.approx(0.079815806)

    answer = pytest.approx([1, 0.84134475, 0.65424223, 0.53021317])
    assert stats.mid_p_interval([1., 2, 3, 4], 1., is_upper=True) == answer


# def test_ratio_values():
#     denom = np.array([0, 1, 2, 3, 3])
#     num = np.array([0, 1, 1, 1, 1])
#     denom_err_sq = np.array([2, 2, 2, 2, 3])
#     num_err_sq = np.array([1, 1, 1, 1, 2]) / 2.
#     exp_ratio = np.array([0, 1.0, 0.5, 0.333333333333, 0.333333333333                      ])
#     exp_down  = np.array([0, 0.666666665839, 0.364845256467, 0.234807903596, 0.232016348009])
#     exp_up    = np.array([0, 11.2424952374, 4.15721634828, 2.22933131486, 11.0911931869    ])

#    denom_err_sq = np.array([1, 1, 1, 1, 2])
#    num_err_sq = np.array([1, 1, 1, 1, 2])
#    exp_ratio = np.array([0, 1.0, 0.5, 0.33333333333333337, 0.33333333333333337                              ])
#    exp_down  = np.array([0, 0.8114265836332124, 0.3633565038091062, 0.23388636326766077, 0.23480790359560538])
#    exp_up    = np.array([0, 4.302974402579285, 1.100720686840781, 0.6374025450676509, 2.229331314861169     ])

#    denom_err_sq = np.array([1, 1, 1, 1, 2])
#    num_err_sq = np.array([1, 1, 1, 1, 1])
#    exp_ratio = np.array([0, 1.0, 0.5, 0.33333333333333337, 0.33333333333333337                              ])
#    exp_down  = np.array([0, 0.8114265836332124, 0.3633565038091062, 0.23388636326766077, 0.25178266322490783])
#    exp_up    = np.array([0, 4.302974402579285, 1.100720686840781, 0.6374025450676509, 0.8822589396874957    ])

#    num_err_sq = np.array([1, 1, 1, 1, 1])
#    denom_err_sq = np.array([1, 1, 1, 1, 2])
#    exp_ratio = np.array([0, 1.0, 0.5, 0.333333333333, 0.333333333333                      ])
#    exp_down  = np.array([0, 0.811426583633, 0.363356503809, 0.233886363268, 0.228393011313])
#    exp_up    = np.array([0, 4.30297440258, 1.10072068684, 0.637402545068, 1.83580355825   ])

#    ratio, down, up = stats.ratio_vals2(num, num_err_sq * num_err_sq, denom, denom_err_sq * denom_err_sq)
#    print("down", down)
#    print("up", up)
#    down = ratio - down
#    up = up - ratio
#    assert len(ratio) == 5

#    print("down", down)
#    print("up", up)
#    print("ratio:", ratio)
#    print("exp/obs:", exp_up/up, exp_down / down)
#    print("exp-obs:", exp_up-up, exp_down - down)
#    assert ratio == pytest.approx(exp_ratio)
#    assert up == pytest.approx(exp_up)
#    assert down == pytest.approx(exp_down)
