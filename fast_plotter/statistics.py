import numpy as np
from scipy.stats import beta


@np.vectorize
def mid_p_interval(total, passed, conf=0.682689492137, is_upper=True):
    alpha = 1. - conf
    alpha_min = alpha / 2
    vmin = alpha_min if is_upper else (1. - alpha_min)
    tol = 1e-9  # tolerance
    pmin = 0
    pmax = 1
    p = 0

    # treat special case for 0<passed<1
    # do a linear interpolation of the upper limit values
    if passed > 0 and passed < 1:
        p0 = mid_p_interval(total, 0.0, is_upper)
        p1 = mid_p_interval(total, 1.0, is_upper)
        p = (p1 - p0) * passed + p0
        return p

    while abs(pmax - pmin) > tol:
        p = (pmin + pmax) / 2
        # make it work for non integer using the binomial - beta relationship
        v = 0.5 * beta.pdf(p, passed + 1., total - passed + 1) / (total + 1)
        # compute the binomial cdf at passed -1
        if passed >= 1:
            v += 1 - beta.cdf(p, passed, total - passed + 1)

        if v > vmin:
            pmin = p
        else:
            pmax = p

    return p


def ratio_values(num, num_err_sq, denom, denom_err_sq,):
    num_sum = num.sum()
    num_sum_sq = num_err_sq.sum()
    denom_sum = denom.sum()
    denom_sum_sq = denom_err_sq.sum()

    numerator = np.zeros_like(num)
    denominator = np.zeros_like(denom)
    mask = num_err_sq != 0
    numerator[mask] = num[mask] * num[mask] / num_err_sq[mask]
    mask = denom_err_sq != 0
    denominator[mask] = denom[mask] * num[mask] / denom_err_sq[mask]

    wratio = np.zeros_like(numerator)
    mask = (num > 0) & (denom > 0)
    wratio[mask] = (num_err_sq[mask] / num[mask]) / (denom_err_sq[mask] / denom[mask])
    mask = (num == 0) & (denom > 0)
    wratio[mask] = (num_sum_sq / num_sum) / (denom_err_sq[mask] / denom[mask])
    mask = (num > 0) & (denom == 0)
    wratio[mask] = (num_err_sq[mask] / num[mask]) / (denom_sum_sq / denom_sum)

    ratio = np.zeros_like(numerator)
    mask = denominator != 0
    ratio[mask] = numerator[mask] / denominator[mask]
    lower = mid_p_interval(denominator, numerator, is_upper=False)
    upper = mid_p_interval(denominator, numerator, is_upper=True)

    ratio = ratio / (1 - ratio) * wratio
    lower = lower / (1 - lower) * wratio
    upper = upper / (1 - upper) * wratio

    return ratio, lower, upper


def ratio_vals2(num, num_err_sq, denom, denom_err_sq, conf=0.682689492137):
    """
    Direct port of ROOT code from  TGraphAsymmErrors::Divide with options "midp pois"
    But it doesn't seem to work properly....
    """
    psumw = num.sum()
    psumw2 = num_err_sq.sum()
    tsumw = denom.sum()
    tsumw2 = denom_err_sq.sum()

    # Set the graph to have a number of points equal to the number of histogram
    # bins
    nbins = len(num)

    # Ok, now set the points for each bin
    # (Note: the TH1 bin content is shifted to the right by one:
    #  bin=0 is underflow, bin=nbins+1 is overflow.)

    # this keeps track of the number of points added to the graph
    npoint = 0

    # loop over all bins and fill the graph
    out_eff = np.zeros(nbins)
    out_low = np.zeros(nbins)
    out_upper = np.zeros(nbins)
    for i_point in range(nbins):

        tw = denom[i_point]
        pw = num[i_point]
        tw2 = denom_err_sq[i_point]
        pw2 = num_err_sq[i_point]

        # compute ratio on the effective entries ( p and t)
        # special case is when (pw=0, pw2=0) in this case we cannot get the bin weight.
        # we use then the overall weight of the full histogram
        if pw == 0 and pw2 == 0:
            p = 0
        else:
            p = (pw * pw) / pw2

        if tw == 0 and tw2 == 0:
            t = 0
        else:
            t = (tw * tw) / tw2

        wratio = 1
        if pw > 0 and tw > 0:
            # this is the ratio of the two bin weights ( pw/p  / t/tw )
            wratio = (pw * t) / (p * tw)
        elif pw == 0 and tw > 0:
            # case p histogram has zero  compute the weights from all the histogram
            # weight of histogram - sumw2/sumw
            wratio = (psumw2 * t) / (psumw * tw)
        elif tw == 0 and pw > 0:
            # case t histogram has zero  compute the weights from all the histogram
            # weight of histogram - sumw2/sumw
            wratio = (pw * tsumw) / (p * tsumw2)
        elif p > 0:
            wratio = pw / p  # not sure if needed

        t += p

        # when not using weights (all cases) or in case of  Poisson ratio with weights
        eff = 0
        if t:
            eff = p / t

        low = mid_p_interval(t, p, conf, False)
        upper = mid_p_interval(t, p, conf, True)

        # take the intervals in eff as intervals in the Poisson ratio
        eff = eff / (1 - eff) * wratio
        low = low / (1. - low) * wratio
        upper = upper / (1. - upper) * wratio

        # Set the point center and its errors
        if not np.isinf(eff):
            out_eff[i_point] = eff
            out_low[i_point] = low
            out_upper[i_point] = upper
            npoint += 1  # we have added a point to the graph

    # if (npoint < nbins):
    #   Warning("Divide","Number of graph points is different than histogram
    #   bins - %d points have been skipped",nbins-npoint)
    return out_eff, out_low, out_upper


def ratio_root(num, num_err_sq, denom, denom_err_sq):
    import rootpy.plotting as rp
    # rootpy seems to switch this on, flooding the terminal with debugging output
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)

    # Set up the histograms
    top = rp.Hist(len(num), 0, 1)
    bottom = rp.Hist(len(num), 0, 1)
    for i, (d, n, d_err, n_err) in enumerate(zip(denom, num, denom_err_sq, num_err_sq)):
        bottom[i + 1] = (d, d_err)
        top[i + 1] = (n, n_err)

    # Do the actual division
    div = rp.Graph.divide(top, bottom, "e0 midp pois")

    # Convert this back to the array of points for the ratio plots
    ratios = np.zeros_like(num)
    low = np.zeros_like(num)
    upper = np.zeros_like(num)
    filled_indices = [top.FindBin(point.x.value) - 1 for point in div]
    ratios[filled_indices] = [point.y.value for point in div]
    low[filled_indices] = [point.y.error_low for point in div]
    upper[filled_indices] = [point.y.error_hi for point in div]
    return ratios, low, upper


def try_root_ratio_plot(*args, **kwargs):
    try:
        result = ratio_root(*args, **kwargs)
        return result
    except ImportError as e:
        if "rootpy" not in str(e):
            raise
        print("\n\tWarning: Using the broken errorbar method for ratio plots.\n\tInstall ROOT and rootpy to resolve\n")
        return ratio_vals2(*args, **kwargs)
