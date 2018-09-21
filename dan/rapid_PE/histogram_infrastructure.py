from __future__ import print_function

import numpy
import numpy as np
import cupy
import scipy.stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seed = 0
random = numpy.random.RandomState(seed)

histogram_color = "C0"
histogram_linestyle = "solid"
analytic_color="C3"
analytic_linestyle = "dashed"


def histogram(samples, n_bins, xpy=numpy):
    n_samples = samples.size

    # Compute the histogram counts.
    indices = xpy.trunc(samples * n_bins).astype(np.int32)
    histogram_counts = xpy.bincount(
        indices, minlength=n_bins,
        weights=xpy.broadcast_to(
            xpy.asarray([float(n_bins)/n_samples]),
            (n_samples,),
        ),
    )
    return histogram_counts


class MCSampler(object):
    def __init__(self, xpy=numpy):
        self.xpy = xpy
        self.setup_hist()

    def setup_hist(self):
        """
        Initializes dictionaries for all of the info that needs to be stored for
        the histograms, across every parameter.
        """
        self.x_min = {}
        self.x_max = {}
        self.x_max_minus_min = {}
        self.dx = {}
        self.n_bins = {}

        self.histogram_edges = {}
        self.histogram_values = {}
        self.histogram_cdf = {}


    def setup_hist_single_param(self, x_min, x_max, n_bins, param):
        # Compute the range of allowed values.
        x_max_minus_min = x_max - x_min
        # Compute the points at which the histogram will be evaluated, and store
        # the spacing used.
        histogram_edges, dx = self.xpy.linspace(
            0.0, 1.0, n_bins+1,
            retstep=True,
        )

        # Initialize output array for CDF.
        histogram_cdf = self.xpy.empty(n_bins+1, dtype=numpy.float64)

        # Store basic setup parameters
        self.x_min[param] = x_min
        self.x_max[param] = x_max
        self.x_max_minus_min[param] = x_max_minus_min
        self.dx[param] = dx
        self.n_bins[param] = n_bins

        self.histogram_edges[param] = histogram_edges
        self.histogram_cdf[param] = histogram_cdf


    def compute_hist(self, x_samples, param):
        # Rescale the samples to [0, 1]
        y_samples = (
            (x_samples - self.x_min[param]) / self.x_max_minus_min[param]
        )
        # Evaluate the histogram at each of the bins.
        histogram_values = histogram(
            y_samples, self.n_bins[param],
            xpy=self.xpy,
        )
        # Evaluate the CDF by taking a cumulative sum of the histogram.
        self.xpy.cumsum(histogram_values, out=self.histogram_cdf[param][1:])
        self.histogram_cdf[param] *= self.dx[param]
        self.histogram_cdf[param][0] = 0.0

        # Renormalize histogram.
        histogram_values /= self.x_max_minus_min[param]

        # Store histogram values.
        self.histogram_values[param] = histogram_values


    def cdf_inverse_from_hist(self, P, param):
        # Compute the value of the inverse CDF, but scaled to [0, 1].
        y = np.interp(
            P, self.histogram_cdf[param],
            self.histogram_edges[param],
        )
        # Return the value in the original scaling.
        return y*self.x_max_minus_min[param] + self.x_min[param]

    def pdf_from_hist(self, x, param):
        # Rescale `x` to [0, 1].
        y = (x - self.x_min[param]) / self.x_max_minus_min[param]
        # Compute the indices of the histogram bins that `x` falls into.
        indices = self.xpy.trunc(y / self.dx[param], out=y).astype(np.int32)
        # Return the value of the histogram.
        return self.histogram_values[param][indices]



def test_beta_dist(
        filename, n_samples, n_bins,
        alpha=3.5, beta=4.0,
        x_min=5.0, x_max=15.5,
        n_plotting_points=1000,
        xpy=numpy,
    ):
    # Dummy parameter name.
    param = "x"
    # Label for plotting
    label = "$x$"

    # Rescale beta distribution.
    loc = x_min
    scale = x_max - x_min

    # Setup the distribution function in scipy.
    dist = scipy.stats.beta(alpha, beta, loc=loc, scale=scale)

    # Generate random samples used for training.
    samples = xpy.asarray(dist.rvs(size=n_samples, random_state=random))

    # Set up the MCSampler object.
    mc_sampler = MCSampler(xpy=xpy)
    mc_sampler.setup_hist_single_param(
        xpy.asarray(x_min), xpy.asarray(x_max),
        n_bins,
        param,
    )
    mc_sampler.compute_hist(samples, param)


    fig, (ax_pdf, ax_icdf) = plt.subplots(2, sharex=True)

    x_plotting = numpy.linspace(x_min, x_max, n_plotting_points+1)[:-1]
    P_plotting = numpy.linspace(0.0, 1.0, n_plotting_points+1)[:-1]

    x_plotting_xpy = xpy.asarray(x_plotting)
    P_plotting_xpy = xpy.asarray(P_plotting)

    pdf_analytic = dist.pdf(x_plotting)
    pdf_histogram = mc_sampler.pdf_from_hist(x_plotting_xpy, param)

    icdf_analytic = dist.ppf(P_plotting)
    icdf_histogram = mc_sampler.cdf_inverse_from_hist(P_plotting_xpy, param)
    # icdf_analytic = numpy.zeros_like(P_plotting)
    # icdf_histogram = xpy.zeros_like(P_plotting_xpy)

    if xpy is cupy:
        pdf_histogram = cupy.asnumpy(pdf_histogram)
        icdf_histogram = cupy.asnumpy(icdf_histogram)

    ax_pdf.plot(
        x_plotting, pdf_histogram,
        color=histogram_color, linestyle=histogram_linestyle,
    )
    ax_pdf.plot(
        x_plotting, pdf_analytic,
        color=analytic_color, linestyle=analytic_linestyle,
    )

    ax_icdf.plot(
        icdf_histogram, P_plotting,
        color=histogram_color, linestyle=histogram_linestyle,
    )
    ax_icdf.plot(
        icdf_analytic, P_plotting,
        color=analytic_color, linestyle=analytic_linestyle,
    )

    fig.savefig(filename)


test_beta_dist(
    "beta_dist_histogram_test.png", 20000, 20,
    xpy=numpy,
)
