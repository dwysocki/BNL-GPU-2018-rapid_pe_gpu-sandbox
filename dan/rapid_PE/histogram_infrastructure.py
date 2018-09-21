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


class MCSampler(object):
    def __init__(self, xpy=numpy):
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


    def compute_hist(self, x_samples, x_min, x_max, n_bins, param):
        # Compute the range of allowed values.
        x_max_minus_min = x_max - x_min
        # Rescale the samples to [0, 1]
        y_samples = (x_samples - x_min) / x_max_minus_min
        # Compute the points at which the histogram will be evaluated, and store
        # the spacing used.
        histogram_edges, dx = np.linspace(0.0, 1.0, n_bins, retstep=True)
        # Evaluate the histogram at each of the bins.
        histogram_values, _ = np.histogram(
            y_samples, bins=histogram_edges, density=True,
        )
        # Evaluate the CDF by taking a cumulative sum of the histogram.
        histogram_cdf = np.empty(
            histogram_values.size+1, dtype=histogram_values.dtype,
        )
        np.cumsum(histogram_values, out=histogram_cdf[1:])
        histogram_cdf *= dx
        histogram_cdf[0] = 0.0

        # Renormalize histogram.
        histogram_values /= x_max_minus_min

        # Store basic setup parameters
        self.x_min[param] = x_min
        self.x_max[param] = x_max
        self.x_max_minus_min[param] = x_max_minus_min
        self.dx[param] = dx
        self.n_bins[param] = n_bins

        self.histogram_edges[param] = histogram_edges
        self.histogram_values[param] = histogram_values
        self.histogram_cdf[param] = histogram_cdf


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
        indices = np.trunc(y / self.dx[param], out=y).astype(np.int32)
        print(indices)
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
    samples = dist.rvs(size=n_samples, random_state=random)

    # Set up the MCSampler object.
    mc_sampler = MCSampler()
    mc_sampler.compute_hist(samples, x_min, x_max, n_bins, param)


    fig, (ax_pdf, ax_icdf) = plt.subplots(2, sharex=True)

    x_plotting = numpy.linspace(x_min, x_max, n_plotting_points+1)[:-1]
    P_plotting = numpy.linspace(0.0, 1.0, n_plotting_points+1)[:-1]

    pdf_analytic = dist.pdf(x_plotting)
    pdf_histogram = mc_sampler.pdf_from_hist(x_plotting, param)

    icdf_analytic = dist.ppf(P_plotting)
    icdf_histogram = mc_sampler.cdf_inverse_from_hist(P_plotting, param)

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
)
