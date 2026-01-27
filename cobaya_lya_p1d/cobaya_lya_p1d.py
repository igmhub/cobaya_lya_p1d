from cobaya.likelihood import Likelihood
import numpy as np


def fit_polynomial(xmin, xmax, x, y, deg=2):
    """Fit a polynomial on the log of the function, within range"""
    x_fit = (x > xmin) & (x < xmax)
    # We could make these less correlated by better choice of parameters
    poly = np.polyfit(np.log(x[x_fit]), np.log(y[x_fit]), deg=deg)
    return np.poly1d(poly)


def fit_linP_kms(
    provider,
    zstar,
    kp_kms,
    deg=2,
    fit_min=0.5,
    fit_max=2.0,
):
    """Given input cosmology, compute linear power at zstar
    (in km/s) and fit polynomial around kp_kms."""

    k_kms = np.logspace(np.log10(0.5 * kp_kms), np.log10(2.0 * kp_kms), 100)

    ## Get conversion factor from velocity units to comoving
    H_z = provider.get_Hubble(zstar, units="km/s/Mpc")[0]
    dvdX = H_z / (1 + zstar)
    k_Mpc = k_kms * dvdX

    Pk_k, Pk_z, Pk_P = provider.get_Pk_grid(
        var_pair=("delta_nonu", "delta_nonu"), nonlinear=False
    )

    P_Mpc = np.exp(np.interp(np.log(k_Mpc), np.log(Pk_k), np.log(Pk_P[0])))
    P_kms = P_Mpc * dvdX**3

    # specify wavenumber range to fit
    kmin_kms = fit_min * kp_kms
    kmax_kms = fit_max * kp_kms
    # compute ratio
    P_fit = fit_polynomial(
        kmin_kms / kp_kms, kmax_kms / kp_kms, k_kms / kp_kms, P_kms, deg=deg
    )

    Delta2_star = np.exp(P_fit[0]) * kp_kms**3 / (2.0 * np.pi**2)
    n_star = P_fit[1]
    # note that the curvature is alpha/2
    alpha_star = 2.0 * P_fit[2]

    results = {
        "Delta2_star": Delta2_star,
        "n_star": n_star,
        "alpha_star": alpha_star,
    }

    return results


class Cobaya_lya_p1d(Likelihood):
    params = {
        "delta2star_mean": 0.379,
        "delta2star_std": 0.032,
        "nstar_mean": -2.309,
        "nstar_std": 0.019,
        "correlation": -0.1738,
        "zstar": 3.0,
        "kstar_kms": 0.009,
        "speed": 30000,
    }

    def initialize(self):
        """
        Set best-fitting value of the likelihood parameters
        """

        # Initizalize the data vector
        self.data = np.array(
            [self.params["delta2star_mean"], self.params["nstar_mean"]]
        )

        # Compute the correlation term
        cross_term = (
            self.params["correlation"]
            * self.params["delta2star_std"]
            * self.params["nstar_std"]
        )

        # Initialize the covariance matrix and compute its inverse
        self.cov = np.array(
            [
                [self.params["delta2star_std"] ** 2, cross_term],
                [cross_term, self.params["nstar_std"] ** 2],
            ]
        )
        self.inv_cov = np.linalg.inv(self.cov)

    def get_requirements(self, kmax_Mpc=2.0):
        """Quantities calculated by a theory code that are needed"""
        reqs = {
            "Pk_grid": {
                "z": self.params["zstar"],
                "k_max": kmax_Mpc,
                "nonlinear": False,
                "vars_pairs": [("delta_nonu", "delta_nonu")],
            },
            "Hubble": {"z": [self.params["zstar"]]},
        }

        return reqs

    def logp(self, **params_values):
        """Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """

        # compute value of compressed parameters
        th_params = fit_linP_kms(
            self.provider,
            self.params["zstar"],
            self.params["kstar_kms"],
        )

        # Compute the chi^2
        theory = np.array([th_params["Delta2_star"], th_params["n_star"]])
        diff = theory - self.data
        chi2 = diff.T.dot(self.inv_cov.dot(diff))

        return -chi2 / 2
