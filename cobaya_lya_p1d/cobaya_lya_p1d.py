from cobaya.likelihood import Likelihood
import numpy as np


def fit_polynomial(xmin, xmax, x, y, deg=2):
    """
    Fit a polynomial to the logarithm of a function over a specified range.

    The function performs a polynomial fit to log(y) as a function of log(x)
    using only data points with xmin < x < xmax. The resulting polynomial
    can be evaluated to approximate log(y) within the fitting range.

    Parameters
    ----------
    xmin, xmax : float
        Lower and upper bounds of the fitting range in x.
    x : array_like
        Independent variable.
    y : array_like
        Dependent variable, assumed to be positive.
    deg : int, optional
        Degree of the polynomial fit (default: 2).

    Returns
    -------
    poly : numpy.poly1d
        Polynomial representing log(y) as a function of log(x).
    """
    # Select data points within the fitting range
    x_fit = (x > xmin) & (x < xmax)

    # Fit a polynomial in log–log space; alternative parameterizations
    # could reduce correlations between polynomial coefficients
    poly = np.polyfit(np.log(x[x_fit]), np.log(y[x_fit]), deg=deg)

    # Return the fitted polynomial as a callable object
    return np.poly1d(poly)


class Cobaya_lya_p1d(Likelihood):
    """
    Gaussian likelihood for compressed Lyman-alpha forest 1D power-spectrum parameters.

    This likelihood constrains the amplitude and slope of the linear matter power
    spectrum at a pivot redshift and scale, using compressed parameters
    (Delta2_star, n_star) derived from the Lyman-alpha forest 1D power spectrum.
    """

    delta2star_mean: float
    delta2star_std: float
    nstar_mean: float
    nstar_std: float
    correlation: float
    zstar: float
    kstar_kms: float

    def initialize(self):
        """
        Initialize the data vector, covariance matrix, and theory options.

        This method constructs the Gaussian data vector and its covariance,
        and defines the options used to request the linear matter power spectrum
        from the Boltzmann solver via Cobaya.
        """

        # Data vector: mean values of the compressed parameters
        self.data = np.array([self.delta2star_mean, self.nstar_mean])

        # Off-diagonal covariance term from the correlation coefficient
        cross_term = self.correlation * self.delta2star_std * self.nstar_std

        # Covariance matrix and its inverse
        self.cov = np.array(
            [
                [self.delta2star_std**2, cross_term],
                [cross_term, self.nstar_std**2],
            ]
        )
        self.inv_cov = np.linalg.inv(self.cov)

        # Options controlling the P(k) grid returned by the theory provider
        self.options_get_Pk_grid = {
            "nonlinear": False,
            "vars_pairs": ("delta_nonu", "delta_nonu"),
            "kmax_Mpc": 3.0,
        }

    def get_requirements(self):
        """
        Declare theory quantities required by the likelihood.

        Returns
        -------
        dict
            Dictionary specifying the power-spectrum grid and Hubble parameter
            required from the theory code.
        """
        reqs = {
            "Pk_grid": {
                "z": self.zstar,
                "k_max": self.options_get_Pk_grid["kmax_Mpc"],
                "nonlinear": self.options_get_Pk_grid["nonlinear"],
                "vars_pairs": self.options_get_Pk_grid["vars_pairs"],
            },
            "Hubble": {"z": [self.zstar]},
        }

        return reqs

    def fit_linP_kms(
        self,
        deg=2,
        fit_min=0.5,
        fit_max=2.0,
    ):
        """
        Compute and locally fit the linear matter power spectrum at zstar.

        The method converts the linear matter power spectrum from comoving
        units to velocity units (km/s), fits a polynomial in log–log space
        around kstar_kms, and extracts compressed parameters describing the
        local amplitude, slope, and curvature.

        Parameters
        ----------
        deg : int, optional
            Degree of the polynomial fit (default: 2).
        fit_min, fit_max : float, optional
            Range of the fit in units of k / kstar_kms.

        Returns
        -------
        dict
            Dictionary containing Delta2_star, n_star, and alpha_star.
        """

        # k grid in velocity units around the pivot scale
        k_kms = np.logspace(
            np.log10(fit_min * self.kstar_kms),
            np.log10(fit_max * self.kstar_kms),
            100,
        )

        # Conversion factor from velocity units to comoving units
        H_z = self.provider.get_Hubble(self.zstar, units="km/s/Mpc")[0]
        dvdX = H_z / (1 + self.zstar)
        k_Mpc = k_kms * dvdX

        # Retrieve linear matter power spectrum P(k, z)
        # The minimum k is set internally by the theory code;
        # the maximum k is controlled via kmax_Mpc
        Pk_k, Pk_z, Pk_P = self.provider.get_Pk_grid(
            var_pair=self.options_get_Pk_grid["vars_pairs"],
            nonlinear=self.options_get_Pk_grid["nonlinear"],
        )

        # Ensure requested k range is covered by the theory prediction
        if k_Mpc[-1] > Pk_k[-1]:
            raise ValueError(
                "k_Mpc is not within Pk_k range; increase kmax_Mpc in get_requirements"
            )

        # Interpolate the power spectrum in log–log space
        P_Mpc = np.exp(np.interp(np.log(k_Mpc), np.log(Pk_k), np.log(Pk_P[0])))

        # Convert power spectrum to velocity units
        P_kms = P_Mpc * dvdX**3

        # Polynomial fit of log P(k) around the pivot scale
        P_fit = fit_polynomial(
            fit_min, fit_max, k_kms / self.kstar_kms, P_kms, deg=deg
        )

        # Extract compressed parameters at the pivot scale following
        # Appendix C of Pedersen+2023 (https://arxiv.org/abs/2209.09895)
        Delta2_star = (
            np.exp(P_fit[0]) * self.kstar_kms**3 / (2.0 * np.pi**2)
        )
        n_star = P_fit[1]
        alpha_star = 2.0 * P_fit[2]

        return {
            "Delta2_star": Delta2_star,
            "n_star": n_star,
            "alpha_star": alpha_star,
        }

    def logp(self, **params_values):
        """
        Compute the log-likelihood for a given set of cosmological parameters.

        Parameters
        ----------
        **params_values : dict
            Dictionary of sampled parameter values.

        Returns
        -------
        float
            Log-likelihood value assuming a multivariate Gaussian.
        """

        # Compute theoretical predictions for the compressed parameters
        th_params = self.fit_linP_kms()

        # Gaussian chi^2
        theory = np.array([th_params["Delta2_star"], th_params["n_star"]])
        diff = theory - self.data
        chi2 = diff.T.dot(self.inv_cov.dot(diff))

        return -chi2 / 2
