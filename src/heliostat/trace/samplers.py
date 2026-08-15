"""Angular samplers for the sun's disk.

A Monte Carlo ray needs a direction drawn from the sun's angular brightness
distribution ``I(theta)`` (radiance per solid angle, radially symmetric
about the disk centre). The radial probability density is ``I(theta) *
theta`` — the small-angle solid-angle Jacobian — and both samplers here work
the same way: tabulate that density on a dense grid, invert its cumulative
distribution once at construction time, and sample by table lookup
thereafter. Grid resolution error is of order 1e-6 of the angular width,
far below Monte Carlo shot noise at any ray count worth tracing.

Two profiles are pinned:

* :class:`SuperGaussSampler` — a super-Gaussian ``I(theta) =
  exp(-(theta^2 / 2 sigma^2)^n)``. The ``sigma``/``n`` pair is measured
  against reference data, not derived; the naive small-angle reading from a
  nominal solar half-width is too narrow by a factor of ``sqrt(2)``.
* :class:`BuieSampler` — the limb-darkened solar disk of Buie, Monger & Dey
  (2003), a ratio-of-cosines profile out to a fixed limb angle with no
  circumsolar aureole term. Its sampled RMS angular radius is measurably
  wider than the super-Gaussian it can replace, which is the point of
  offering both: the choice materially changes how much a receiver spot
  blurs.
"""

from __future__ import annotations

import numpy as np

# Pinned super-Gaussian parameters.
SUPER_GAUSS_SIGMA_RAD = 0.0024
SUPER_GAUSS_ORDER = 2.0
MAX_THETA_RAD = 0.1  # truncation radius; cosmetic at this sigma

# Limb angle of the Buie disk, milliradians.
BUIE_LIMB_MRAD = 4.65


def _invert_cdf(theta: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    """Trapezoid-rule CDF of ``pdf`` over ``theta``, normalised to [0, 1].

    Shared by every sampler here so the quadrature rule and normalisation
    cannot drift between profiles.
    """
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]))])
    return cdf / cdf[-1]


class SuperGaussSampler:
    """Inverse-CDF sampler for the pinned super-Gaussian sunshape."""

    def __init__(
        self,
        sigma_rad: float = SUPER_GAUSS_SIGMA_RAD,
        order: float = SUPER_GAUSS_ORDER,
        max_theta_rad: float = MAX_THETA_RAD,
        n_grid: int = 200_001,
    ):
        upper = min(max_theta_rad, 8.0 * sigma_rad)  # tail beyond is < 1e-11
        theta = np.linspace(0.0, upper, n_grid)
        pdf = np.exp(-(((theta**2) / (2.0 * sigma_rad**2)) ** order)) * theta
        self._theta = theta
        self._cdf = _invert_cdf(theta, pdf)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.interp(rng.random(n), self._cdf, self._theta)


class BuieSampler:
    """Inverse-CDF sampler for the limb-darkened Buie solar disk.

    ``I(theta) = cos(0.326 * theta_mrad) / cos(0.308 * theta_mrad)`` out to
    the limb, zero beyond. No circumsolar aureole term.
    """

    LIMB_MRAD = BUIE_LIMB_MRAD

    def __init__(self, n_grid: int = 200_001):
        theta = np.linspace(0.0, self.LIMB_MRAD * 1e-3, n_grid)
        t_mrad = theta * 1e3
        profile = np.cos(0.326 * t_mrad) / np.cos(0.308 * t_mrad)
        pdf = profile * theta  # small-angle Jacobian
        self._theta = theta
        self._cdf = _invert_cdf(theta, pdf)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.interp(rng.random(n), self._cdf, self._theta)


def make_sampler(name: str):
    """Build a sampler by profile name."""
    if name == "super_gauss":
        return SuperGaussSampler()
    if name == "buie":
        return BuieSampler()
    raise ValueError(f"unknown sunshape model {name!r}")
