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
  (2003), a ratio-of-cosines profile out to a fixed limb angle. Its sampled
  RMS angular radius is measurably wider than the super-Gaussian it can
  replace, which is the point of offering both: the choice materially
  changes how much a receiver spot blurs. ``circumsolar_ratio`` (docs/
  ui-spec-v0.2.md §O) adds that same paper's circumsolar aureole term
  beyond the limb -- see the class docstring and :func:`_buie_full_profile`
  for the exact formulas and the ``circumsolar_ratio == 0`` bit-identity
  guarantee.
"""

from __future__ import annotations

import numpy as np

# Pinned super-Gaussian parameters.
SUPER_GAUSS_SIGMA_RAD = 0.0024
SUPER_GAUSS_ORDER = 2.0
MAX_THETA_RAD = 0.1  # truncation radius; cosmetic at this sigma

# Limb angle of the Buie disk, milliradians.
BUIE_LIMB_MRAD = 4.65

#: Angular extent of the circumsolar aureole's validity domain, milliradians
#: (2.5 degrees) -- docs/ui-spec-v0.2.md §O. Buie, Monger & Dey (2003),
#: "Sunshape distributions for terrestrial solar simulations", Solar Energy
#: 74 pp.113-122, fit kappa/gamma (below) against LBL/DLR brightness data
#: out to 2.5 degrees and chose the constants so their own circumsolar-ratio
#: definition (that paper's eq. 3) is only *approximately* satisfied when
#: integrated over that same domain -- not exactly, and not for any other
#: truncation. Numerically extending the truncation radius well past 43.6
#: mrad does not converge the realised ratio closer to the nominal CSR; it
#: overshoots further, because the aureole's power-law tail decays too
#: slowly (gamma+1 > -1 at low CSR) to be integrable at infinity in the
#: first place -- the published fit is only ever meant to be read out to
#: 2.5 degrees. So this is the literature's own domain, not an arbitrary
#: cutoff of convenience.
AUREOLE_LIMIT_MRAD = 43.6


def _buie_kappa_gamma(circumsolar_ratio: float) -> tuple[float, float]:
    """``(kappa, gamma)`` of the Buie, Monger & Dey (2003) aureole fit --
    that paper's eq. 2, verified against a peer-reviewed secondary source
    that reproduces the same two equations while citing the original
    (Kalapatapu, Armstrong, Chiesa & Wilbert, "Measurement of DNI Angular
    Distribution with a Sunshape Profiling Irradiometer", SolarPACES 2012,
    eq. 2, itself citing Buie's 2004 Sydney PhD thesis). Undefined at
    ``circumsolar_ratio == 0`` (both terms take ``log(0)``) -- callers must
    special-case zero rather than evaluate this there; see
    :class:`BuieSampler`'s own docstring for why that split also matters
    for bit-identity.
    """
    chi = circumsolar_ratio
    gamma = 2.2 * np.log(0.52 * chi) * chi**0.43 - 0.1
    kappa = 0.9 * np.log(13.5 * chi) * chi**-0.3
    return kappa, gamma


def _buie_full_profile(t_mrad: np.ndarray, circumsolar_ratio: float) -> np.ndarray:
    """Buie, Monger & Dey (2003) brightness profile, disk plus aureole,
    ``t_mrad`` (angular radius, milliradians) an array, ``circumsolar_ratio``
    a scalar > 0.

    ``phi(theta) = cos(0.326 theta)/cos(0.308 theta)`` for ``theta <=
    BUIE_LIMB_MRAD`` (the disk -- identical formula the ``circumsolar_ratio
    == 0`` path already used), ``phi(theta) = exp(kappa) * theta^gamma``
    beyond it (the aureole, :func:`_buie_kappa_gamma`). The two pieces are
    NOT continuous at the limb by construction -- that discontinuity is in
    the published model itself (independently fit pieces; see the
    reference's own Figure 2, a visible step in log-log relative intensity
    right at the limb), not a bug here.

    Only ever called for ``circumsolar_ratio > 0`` -- the zero case is
    handled by the original disk-only code path in :class:`BuieSampler`/
    ``sunshape_kernel`` verbatim, never through this function, so that path
    cannot be perturbed by the aureole branch existing.
    """
    disk_t = np.minimum(t_mrad, BUIE_LIMB_MRAD)
    disk = np.cos(0.326 * disk_t) / np.cos(0.308 * disk_t)
    kappa, gamma = _buie_kappa_gamma(circumsolar_ratio)
    aureole = np.exp(kappa) * np.power(np.maximum(t_mrad, 1e-12), gamma)
    return np.where(t_mrad <= BUIE_LIMB_MRAD, disk, aureole)


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
    """Inverse-CDF sampler for the Buie, Monger & Dey (2003) solar disk,
    optionally with its circumsolar aureole (docs/ui-spec-v0.2.md §O).

    ``I(theta) = cos(0.326 * theta_mrad) / cos(0.308 * theta_mrad)`` out to
    the limb (``BUIE_LIMB_MRAD``, 4.65 mrad) -- the limb-darkened disk, same
    at every ``circumsolar_ratio``. ``circumsolar_ratio == 0`` (the
    default) stops there, zero beyond, EXACTLY the disk-only sampler this
    class has always been -- §O's binding bit-identity requirement, so this
    branch is left as the original code, untouched, rather than routed
    through the more general aureole machinery below at a value that
    happens to zero it out (which the formulas cannot do anyway: both
    ``kappa``/``gamma`` take ``log(0)`` at ``circumsolar_ratio == 0``).
    ``circumsolar_ratio > 0`` extends the tabulated profile out to
    ``AUREOLE_LIMIT_MRAD`` (43.6 mrad, 2.5 degrees -- the published fit's
    own validity domain) with :func:`_buie_full_profile`'s aureole term
    added beyond the limb.
    """

    LIMB_MRAD = BUIE_LIMB_MRAD

    def __init__(self, circumsolar_ratio: float = 0.0, n_grid: int = 200_001):
        self.circumsolar_ratio = float(circumsolar_ratio)
        if self.circumsolar_ratio <= 0.0:
            theta = np.linspace(0.0, self.LIMB_MRAD * 1e-3, n_grid)
            t_mrad = theta * 1e3
            profile = np.cos(0.326 * t_mrad) / np.cos(0.308 * t_mrad)
            pdf = profile * theta  # small-angle Jacobian
            self._theta = theta
            self._cdf = _invert_cdf(theta, pdf)
            return
        theta = np.linspace(0.0, AUREOLE_LIMIT_MRAD * 1e-3, n_grid)
        t_mrad = theta * 1e3
        profile = _buie_full_profile(t_mrad, self.circumsolar_ratio)
        pdf = profile * theta  # small-angle Jacobian
        self._theta = theta
        self._cdf = _invert_cdf(theta, pdf)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.interp(rng.random(n), self._cdf, self._theta)


def make_sampler(name: str, circumsolar_ratio: float = 0.0):
    """Build a sampler by profile name. ``circumsolar_ratio`` (docs/ui-spec-
    v0.2.md §O) applies to ``"buie"`` only -- see :class:`BuieSampler`."""
    if name == "super_gauss":
        return SuperGaussSampler()
    if name == "buie":
        return BuieSampler(circumsolar_ratio=circumsolar_ratio)
    raise ValueError(f"unknown sunshape model {name!r}")
