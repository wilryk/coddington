# References and provenance

Everything this package borrows, and from whom. Items marked **[confirm]**
still need the maintainer to verify.

---

## How to cite

If this software contributes to published research, please cite the companion
paper (reference forthcoming) and this repository. See `CITATION.cff`.

---

## Physical models

### Sunshape — the sun's angular brightness distribution

Two source models ship, both in `heliostat.trace.samplers`:

- **Super-Gaussian**, `I(theta) = exp(-(theta^2 / 2 sigma^2)^n)` with
  sigma = 2.4 mrad, n = 2. A smooth analytic profile used as the package
  default; not drawn from a published sunshape measurement.
- **Buie disk**, a limb-darkened solar disk out to a 4.65 mrad limb.

  > Buie, D., Monger, A. G., & Dey, C. J. (2003). Sunshape distributions for
  > terrestrial solar simulations. *Solar Energy*, 74(2), 113-122.

  **What was changed:** this implementation uses Buie's limb-darkened *disk*
  term only. The circumsolar aureole — the part of the published model
  governed by the circumsolar ratio — is deliberately omitted, so the profile
  ends at the limb. That is a simplification chosen for the companion paper's
  runs, and results produced with it should not be described as "the Buie
  sunshape" without saying so.

### Solar position

`heliostat.solar` implements the **NOAA Global Monitoring Laboratory solar
calculator** method — the same formulas as NOAA's published spreadsheet and
online calculator:

> NOAA Global Monitoring Laboratory, *Solar Calculator*.
> <https://gml.noaa.gov/grad/solcalc/azel.html>
> (spreadsheet method: <https://gml.noaa.gov/grad/solcalc/calcdetails.html>)

NOAA's calculator is in turn based on:

> Meeus, J. *Astronomical Algorithms*. Willmann-Bell.

Ported from a MATLAB implementation of that spreadsheet
(`SolarPositionCalculatorV3`), formulas unchanged, with three fixes: a
refraction branch that only worked for array input, unclipped `arccos` at
the horizon, and a fraction-of-day time convention that caused unit errors.
Excel-epoch Julian day arithmetic is kept so results track the spreadsheet.

### Optical figure (mirror shape) convention

Facet figures use the astigmatic-plus-defocus Zernike form with the radius
normalised to 1 mm, following the ophthalmic wavefront-reporting convention:

> ANSI Z80.28, *Ophthalmics — Methods for Reporting Optical Aberrations of
> Eyes*. **[confirm]** the edition in force.

---

## Numerical methods

### The fast ("cone optics") backend

`heliostat.trace.cone` computes flux without random sampling: it samples the
mirror surface on a deterministic grid, measures the local Jacobian
d(receiver uv)/d(source angle) by finite differences through the real optical
chain, and lays the sun's angular density down through that Jacobian
analytically.

**This sits inside an established tradition and does not originate it.**
Computing heliostat flux by convolving a sunshape with an optical mapping,
rather than tracing random rays, goes back to the first generation of
central-receiver codes. Lineage to narrow and confirm: **[confirm]**

- **HELIOS** — Biggs, F., & Vittitoe, C. N. (Sandia National Laboratories):
  the convolution model for reflecting solar concentrators.
- **Hermite-expansion flux methods**, as used in DELSOL — Walzel, M. D.,
  Lipps, F. W., & Vant-Hull, L. L. (1977), *Solar Energy*, 19; and Kistler,
  B. L. (1986), DELSOL (Sandia). The phrase "cone optics" in central-receiver
  work traces to the University of Houston group (Vant-Hull, Lipps).
- **HFLCAL** — Schwarzbözl, P., Schmitz, M., & Pitz-Paal, R.: a Gaussian
  analytic flux model.
- **Analytical flux models** — Collado, F. J.; Bendt, P., & Rabl, A.

**What appears specific to this implementation**, so a reader can see the
boundary:

- the Jacobian is *measured* through the real optical chain rather than
  assumed Gaussian or expanded analytically, so off-axis astigmatism, cone
  folding and hyperboloid magnification are inherited rather than modelled;
- an optional second-order mode measures the Hessian with a nine-ray stencil
  and inverts the quadratic map when depositing;
- edge effects (secondary rims, window borders, neighbour shadows and
  blocking) are measured per sample on a node grid in angle space, so a
  partially clipped kernel deposits with penumbra instead of being dropped.

### Polygon shading and blocking

Shadow and blocking overlaps are computed by projecting neighbour outlines and
clipping them against the mirror aperture with:

> Sutherland, I. E., & Hodgman, G. W. (1974). Reentrant polygon clipping.
> *Communications of the ACM*, 17(1), 32-42.

### Annual energy interpolation

Traced timesteps are resampled onto a normalised hour angle and interpolated
across solar declination with monotone piecewise-cubic interpolation
(`scipy.interpolate.PchipInterpolator`):

> Fritsch, F. N., & Carlson, R. E. (1980). Monotone piecewise cubic
> interpolation. *SIAM Journal on Numerical Analysis*, 17(2), 238-246.

---

## Field layouts

### Fermat-spiral ("sunflower") layout

`heliostat.field_layouts` generates a golden-angle Fermat spiral with angular
wedge and road filters. The code was ported from the maintainer's MATLAB
reference (`FermatSpiral.m`) and reproduces it bit-for-bit — but the *idea* of
laying a heliostat field out on a phyllotaxis spiral is published work:

> Noone, C. J., Torrilhon, M., & Mitsos, A. (2012). Heliostat field
> optimization: A new computationally efficient model and biomimetic layout.
> *Solar Energy*, 86(2), 792-803.

The underlying sunflower-spiral construction is:

> Vogel, H. (1979). A better way to construct the sunflower head.
> *Mathematical Biosciences*, 44(3-4), 179-189.

---

## Data sources

Both providers below ask to be acknowledged by works that use their data.
Anyone publishing results produced with them should carry that through.

### NASA POWER

Hourly DNI climatology — used for the companion paper's site results and
shipped in `examples/paper/data/` — comes from:

> NASA Prediction Of Worldwide Energy Resources (POWER) Project, NASA Langley
> Research Center. <https://power.larc.nasa.gov/>

Retrieved through the POWER hourly point API, parameter
`ALLSKY_SFC_SW_DNI`. POWER's data policy requests acknowledgement in
publications. **[confirm]** the exact wording POWER currently asks for.

### PVGIS

Typical-meteorological-year DNI is fetched from:

> Photovoltaic Geographical Information System (PVGIS), European Commission
> Joint Research Centre. <https://re.jrc.ec.europa.eu/pvg_tools/>

Retrieved through the PVGIS TMY API, column `Gb(n)`. **[confirm]** the JRC's
current citation requirement.

---

## Software this is built on

> Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585,
> 357-362.
>
> Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for
> scientific computing in Python. *Nature Methods*, 17, 261-272.

Also pandas, Matplotlib, FastAPI and Uvicorn. The 3-D scene view in the web
app is hand-written and uses no third-party graphics library.

---

## Provenance of this codebase

A port and generalisation of a private research codebase built for the
companion paper. Two things follow:

1. **The Monte Carlo tracer reproduces that research engine bit-for-bit** — 45
   golden fixtures (five heliostats x three sun positions x three optical
   layouts), matching on loss-chain counters, quantised receiver rays and
   recomputed spot metrics. See `tests/test_mc_parity.py`.
2. **That research engine was itself validated against a commercial optical
   CAD ray tracer** to 0.15% annual agreement. That validation is inherited
   here through (1). It was not re-performed in this repository, and no
   commercial tracer is needed to run or check anything in it.

`examples/paper/` reproduces the companion paper's published numbers using
this package alone: 54 of 54 instantaneous metrics within 0.644%, and annual
energy within 0.023% for the configuration validated at full scale.

---

## Still to confirm

- [ ] The cone-optics lineage above — narrow it to the references that
      genuinely precede this method, and say plainly which elements are
      inherited.
- [ ] ANSI Z80.28 edition in force.
- [ ] NASA POWER and PVGIS acknowledgement wording as currently requested.
- [ ] Companion paper citation, once it has a DOI.
