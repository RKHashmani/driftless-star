# Stage 3: Neoclassical Transport

## Overview

The workflow can omit unused flux producers. See [automatic producer skipping](../mvp-pipeline.md#automatic-producer-skipping) for selector, input, and output requirements.

Stage 3 computes neoclassical transport properties from the equilibrium. It has two solver families.

1. **`NEO` / `NEO_JAX`** -- Computes effective ripple (epsilon_eff), a screening/optimization diagnostic. **NOT a transport state variable** -- does not feed into profile evolution. Runs in parallel with the transport code.
2. **`SFINCS` / `DKX`** -- Solves the full drift-kinetic equation for neoclassical particle flux, heat flux, bootstrap current, and ambipolar E_r. Feeds `NEOPAX` (Stage 5).

**Position in pipeline:** `NEO_JAX` receives `boozmn_*.nc` from Stage 2 (Boozer). `DKX` solves on the `wout_*.nc` from Stage 1 (Equilibrium) and additionally reads the Stage 2 `boozmn_*.nc`, which fixes the minor radius its analytical profile grid is reconstructed on, so the radial scan follows Stage 2 in the DAG. Stage 3 runs in parallel with Stage 4 (Turbulence).

**Reference:** `stellarator_workflow.tex`, Sections 4.4-4.5; `stellarator_io_reference.tex`, Sections 3.4-3.5.

---

## Sub-Stage 3a: `NEO` / `NEO_JAX` (Effective Ripple)

### Codes

**NEO_JAX (Primary JAX):** <https://github.com/uwplasma/NEO_JAX>

**`NEO` (Legacy, part of `STELLOPT`):** <https://github.com/PrincetonUniversity/STELLOPT>

### Input Specification

Reference: `stellarator_io_reference.tex`, Section 3.4.

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `boozmn_*.nc` | NetCDF file | Boozer-coordinate equilibrium | Stage 2 |
| `neo_in.*` / `neo_param.*` | Control file | Surface list, angular resolution (theta_n, phi_n), Fourier cutoffs, MC controls, accuracy targets, current calculation switch (CALC_CUR) | User-specified |

#### Input Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

### Output Specification

Reference: `stellarator_io_reference.tex`, Section 3.4.

**Primary output:** `neo_out.*` and `neolog.*`

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `epstot` | 1D array (per surface) | epsilon_eff^{3/2} (total effective ripple) | **Screening objective only** |
| `epspar` | 1D array | Parallel epsilon | Diagnostic |
| `reff` | 1D array | Effective radius | Diagnostic |
| `iota` | 1D array | Rotational transform | Cross-check |
| `b_ref` | scalar | Reference magnetic field | Normalization |
| `r_ref` | scalar | Reference radius | Normalization |
| `ctrone` | 1D array | Contribution from one class | Diagnostic |
| `ctrtot` | 1D array | Total contribution | Diagnostic |
| `bareph` | 1D array | Parallel epsilon (bar) | Diagnostic |
| `barept` | 1D array | Perpendicular epsilon (bar) | Diagnostic |
| `yps` | 1D array | Normalized toroidal flux | Coordinate |

**`NEO_JAX` result objects:** `epsilon_effective`, `epsilon_effective_by_class`

Optional outputs (if CALC_CUR=1): `neo_cur.*`, `current.dat`, `conver.dat`, `diagnostic.dat`, `diagnostic_add.dat`, `diagnostic_bigint.dat`

**Role:** Screening/optimization diagnostic. Usually NO direct transport consumer. epsilon_eff is NOT what `Trinity3D` or `NEOPAX` advances in time.

#### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

### Governing Equations

Field-line integrals:

$$y_2 = \int d\phi\, B^{-2}, \quad y_3 = \int d\phi\, |\nabla\psi| B^{-2}, \quad y_4 = \int d\phi\, K_G B^{-3}$$

Trapped-particle integrals:

$$I_f = \int d\phi\, \sqrt{1 - \frac{B}{B_0 \eta}}\, B^{-2}$$

$$H_f = \int d\phi\, \sqrt{1 - \frac{B}{B_0 \eta}} \left(\frac{4}{B/B_0} - \frac{1}{\eta}\right) \frac{K_G}{\sqrt{\eta}} B^{-2}$$

Class-resolved effective ripple:

$$\epsilon_{\text{eff}}^{3/2}(m) = C_\epsilon \frac{y_2}{y_3^2} \text{BigInt}(m), \quad C_\epsilon = \frac{\pi R_0^2 \Delta\eta}{8\sqrt{2}}$$

Total `epstot` is the sum over classes.

**Reference:** `stellarator_workflow.tex`, Section 4.4.

---

## Sub-Stage 3b: `SFINCS` / `DKX` (Full Neoclassical)

### Codes

**DKX (Primary JAX):** <https://github.com/uwplasma/DKX>

**SFINCS (Legacy):** <https://github.com/landreman/sfincs>

The DKX environments pin version 2.4.0 to [Git revision `94506ae337b7825a161a47a614511ab5b538b37b`](https://github.com/uwplasma/DKX/tree/94506ae337b7825a161a47a614511ab5b538b37b). DKX requires Python 3.11 or later and SOLVAX 0.21 or later. The stage lockfile supplies shared scientific packages through conda, including SOLVAX 0.22.

### Input Specification

Reference: `stellarator_io_reference.tex`, Section 3.5.

| Field | Type | Required | Description | Source |
|-------|------|----------|-------------|--------|
| `input.namelist` | Fortran namelist text | **Yes** | Primary run configuration (all namelist groups). | User / workflow |
| `wout_*.nc` or `.bc` equilibrium | NetCDF / Boozer file | **Yes** (directly or via `equilibriumFile`) | Magnetic geometry input. In `DKX`, CLI `--wout-path` / `--equilibrium-file` overrides the namelist path and is written into embedded `input.namelist` in output. | Stage 1 or user |
| `boozmn_*.nc` | NetCDF | **Yes** for the radial scan | Not read by the solve. The scan takes the boundary `R00` from `rmnc_b` which, with the wout's `volume_p`, gives the minor radius NEOPAX grids on, and the analytical profile path reconstructs its faces on that grid. The bridge resolves it from `[geometry].boozer_file` in `common_input.toml` exactly as it resolves the wout from `[geometry].vmec_file`, and CLI `--boozer-path` overrides it. | Stage 2 |

**Required input fields** :

**Geometry & Equilibrium:**
- `geometryScheme`: geometry model/file mode. Common values: `1` (analytic tokamak-like model), `4` (analytic W7-X-like reduced model), `5` (VMEC from `wout_*.nc` or compatible VMEC equilibrium file), `11/12` (Boozer `.bc` file, with / without stellarator symmetry).
- `equilibriumFile` (or CLI override `--wout-path` / `--equilibrium-file`): path to the equilibrium file required by the selected geometry mode. For `geometryScheme=5`, this is a VMEC `wout` file. For `geometryScheme=11/12`, this is a Boozer `.bc` file.
- `inputRadialCoordinate`: selects which radial coordinate is used to identify the flux surface in the geometry/equilibrium input. This is used for file-backed geometry modes such as `geometryScheme=5` and `11/12`, and common choices are `0/1/2/3` for `psiHat` / `psiN` / `rHat` / `rN`.
- `inputRadialCoordinateForGradients`: selects which radial coordinate is used for profile and electrostatic-potential gradients. Common choices are `0/1/2/3` for gradients with respect to `psiHat` / `psiN` / `rHat` / `rN`.

**Species Definition:**
The definitions below are for default values of 'nu_n', 'Delta' and 'alpha'
- `Zs`: array species charges normalized to proton charge.
- `mHats`: array of mass ratios ($m_s / m_{\text{ref}}$, typically deuterium = 2).
- `nHats`: array of species density values in $10^20 m^{3}$.
- `THats`: array of species temperature values in $keV$.

**Density & Temperature Gradients:**
- `dnHatdpsiHat`, `dnHatdpsiN`, `dnHatdrHat`, or `dnHatdrN`: density gradient profile, with the active variable selected by `inputRadialCoordinateForGradients`.
- `dTHatdpsiHat`, `dTHatdpsiN`, `dTHatdrHat`, or `dTHatdrN`: temperature gradient profile, with the active variable selected by `inputRadialCoordinateForGradients`.

**Electric Field or Potential Gradient:**
- One of: `dPhiHatdpsiHat`, `dPhiHatdpsiN`, `dPhiHatdrHat`, `dPhiHatdrN`, or `Er` (radial electric field), with the active gradient coordinate selected by `inputRadialCoordinateForGradients`.

**Physics Controls:**
- `RHSMode`: right-hand-side / solve mode. `1` = standard drift-kinetic solve for fluxes and flows, `2` = solve multiple right-hand sides to assemble the transport matrix, `3` = monoenergetic transport-coefficient mode.
- `collisionOperator`: collision model. Common values are `0` (full linearized Fokker-Planck operator) and `1` (pitch-angle-scattering / Lorentz operator without the momentum-conserving field term).
- `constraintScheme`: constraint choice used to remove null-space / gauge freedom in the kinetic solve. Common values are `-1` (automatic, generally recommended), `0` (no constraints), `1` (enforce flux-surface-averaged density and pressure constraints), and `2` (enforce the $L=0$ constraint at each energy grid point).
- `Delta`: normalized gyroradius scale, roughly $\rho_*$ at the chosen reference parameters.
- `alpha`: electrostatic normalization factor, $e \bar{\Phi} / \bar{T}$, used to nondimensionalize `Er`, `Phi`, and `dPhiHatd*` quantities.
- `nu_n`: normalized collisionality used in the standard kinetic solve (`RHSMode=1`, and commonly also transport-matrix `RHSMode=2`). In `RHSMode=3`, the monoenergetic collisionality is instead specified through `nuPrime`.

**Resolution Controls:**
- `Ntheta`: poloidal grid points.
- `Nzeta`: toroidal grid points.
- `Nxi`: pitch-angle grid points.
- `Nx`: energy grid points.
- Solver tolerance: `solverTolerance` (commonly around $10^{-6}$ to $10^{-10}$ depending on the case and solver path). In this `DKX` checkout, this is the main named tolerance parameter exposed in the input namelist for the linear solve.

> [!NOTE]
> **Radial-scan bridge namelist overrides.** `dkx_radial_scan.py` edits a new template copy for each surface. It sets `RHSMode = 1` and `inputRadialCoordinate = 3` (`rN`). It sets `rN_wish` to the surface's rho. Before writing `dNHatdrNs` and `dTHatdrNs`, it removes every template density and temperature gradient assignment, including mixed-case and indexed forms. DKX would select stale `rHat` gradients from the HSX template before the intended `rN` values. Removing the template gradients prevents this conflict. The script also removes `inputRadialCoordinateForGradients`. DKX then infers the coordinates for species gradients and the electric field independently. The species fields select mode 3. `Er` selects mode 4.
>
> These are NEOPAX's face gradients in units per rho. The script applies no further scale factor because `rN` equals rho. Temperatures use keV. Temperature gradients use keV per rho. Densities use units of 1e20 m^-3. Density gradients use units of 1e20 m^-3 per rho. The output provenance section below describes how the script handles each profile source.
>
> Unless overridden on the command line, the scan uses the reduced quick-run resolution `Ntheta = 5, Nzeta = 11, Nxi = 12, NL = 3, Nx = 4, solverTolerance = 1e-6`. The worker lets DKX read `solverTolerance` from the generated namelist.

**Optional input fields** :

**Phi1 / Electrostatic Effects:**
- `includePhi1`: `.true./.false.` — supported in this `DKX` checkout; enables the Phi1 / quasineutrality / lambda block.
- `includePhi1InKineticEquation`: supported, but only as the current parity-first / frozen-linearization implementation of Phi1 coupling in the kinetic equation.
- `includePhi1InCollisionOperator`: supported for the Fokker-Planck (`collisionOperator = 0`) path, and requires `includePhi1 = .true.` plus `includePhi1InKineticEquation = .true.`.
- `readExternalPhi1`: recognized in the input surface, but not currently supported end-to-end in this `DKX` checkout.




**Distribution Function Export:**
- `export_full_f`: `.true./.false.` — write full (Maxwellian + perturbation) distribution to output.
- `export_delta_f`: `.true./.false.` — write perturbed part of distribution separately.
- `export_f_theta_option`, `export_f_zeta_option`, `export_f_xi_option`, `export_f_x_option`: select how the export grid is chosen for each coordinate.
- `export_f_theta`, `export_f_zeta`, `export_f_xi`, `export_f_x`: explicit coordinate values used when the corresponding export option selects custom sampling.



**All Input Fields**:

| Field | Type | Default | Required | Condition | Meaning | Units | 
|---|---|---|---|---|-----------------|----|
| RHSMode | integer | 1 | No (defaulted) | Always | Solve mode: `1` = standard drift-kinetic solve, `2` = transport-matrix assembly from multiple right-hand sides, `3` = monoenergetic transport-coefficient mode |
| outputFileName | string | ``sfincsOutput.h5'' | No (defaulted) | Always | Name which will be used for the HDF5 output file |
| saveMatlabOutput | Boolean | .false. | No (defaulted) | Always | If this switch is set to true, Matlab m-files are created which store the system matrix, right-hand side, and solution vector |
| MatlabOutputFilename | string | ``sfincsMatrices'' | Conditional | Only when saveMatlabOutput == .true.. | Start of the filenames which will be used for Matlab output. |
| saveMatricesAndVectorsInBinary | Boolean | .false. | No (defaulted) | Always | If this switch is set to true, the matrix, right-hand-side, and solution vector of the linear system will be saved in PETSc's binary format |
| binaryOutputFilename | string | ``sfincsBinary'' | Conditional | Only when saveMatricesAndVectorsInBinary == .true.. | Start of the filenames which will be used for binary output of the system matrices, right-hand-side vectors, and solution vectors |
| solveSystem | Boolean | .true. | No (defaulted) | Always | If this parameter is false, the system of equations will not actually be solved |
| ambipolarSolve | Boolean | .false. | Conditional | Primarily when `RHSMode == 1` | Legacy / upstream namelist-driven ambipolar root solve for `Er`; in this checkout, the more practical public workflow is usually the CLI `scan-er` + `ambipolar-solve` path |
| NEr\_ambipolarSolve | integer | 20 | Conditional | When ambipolarSolve == .true.. | Maximum number of solves to allow while finding the ambipolar Er. |
| Er\_search\_tolerance\_dx | real | 1.d-8 | Conditional | When ambipolarSolve == .true. and ambipolarSolveOption/=2. | Tolerance used for ambipolar solve |
| Er\_search\_tolerance\_f | real | 1.d-10 | Conditional | When ambipolarSolve == .true.. | Tolerance used for ambipolar solve |
| ambipolarSolveOption | integer | 1 | Conditional | When ambipolarSolve == .true. | Indicates which root solving algorithm to use for ambipolar solve |
| Er\_min | real | -100 | Conditional | When ambipolarSolve == .true. and ambipolarSolveOption /= 3. | Minimum value of Er used to bracket the ambipolar root. |
| Er\_max | real | 100 | Conditional | When ambipolarSolve == .true. and ambipolarSolveOption /= 3. | Maximum value of Er used to bracket the ambipolar root. |
| geometryScheme | integer | 1 | No (defaulted) | Always | How the magnetic geometry is specified. In this `DKX` checkout, the implemented modes are `1`, `2`, `4`, `5`, `11`, and `12` |
| inputRadialCoordinate | integer | 3 | Conditional | When the selected geometry needs a flux-surface choice | Which radial coordinate is used to select the target flux surface (`psiHat`, `psiN`, `rHat`, or `rN`) |
| inputRadialCoordinateForGradients | integer | 4 | Conditional | Whenever profile / electric-field gradients are specified | Which radial coordinate is used for input gradients. `0/1/2/3/4` correspond to `psiHat`, `psiN`, `rHat`, `rN`, and `Er`-based input |
| B0OverBBar | real | 1.0 | Conditional | Only when geometryScheme == 1 | Magnitude of (0,0) Boozer harmonic of B field |
| GHat | real | 3.7481 | Conditional | Only when geometryScheme == 1 | Poloidal current outside flux surface |
| IHat | real | 0.0 | Conditional | Only when geometryScheme == 1 | Toroidal current inside flux surface |
| iota | real | 0.4542 | Conditional | Only when geometryScheme == 1 | Rotational transform |
| psiAHat | real | 0.15596 | Conditional | Only when geometryScheme == 1 | Normalized toroidal flux at LCFS |
| aHat | real | 0.5585 | Conditional | Only when geometryScheme == 1 | Effective minor radius at LCFS |
| equilibriumFile | string | ``'' | Conditional | Only when geometryScheme == 5, 11, or 12 | Filename for magnetic equilibrium (vmec wout or .bc) |
| VMECRadialOption | integer | 1 | Conditional | Only when geometryScheme == 5, 11 or 12 | Controls interpolation vs nearest surface lookup |
| rippleScale | real | 1.0 | Conditional | Only when geometryScheme == 5 | Scales VMEC geometry components |
| Zs | 1D array of reals | 1.0 | No (defaulted) | Always | Charges of each species (proton units) |
| mHats | 1D array of reals | 1.0 | No (defaulted) | Always | Masses of each species (reference mass units) |
| nHats | 1D array of reals | 1.0 | Conditional | Whenever RHSMode == 1 | Densities of each species |
| THats | 1D array of reals | 1.0 | Conditional | Whenever RHSMode == 1 | Temperatures of each species |
| dnHatdpsiHats | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 0 | Radial density gradients w.r.t psiHat |
| dTHatdpsiHats | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 0 | Radial temperature gradients w.r.t psiHat |
| dnHatdpsiNs | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 1 | Radial density gradients w.r.t psiN |
| dTHatdpsiNs | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 1 | Radial temperature gradients w.r.t psiN |
| dnHatdrHats | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 2 | Radial density gradients w.r.t rHat |
| dTHatdrHats | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 2 | Radial temperature gradients w.r.t rHat |
| dnHatdrNs | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 3 | Radial density gradients w.r.t rN |
| dTHatdrNs | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 3 | Radial temperature gradients w.r.t rN |
| withAdiabatic | Boolean | .false. | Conditional | When RHSMode == 1 and includePhi1 == .true. | Add adiabatic species to quasineutrality |
| adiabaticZ | real | -1.0 | Conditional | When includePhi1 == .true. and withAdiabatic == .true. | Charge of adiabatic species |
| adiabaticMHat | real | 5.44617e-4 | Conditional | When includePhi1 == .true. and withAdiabatic == .true. | Mass of adiabatic species |
| adiabaticNHat | real | 1.0 | Conditional | When includePhi1 == .true. and withAdiabatic == .true. | Density of adiabatic species |
| adiabaticTHat | real | 1.0 | Conditional | When includePhi1 == .true. and withAdiabatic == .true. | Temperature of adiabatic species |
| withNBIspec | Boolean | .false. | Conditional | When `RHSMode == 1`, `includePhi1 == .true.`, `readExternalPhi1 == .false.`, and typically `quasineutralityOption == 1` | Add an NBI species to quasineutrality only, not to the kinetic equation |
| NBIspecZ | real | 1.0 | Conditional | When withNBIspec== .true. | Charge of NBI species |
| NBIspecNHat | real | 0.0 | Conditional | When withNBIspec== .true. | Density of NBI species |
| Delta | real | 4.5694e-3 | Conditional | Standard kinetic / transport workflows | Normalized gyroradius scale, roughly the reference `rho_*` used in the SFINCS normalization |
| alpha | real | 1.0 | Conditional | Standard kinetic / transport workflows | Electrostatic normalization factor, approximately `e * PhiBar / TBar`; it sets the normalization used for `Phi`, `Er`, and `dPhiHatd*` quantities |
| nuPrime | real | 1.0 | Conditional | Only when RHSMode == 3 | Dimensionless collisionality for monoenergetic coeffs |
| EStar | real | 0.0 | Conditional | Only when RHSMode == 3 | Normalized radial E field for monoenergetic coeffs |
| EParallelHat | real | 0.0 | Conditional | Standard kinetic solves | Inductive / applied parallel electric field in SFINCS normalization |
| dPhiHatdpsiHat | real | 0.0 | Conditional | When inputRadialCoordinateForGradients == 0 | Electrostatic potential gradient w.r.t psiHat |
| dPhiHatdpsiN | real | 0.0 | Conditional | When inputRadialCoordinateForGradients == 1 | Electrostatic potential gradient w.r.t psiN |
| dPhiHatdrHat | real | 0.0 | Conditional | When inputRadialCoordinateForGradients == 2 | Electrostatic potential gradient w.r.t rHat |
| dPhiHatdrN | real | 0.0 | Conditional | When inputRadialCoordinateForGradients == 3 | Electrostatic potential gradient w.r.t rN |
| Er | real | 0.0 | Conditional | When inputRadialCoordinateForGradients == 4 | Radial electric field |
| collisionOperator | integer | 0 | No (defaulted) | Always | Choice of collision operator: `0` = full linearized Fokker-Planck, `1` = pitch-angle-scattering / Lorentz model |
| constraintScheme | integer | -1 | No (defaulted) | Always | Constraint control for null space and conservation |
| includeXDotTerm | Boolean | .true. | Conditional | When radial E field is nonzero | Include speed-change term from E_r |
| includeElectricFieldTermInXiDot | Boolean | .true. | Conditional | When radial E field is nonzero | Include pitch-angle-change term from E_r |
| useDKESExBDrift | Boolean | .false. | Conditional | When radial electric-field terms are active | Use the DKES-style `E x B` drift formula rather than the full-trajectory form |
| includePhi1 | Boolean | .false. | Conditional | Whenever RHSMode == 1 | Include first-order potential Phi1 |
| readExternalPhi1 | Boolean | .false. | Conditional | When includePhi1 == .true. | Recognized input switch for reading `Phi1Hat` from an external file, but not currently supported end-to-end in this `DKX` checkout |
| externalPhi1Filename | string | ``externalPhi1.h5'' | Conditional | When readExternalPhi1 == .true. | Filename for the external `Phi1Hat` input in that same recognized-but-not-fully-supported path |
| includePhi1InKineticEquation | Boolean | .true. | Conditional | When includePhi1 == .true. | Couple Phi1 into kinetic equation |
| includePhi1InCollisionOperator | Boolean | .false. | Conditional | When includePhi1 == .true. | Include Phi1 in collision operator |
| quasineutralityOption | integer | 1 | Conditional | When includePhi1 == .true. and readExternalPhi1 == .false. | Choice of quasineutrality equation (1 or 2) |
| includeTemperatureEquilibrationTerm | Boolean | .false. | Conditional | Whenever RHSMode == 1 | Include temperature equilibration term |
| magneticDriftScheme | integer | 0 | Conditional | Whenever RHSMode == 1 | Control poloidal/toroidal magnetic drifts |
| EParallelHatSpec | 1D array of reals | 0.0 | Conditional | When used in kinetic solves | Species-dependent parallel forcing / drive term; this is not one of the most commonly used public `DKX` input paths |
| Ntheta | integer | 15 | No (defaulted) | Always | Poloidal grid points |
| Nzeta | integer | 15 | No (defaulted) | Always | Toroidal grid points per period |
| Nxi | integer | 16 | No (defaulted) | Always | Pitch-angle grid (Legendre polynomials) |
| Nx | integer | 5 | No (defaulted) | Always | Energy grid points |
| solverTolerance | real | 1e-6 | Conditional | When useIterativeLinearSolver == .true. | Krylov solver convergence tolerance |
| NL | integer | 4 | Conditional | When collisionOperator == 0 | Legendre polynomials for Rosenbluth potentials |
| NxPotentialsPerVth | real | 40.0 | Conditional | When collisionOperator == 0 and xGridScheme ≠ 5 | Rosenbluth grid points (obsolete) |
| xMax | real | 5.0 | Conditional | When collisionOperator == 0 and xGridScheme ≠ 5 | Rosenbluth max speed (obsolete) |
| forceOddNthetaAndNzeta | Boolean | .true. | No (defaulted) | Always | Force odd Ntheta and Nzeta |
| thetaDerivativeScheme | integer | 2 | No (defaulted) | Always | Poloidal discretization (0=spectral, ...) |
| zetaDerivativeScheme | integer | 2 | No (defaulted) | Always | Toroidal discretization (0=spectral, ...) |
| ExBDerivativeSchemeTheta | integer | 0 | Conditional | When radial electric-field terms are active | `E x B` drift upwinding / derivative scheme in `theta` |
| ExBDerivativeSchemeZeta | integer | 0 | Conditional | When radial electric-field terms are active | `E x B` drift upwinding / derivative scheme in `zeta` |
| magneticDriftDerivativeScheme | integer | 3 | Conditional | When `magneticDriftScheme != 0` | Magnetic-drift derivative / upwinding scheme |
| xGridScheme | integer | 5 | Conditional | When RHSMode == 1 or 2 | Speed discretization scheme |
| xPotentialsGridScheme | integer | 2 | Conditional | When RHSMode == 1 or 2 and xGridScheme == 5 | Rosenbluth potential grid scheme |
| xDotDerivativeScheme | integer | 0 | Conditional | When includeXDotTerm == .true. | Collisionless differentiation matrix |
| useIterativeLinearSolver | Boolean | .true. | No (defaulted) | Always | Use iterative (vs direct) solver |
| whichParallelSolverToFactorPreconditioner | integer | 1 | No (defaulted) | Always | Solver for preconditioner factorization |
| PETSCPreallocationStrategy | integer | 1 | No (defaulted) | Always | Memory allocation strategy for matrix |
| reusePreconditioner | Boolean | .true. | Conditional | When includePhi1 == .true. | Reuse preconditioner across iterations |
| psiHat_wish | real | -1 | Conditional | When inputRadialCoordinate == 0 | Requested flux surface (psiHat) |
| psiN_wish | real | 0.25 | Conditional | When inputRadialCoordinate == 1 | Requested flux surface (psiN) |
| rHat_wish | real | -1 | Conditional | When inputRadialCoordinate == 2 | Requested flux surface (rHat) |
| rN_wish | real | 0.5 | Conditional | When inputRadialCoordinate == 3 | Requested flux surface (rN) |
| epsilon_t | real | -0.07053 | Conditional | Only when geometryScheme == 1 | Toroidal variation in B |
| epsilon_h | real | 0.05067 | Conditional | Only when geometryScheme == 1 | Helical variation in B |
| epsilon_antisymm | real | 0.0 | Conditional | Only when geometryScheme == 1 | Stellarator-antisymmetric variation in B |
| helicity_l | integer | 2 | Conditional | When geometryScheme == 1 or 5 with rippleScale ≠ 1 | Poloidal mode of helical variation |
| helicity_n | integer | 10 | Conditional | When geometryScheme == 1 or 5 with rippleScale ≠ 1 | Toroidal mode of helical variation |
| helicity_antisymm_l | integer | 1 | Conditional | Only when geometryScheme == 1 | Poloidal mode of antisymmetric variation |
| helicity_antisymm_n | integer | 0 | Conditional | Only when geometryScheme == 1 | Toroidal mode of antisymmetric variation |
| VMEC_Nyquist_option | integer | 1 | Conditional | Only when geometryScheme == 5 | VMEC mode number handling |
| min_Bmn_to_load | real | 0.0 | Conditional | When geometryScheme == 5, 11, or 12 | Filter cutoff for B field harmonics |
| nu_n | real | 8.330e-3 | Conditional | Standard kinetic / transport workflows (`RHSMode = 1`, and commonly `2`) | Normalized collisionality. In `RHSMode = 3`, `nu_n` is effectively overridden by `nuPrime` |
| xGrid_k | integer | 0 | Conditional | When `RHSMode == 1 or 2` and `xGridScheme` is one of `{1,2,5,6}` | Orthogonal-polynomial weight exponent for the speed grid |
| Nxi_for_x_option | integer | 1 | No (defaulted) | Always | How Nxi depends on speed |
| preconditioner_species | integer | 1 | Conditional | When `useIterativeLinearSolver == .true.` and `Nspecies >= 2` | Species coupling retained in the preconditioner |
| preconditioner_x | integer | 1 | Conditional | When `useIterativeLinearSolver == .true.` and `RHSMode` is one of `{1,2}` | Speed-grid coupling retained in the preconditioner |
| preconditioner_x_min_L | integer | 0 | Conditional | When `preconditioner_x != 0` | Legendre-mode threshold for preconditioner simplification |
| preconditioner_theta | integer | 0 | Conditional | When useIterativeLinearSolver == .true. | Theta coupling in preconditioner |
| preconditioner_theta_min_L | integer | 0 | Conditional | When `preconditioner_theta != 0` | Legendre-mode threshold for preconditioner simplification |
| preconditioner_zeta | integer | 0 | Conditional | When useIterativeLinearSolver == .true. | Zeta coupling in preconditioner |
| preconditioner_zeta_min_L | integer | 0 | Conditional | When `preconditioner_zeta != 0` | Legendre-mode threshold for preconditioner simplification |
| preconditioner_xi | integer | 1 | Conditional | When useIterativeLinearSolver == .true. | Pitch-angle coupling (0=full, 1=tridiag) |
| preconditioner_magnetic_drifts_max_L | integer | 2 | Conditional | When useIterativeLinearSolver == .true. | Legendre mode cutoff for magnetic drift terms |
| export_full_f | Boolean | .false. | No (defaulted) | Always | Export full distribution function |
| export_delta_f | Boolean | .false. | No (defaulted) | Always | Export perturbed distribution function |
| export_f_theta_option | integer | 2 | Conditional | When export_full_f or export_delta_f == .true. | Theta grid control for export |
| export_f_zeta_option | integer | 2 | Conditional | When export_full_f or export_delta_f == .true. | Zeta grid control for export |
| export_f_theta | 1D array of reals | 0.0 | Conditional | When `export_f_theta_option` selects explicit / custom sampling | Theta values for distribution export |
| export_f_zeta | 1D array of reals | 0.0 | Conditional | When `export_f_zeta_option` selects explicit / custom sampling | Zeta values for distribution export |
| export_f_xi_option | integer | 1 | Conditional | When export_full_f or export_delta_f == .true. | Xi discretization for export |
| export_f_xi | 1D array of reals | 0.0 | Conditional | When export_f_xi_option == 1 | Xi values for distribution export |
| export_f_x_option | integer | 0 | Conditional | When export_full_f or export_delta_f == .true. | Speed grid control for export |
| export_f_x | 1D array of reals | 1.0 | Conditional | When `export_f_x_option` selects explicit / custom sampling | Speed values for distribution export |

#### Input Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

#### Optional gradient responses

Set the following controls under `stage3.dkx` in `config.yaml`. The command line accepts the same names with hyphens, such as `--response-mode fd_gradients`.

```yaml
stage3:
  dkx:
    response_mode: fd_gradients
    perturb_density_species: "D,e"
    perturb_temperature_species: "D,e"
    dkap_density: 0.5
    dkap_temperature: 0.5
    perturb_rel_step: 0.5
```

`response_mode` defaults to `none`. Species selections default to empty. Step controls default to `0.5`. In `fd_gradients` mode, preparation matches names without regard to case and stores the spelling from `[species].names`, so selecting `d` stores `D` when the input species is `D`. Within each channel, it keeps only the first occurrence of each name. Preparation rejects unknown names and input names that differ only by case. Selected names must match `\w+` for scheduling. Select at least one species. Each radius gets a baseline and siblings ending in `_fd_n_<species>` or `_fd_t_<species>`.

The normalized gradients are `kappa_n = -(dn/d(rho))/n` and `kappa_T = -(dT/d(rho))/T`. The density channel uses `delta = -max(dkap_density, perturb_rel_step * abs(kappa_n))`. It adds `delta` to `kappa_n`. It subtracts the same value from `kappa_T` to preserve the species pressure gradient. The temperature channel uses `delta = max(dkap_temperature, perturb_rel_step * abs(kappa_T))`. It changes only `kappa_T`. The writer converts these gradients back to `dNHatdrNs` and `dTHatdrNs` per unit `rho`. Local density, temperature, geometry and `Er` stay fixed. The gradients of all other species stay fixed. In `fd_gradients` mode, step controls must be finite and nonnegative, and effective steps must be finite and nonzero.

### Output Specification


The Snakemake forward pass runs the `DKX` radial scan through the `stage3_prepare` checkpoint, one `stage3_run_one` job per baseline or perturbation, and `stage3_collect`. It produces one aggregated handoff file plus a per-surface run tree. See [Per-surface fan-out](../mvp-pipeline.md#per-surface-fan-out-stages-3-and-4).

The forward pass supplies flux profiles versus radius to `NEOPAX` (Stage 5) in the HDF5 file `dkx_flux_profiles.h5`. The `collect` step builds this file from every per-surface `result.json`. The root contract validator `validate_dkx_flux` in `src/io_contracts.py` checks the following schema:

| Dataset | Shape | Meaning |
|---------|-------|---------|
| `rho` | `(n_radii,)` | Normalized radial coordinate of each aggregated surface |
| `r` | `(n_radii,)` | Radial coordinate written from `DKX`'s `rHat` (alias of `rHat`) |
| `rHat` | `(n_radii,)` | `DKX` `rHat` radial coordinate |
| `Gamma` | `(n_species, n_radii)` | Neoclassical particle flux in NEOPAX units |
| `Q` | `(n_species, n_radii)` | Neoclassical heat flux in NEOPAX units |
| `Upar` | `(n_species, n_radii)` | Parallel-flow observable |
| `species_names` | `(n_species,)` | UTF-8 species labels (root dataset) |

The validator requires finite, strictly increasing radial coordinates, equal `r` and `rHat`, consistent species and radius dimensions, and finite flux values. If `axis_zero_padded` is true, the first radius and flux column must be zero. The standalone scan and `collect` accept `--output` for a custom aggregate path. Without this option, the command writes the file under `--output-dir`.

**Gradient-response datasets**

Response mode keeps baseline `Gamma`, `Q` and `Upar` on a unique radial axis. It adds the following datasets. `P` counts pairs of channel and input species. `S` counts output species. `R` counts radii, including any synthetic axis.

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `Gamma_perturbed`, `Q_perturbed` | `(P, S, R)` | Full perturbed fluxes in baseline units |
| `perturb_delta` | `(P, R)` | Signed normalized-gradient increment |
| `perturb_present` | `(P, R)` | Boolean mask, true where a perturbation was measured |
| `response_label` | `(P,)` | `density_gradient` or `temperature_gradient` |
| `perturb_species` | `(P,)` | Name of the perturbed input species |

Pairs follow the density species list, then the temperature species list. Output species follow `species_names`. Compute `(F_perturbed - F_base)/delta` only where `perturb_present` is true. The density response includes the compensating temperature-gradient change. At a synthetic axis, response fluxes and increments are zero. The writer sets `perturb_present = false` there. The writer exports no `Upar` response. Baseline mode omits these six datasets.

The collector reads fluxes from `result.json` and signed steps from the manifest, without reading namelists. Existing scans must run `prepare` with this version before collection, which regenerates the manifest and reuses matching completed results.

The `response_note` attribute describes the response convention when these datasets are present. The baseline contract validator accepts extra datasets but does not validate the response schema.

The writer keeps `r = rHat`. It stores `rho` separately. [NEOPAX compatibility work](../potential_issues.md#stage-5----transport) covers the gradient basis and radial grid.

Root attributes

- `axis_zero_padded` (bool, **required** by the contract): the scan drops the magnetic-axis (`rho = 0`) surface, so when no aggregated surface sits at `rho = 0` the `collect` step prepends a zero-flux `rho = 0` column and sets this flag `true`.
- Provenance echoes read back from the scan `manifest.json` rather than measured live: `backend`, `max_parallel`, `profiles_source`, `source_transport_solution`, `source_dkx_template`, `time_index`, `time_value`, `include_phi1`. `profiles_source` is one of `analytical` (kinetic profiles built from the analytical parameters in the `common_input.toml` `[profiles]` block, on `analytical_n_radii` cell faces), `transport_h5` (read from a NEOPAX `transport_solution.h5`, taking its `rho_face` / `*_faces` face state rather than the cell-centered datasets, for the same reason as `prescribed`), or `prescribed` (read from SI profile arrays written into that same `[profiles]` block by the closed loop's Stage 5 post-processing; see [Closing the Loop](../mvp-pipeline.md#closing-the-loop) for the array contract). The latter two take their radial grid from the profile data rather than from `analytical_n_radii`. Under `prescribed` the scan reads the block's `*_face` arrays and samples the transport **face** grid `linspace(0, rho_edge, n_radial + 1)`, not the cell-centered arrays NEOPAX itself reads back; a block carrying only the centered arrays is rejected rather than extrapolated. `analytical_n_radii` counts cell **faces** and defaults to `[geometry].n_radial + 1`, the transport face grid, matching Stage 4. Iteration 1 of the closed loop therefore scans the radii iterations 2 onward read back under `prescribed`, and the face reconstruction described next runs on NEOPAX's own faces, where it reproduces NEOPAX rather than approximating it. An explicit larger or smaller value is still accepted and is still valid input for NEOPAX to interpolate from, since any face grid spanning `[0, rho_edge]` is, but off NEOPAX's faces the reconstruction is a discretization of its own.
- The `dNHatdrNs` / `dTHatdrNs` the bridge patches into each namelist reach it by the same mechanism Stage 4's `tprim` / `fprim` do, which the [Stage 4 spec](../stage4-turbulence/spec.md#aggregated-forward-pass-outputs-radial-scan) documents once for both stages. `transport_h5` reads `density_grad_faces` / `temperature_grad_faces` out of the solution and rescales them from per metre onto rho; `prescribed` reads `density_grad_face` / `temperature_grad_face` out of the `[profiles]` block, already per unit rho; and `analytical` samples the profile formula on the cell centers the faces bound, then rebuilds both the face values and the face gradients with `stages/common/profile_gradients.py`, the NumPy port of NEOPAX's `CellVariable.face_grad` and `CellVariable.face_value`, under the run's `[boundary]` blocks. Only the consumer differs. Stage 3 takes the gradient plain, without the logarithm and sign flip Stage 4's `tprim` applies.
- Unit / convention notes: `Upar_note`, `normalization_note`, `radius_note` (how `Gamma`/`Q`/`Upar` are converted to NEOPAX units and which coordinate `r`/`rHat` carries).
- `solver_name`, `solver_version`, `solver_revision`, and `wout_sha256` identify the installed solver and equilibrium content.

**Per-surface run tree** under `outputs/<run>/stage3_neoclassical/`. The stage's `manifest.json` records the runs and provenance for collection. Each surface has a baseline directory such as `runs/rho_012_r0p4898/`, plus a directory for each selected perturbation with an `_fd_n_<species>` or `_fd_t_<species>` suffix. Each directory contains `input.namelist`, `payload.json`, native `sfincsOutput.h5`, and extracted `result.json`.

Benchmark repeats and warmup apply to baseline runs only. Each perturbation is solved once because collection reports baseline timings only.

Preparation must run in the DKX stage environment. It reads the installed DKX metadata. It reuses a result only when the generated namelist, installed DKX identity, equilibrium digest, and surface data agree. If the Git revision or WOUT digest is unavailable, preparation reports why reuse is disabled. The WOUT path can come from `--wout-path` or the common configuration. Preparation removes an invalid `result.json` so Snakemake schedules the surface again. On each rerun, the worker clears its completion file before solving. It writes a replacement only after a successful solve and diagnostic validation. Collection rejects missing or incompatible results before opening the aggregate destination.

The worker calls `dkx.api.write_output` followed by `dkx.api.read_output`. It requires `particleFlux_vm_rHat`, `heatFlux_vm_rHat`, `FSABFlow`, `rHat`, and `B0OverBBar`. DKX iteration arrays use the first dimension for species. The scan uses the final iteration after checking its values. The surface run fails if diagnostics are missing or malformed, final values are nonfinite, or unit conversion overflows. The scan applies the established physical conversions to particle and heat fluxes to produce NEOPAX units. It converts parallel flow with `2*B0OverBBar/sqrt(pi)` times `FSABFlow`. The scan does not substitute radial flux fields with different normalizations. Values for each surface remain in the native HDF5 and result records. Aggregate metadata describes the shared fields and conversions.

**Native SFINCS output (per surface):** `sfincsOutput.h5` (HDF5) -- the native solver file. The `DKX` radial scan writes one per flux surface (under `runs/rho_*/`) and aggregates them into the handoff above; the standalone `SFINCS` (Fortran) binary writes it directly. Its fields:

| Field | Availability | Meaning | Primary Use | Normalization |
|-------|--------------|---------|-------------|---------------|
| `particleFlux_vm_rHat` | `RHSMode=1` solve outputs | Neoclassical particle flux in vm normalization (`rHat` coordinate) | **Scan transport input** | `vm` flux normalization, reported on `rHat` radial coordinate |
| `heatFlux_vm_rHat` | `RHSMode=1` solve outputs | Neoclassical heat flux in vm normalization (`rHat` coordinate) | **Scan transport input** | `vm` flux normalization, reported on `rHat` radial coordinate |
| `particleFlux_vd_rN`, `heatFlux_vd_rN` | when `includePhi1=.true.` diagnostics are written | Total (magnetic + `E×B`) flux variants with `Phi1` effects | Transport input (Phi1-on workflows) | `vd` (drift + `E×B`) flux normalization, reported on `rN` |
| `FSABjHat` | solved runs | Flux-surface-averaged parallel current (bootstrap diagnostic) | Equilibrium/diagnostic coupling | `Hat` quantity (SFINCS normalized current) and flux-surface-averaged (`FSAB`) |
| `FSABFlow` | solved runs | Flux-surface-averaged parallel flow by species | Diagnostic | SFINCS normalized flow; flux-surface-averaged (`FSA/FSAB`) |
| `Phi1Hat` | when `includePhi1=.true.` | First-order electrostatic potential on `(theta,zeta)` grid | Diagnostic / analysis | `Hat` potential normalization ($\Phi_1$ normalized by $T_{\mathrm{ref}}/e$) |
| `transportMatrix` | `RHSMode=2/3` with `--compute-transport-matrix` transport-matrix workflow | Transport matrix assembled across `whichRHS` solves | Analysis / reduced-model fitting | Mixed normalized transport coefficients in SFINCS internal normalization |

Normalization notes for output names:
- Suffix `_vm` = magnetic-drift transport normalization; `_vd` = drift + `E\times B` transport normalization.
- Radial suffixes: `_psiHat`, `_psiN`, `_rHat`, `_rN` indicate the radial coordinate used for the reported flux.
- `Hat` denotes SFINCS normalized quantities; `FSA`/`FSAB` denotes flux-surface average.

Common required metadata outputs:

**Grids & Geometry:**
- `theta`, `zeta`: poloidal and toroidal angles on the computational grid.
- `x`: normalized kinetic energy (velocity space grid).
- `BHat`: magnetic field strength (normalized).
- `DHat`: geometric factor (Jacobian-related).
- `VPrimeHat`: derivative of volume w.r.t. flux coordinate.
- `FSABHat2`: flux-surface-averaged $B^2$ quantity.

**Run Metadata:**
- `Nspecies`: number of species in calculation.
- `Ntheta`, `Nzeta`, `Nxi`, `Nx`: resolution settings echoed to output.
- `RHSMode`: solve mode used.
- `NIterations`: number of nonlinear iterations taken.
- `elapsed time (s)`: elapsed wall-clock time written in the transport-matrix workflow.

**Species & Profile Information:**
- `Zs`, `mHats`, `nHats`, `THats`: species charges, mass ratios, densities, temperatures (echoed from input).
- `dnHatdpsiHat`, `dnHatdpsiN`, `dnHatdrHat`, `dnHatdrN`: density-gradient values written in all supported radial-coordinate forms.
- `dTHatdpsiHat`, `dTHatdpsiN`, `dTHatdrHat`, `dTHatdrN`: temperature-gradient values written in all supported radial-coordinate forms.
- `psiHat`, `psiN`, `rHat`, `rN`: scalar flux-surface location values written in all supported radial-coordinate forms.
- `iota`: rotational transform at the chosen flux surface.

Common conditional physics outputs:

**Radial Flux Families** (coordinate variants: `_psiHat`, `_psiN`, `_rHat`, `_rN`):
- `particleFlux*`: neoclassical particle flux per species.
- `heatFlux*`: neoclassical heat flux per species.
- `momentumFlux*`: neoclassical parallel momentum flux.
- `FSABFlow` and `FSABFlow_vs_x`: flux-surface-averaged flows by species.

**Flux-Surface Diagnostics:**
- `FSABjHat`: flux-surface-averaged parallel current (bootstrap diagnostic).
- `FSABFlow`: flux-surface-averaged parallel flow.
- `jHat`, `flow`: local (non-flux-surface-averaged) parallel current and flow.
- `densityPerturbation`, `pressurePerturbation`: moment diagnostics derived from the perturbed distribution.
- `NTV`: neoclassical toroidal viscosity (when Phi1 included).

**Classical Transport** (when included):
- `classicalParticleFluxNoPhi1_*`, `classicalHeatFluxNoPhi1_*`: static classical-flux diagnostics without Phi1 contributions.
- `classicalParticleFlux_*`, `classicalHeatFlux_*`: per-iteration classical-flux diagnostics in the reported radial-coordinate variants.

**Distribution Function Exports** (only when `export_full_f` or `export_delta_f` enabled):
- `delta_f`: perturbation part of distribution function on grid (theta, zeta, xi, x).
- `full_f`: total distribution (Maxwellian + perturbation).
- `export_f_theta`, `export_f_zeta`, `export_f_xi`, `export_f_x`: parameter specification for export grid resolution.

**Solver Diagnostics** (DKX-specific):
- `linearSolver*`: residual norms, iteration counts, convergence flags, and related solve metadata written for `RHSMode=1` output solves.
- `transportMatrix`: full matrix when `RHSMode=2/3` is run with `--compute-transport-matrix`.

**Handoff to `Trinity3D`:** The `Trinity3D` adapter reads:
- When `includePhi1=.false.` (standard neoclassical): `particleFlux_vm_rN`, `heatFlux_vm_rN`.
- When `includePhi1=.true.` (with Phi1 effects): `particleFlux_vd_rN`, `heatFlux_vd_rN` (includes both magnetic drift and E×B contributions).

#### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

### Governing Equations

First-order drift-kinetic equation:

$$(v_\parallel \mathbf{b} + \frac{d\Phi_0}{dr}\frac{\mathbf{B}\times\nabla r}{B^2})\cdot\nabla f_{s1} + [\text{mirror/drift terms}]\frac{\partial f_{s1}}{\partial\xi} - (\mathbf{v}_{ms}\cdot\nabla r)\frac{Z_s e}{2T_s x_s}\frac{d\Phi_0}{dr}\frac{\partial f_{s1}}{\partial x_s}$$

$$+ (\mathbf{v}_{ms}\cdot\nabla r)\left[\frac{1}{n_s}\frac{dn_s}{dr} + \frac{Z_s e}{T_s}\frac{d\Phi_0}{dr} + (x_s^2 - \frac{3}{2})\frac{1}{T_s}\frac{dT_s}{dr}\right]f_{sM} = C_s[f_{s1}] + S_s$$

Collision operator: $C_s[f_s] = \sum_b C_{sb}^l[f_s, f_b]$ with Lorentz, energy-diffusion, and field-particle components.

When Phi1 is included, coupled to quasineutrality.

**Reference:** `stellarator_workflow.tex`, Section 4.5.

---

## Installation & Platform

**`sfincs`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-3-sfincs-fortran
```

Install DKX from the locked Pixi environment. From the `stages/` directory, run:

```
pixi install --locked --environment stage-3-dkx
```

The CPU environment supports Linux x86-64, Linux ARM64, and macOS ARM64. `stage-3-dkx-gpu` targets Linux x86-64 with CUDA 12.

**`neo-jax`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-3-neo-jax
```

See `docs/mvp-pipeline.md` for run commands and I/O details.

> [!TODO]
> Document installation instructions and platform notes for `NEO_JAX` and `NEO`.

---

## Convergence & Validity

> [!TODO]
> Document convergence behavior, known failure modes, and recommended tolerances for all three sub-stages.

---

## API Documentation

> [!TODO]
> Document key entry points, configuration parameters, and usage examples for all three sub-stages.

---

## Scripts & Workflows

For a native single-surface DKX output, run the direct Pixi task from `stages/` after Stage 1.

```sh
pixi run --locked -e stage-3-dkx stage-3-dkx
```

The task calls `dkx sfincs write-output` with `--input`, `--out`, and `--equilibrium-file`. It uses `sfincs_input.HSX_vacuum_ns201_quickrun` and the Stage 1 wout. It writes `outputs/quick_run/stage3_neoclassical/sfincsOutput_quickrun.h5`.

For the aggregate consumed by NEOPAX, run the radial scan after Stages 1 and 2.

```sh
pixi run --locked -e stage-3-dkx stage-3-dkx-radial-scan --max-parallel 1
```

The scan uses `common_input.toml` for profiles and geometry paths. It writes `outputs/quick_run/stage3_neoclassical/dkx_flux_profiles.h5`. Use `--dkx-template` to override the namelist template. Use `--wout-path` to override VMEC geometry. Use `--boozer-path` to override the Boozer file used for the analytical radius. Use `--output` to change the aggregate filename. Snakemake supplies the equivalent configuration through `stage3.dkx`.

Use a positive `--cores-per-run` to set CPU threads through `DKX_CORES`. A zero or negative value keeps an inherited `DKX_CORES` value, or lets DKX use its default if the variable is unset. Inherited `DKX_CORES=0` enables full-width sizing. BLAS and OMP pools retain a minimum of one thread. Use `--max-parallel` for simultaneous workers. Use `--gpu-ids` with `--backend gpu` to select devices.

See [the MVP reference](../mvp-pipeline.md#stage-3----neoclassical) for commands from the repository root and the difference between native output and the aggregate.

**`SFINCS` (Fortran, via Pixi):** From the `stages`/ directory:

```
pixi run stage-3-sfincs-fortran
```

Alternative implementation to `DKX`. Consumes the same input namelist and writes the native `sfincsOutput.h5`, a different file from the `dkx_flux_profiles.h5` forward-chain handoff and not wired into the Snakemake forward pass; the task stages the namelist as `input.namelist` in the output directory before invocation because the Fortran binary reads that filename from its working directory.

**Input:** the Stage 1 wout and the shared `sfincs_input.HSX_vacuum_ns201_quickrun` namelist.
**Output:** `outputs/quick_run/stage3_neoclassical/sfincsOutput.h5` (the native SFINCS file, separate from the `DKX` handoff).

See `docs/mvp-pipeline.md` for full I/O details.

> [!TODO]
> Add standalone run scripts and workflows for `NEO_JAX`, `NEO`, and `SFINCS`.

---

## W&B Tracking

**Project:** `driftless-star-stage3-neoclassical`

> [!TODO]
> Set up W&B tracking for all three sub-stages.

---

## Container Specification (Phase 2)

**`DKX`:** Built from the single templated `stages/Dockerfile` using build arguments:

```
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-3-dkx stages/        # CPU
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-3-dkx-gpu --build-arg CUDA_VERSION=12 stages/  # GPU
```

CI is configured to build and publish `ghcr.io/driftless-star/driftless-star:stage-3-dkx-cpu` and `stage-3-dkx-gpu` through `.github/workflows/containers.yml`. Apptainer references use the corresponding `apptainer-stage-3-dkx-cpu` and `apptainer-stage-3-dkx-gpu` tags.

See [guide](../guide.md#container-architecture) for full architecture details.

> [!TODO]
> Define container specifications for `NEO_JAX`, `NEO`, and `SFINCS`.

---

## Tests (Phase 2)

See [guide](../guide.md#writing-tests) for examples.

> [!TODO]
> Write unit, regression, and integration tests for all three sub-stages.

---

## Claude Skills

See [guide](../guide.md#step-7-create-claude-skills) for skill types.

> [!TODO]
> Create development and operational Claude skills for all three sub-stages.
