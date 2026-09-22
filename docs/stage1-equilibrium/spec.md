# Stage 1: Equilibrium

## Overview

Stage 1 solves the three-dimensional ideal-MHD equilibrium problem, producing the magnetic field geometry and flux-surface profiles that all downstream stages depend on. This is the entry point of the forward-pass pipeline.

**Physics:** Given a plasma boundary shape and profile guesses (pressure, rotational transform or current), find the 3D magnetic equilibrium satisfying force balance: $\nabla p = \mathbf{J} \times \mathbf{B}$, $\nabla \cdot \mathbf{B} = 0$.

**Position in pipeline:** This stage has no upstream dependencies. Its output (`wout_*.nc`) is consumed by Stage 2 (Boozer Transform) and also directly by some turbulence and transport codes.

The reference manuscript `stellarator_workflow/stellarator_workflow.tex` describes `VMEC++` and the historical `vmec_jax` implementation in Section 4.1, and `DESC` in Section 4.2. The current Stage 1 implementation is VMEX.

---

## Codes

### VMEX (Primary JAX)

- **Repository:** https://github.com/uwplasma/vmex
- **Language:** Python/JAX
- **Role:** Equilibrium solving with VMEC-compatible `wout` output

Stage 1 installs [VMEX](https://github.com/uwplasma/vmex) from the commit pinned in [`stages/pixi.toml`](../../stages/pixi.toml).

### VMEC++ (C++ Alternative)
- **Repository:** https://github.com/proximafusion/vmecpp
- **Documentation:** https://proximafusion.github.io/vmecpp/
- **Language:** C++ with Python bindings
- **Role:** From-scratch C++ reimplementation of `VMEC`. Solves fixed- and free-boundary ideal-MHD equilibria. Preserves the standard `wout` downstream contract.

### DESC (Differentiable Alternative)
- **Repository:** https://github.com/PlasmaControl/DESC
- **Language:** Python/JAX
- **Role:** Differentiable pseudo-spectral equilibrium and optimization suite. Can replace `VMEC++` as the equilibrium engine and also perform some downstream computations (Boozer transform, geometry objectives) internally.

### Installation & Platform

Install VMEX through the Pixi environment from the `stages/` directory.

```
pixi install --environment stage-1-vmex
```

**`desc-opt`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-1-desc
```

See `docs/mvp-pipeline.md` for run commands and I/O details.

> [!TODO]
> Document installation instructions and platform notes for `VMEC++` and `DESC`.

---

## Input Specification

Reference: `stellarator_io_reference.tex`, Section 3.1.

### Physical Inputs

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `RBC(m,n)` | 2D array (float) | Boundary R cosine Fourier coefficients | User-specified |
| `ZBS(m,n)` | 2D array (float) | Boundary Z sine Fourier coefficients | User-specified |
| `AM` | 1D array (float) | Pressure profile coefficients | User-specified |
| `AM_AUX_*` | arrays (float) | Auxiliary pressure arrays (alternative to AM) | User-specified |
| `AI` | 1D array (float) | Rotational transform iota coefficients (if iota-prescribed) | User-specified |
| `AC` | 1D array (float) | Current profile coefficients (if current-prescribed) | User-specified |
| `AC_AUX_*` | arrays (float) | Auxiliary current arrays | User-specified |
| `PHIEDGE` | scalar (float) | Total toroidal flux (magnetic scale) | User-specified |

### Resolution & Solver Controls

| Field                   | Type          | Description                                                 |
| ----------------------- | ------------- | ----------------------------------------------------------- |
| `NS`                    | int or array  | Number of radial grid points (can be a multi-grid sequence) |
| `MPOL`                  | int           | Maximum poloidal mode number                                |
| `NTOR`                  | int           | Maximum toroidal mode number                                |
| `NITER` / `NITER_ARRAY` | int / array   | Iteration budgets                                           |
| `FTOL` / `FTOL_ARRAY`   | float / array | Convergence tolerances                                      |

### Input Formats
- **INDATA files:** Fortran-style text `input.NAME` format (`vmex` and `VMEC++`)
- **JSON:** Programmatic input (`VMEC++` only)
- **Python objects:** In-memory API (both `VMEC++` and `vmex`)
- **Hot restart:** Previous converged output state as initial guess

### Input Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

---

## Output Specification

Reference: `stellarator_io_reference.tex`, Section 3.1.

### Primary Output: `wout_*.nc` (NetCDF)

#### Geometry Scalars

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `aspect` | scalar (float) | Aspect ratio R/a | Objective |
| `Aminor_p` | scalar (float) | Minor radius | Geometry |
| `Rmajor_p` | scalar (float) | Major radius | Geometry |
| `volume_p` | scalar (float) | Plasma volume | Objective |
| `betatotal` | scalar (float) | Total plasma beta | Objective |
| `b0` | scalar (float) | Magnetic field on axis | Geometry |
| `volavgB` | scalar (float) | Volume-averaged |B| | Geometry |
| `fsqr` | scalar (float) | Force residual (radial) | QA signal |
| `fsqz` | scalar (float) | Force residual (vertical) | QA signal |
| `fsql` | scalar (float) | Force residual (lambda) | QA signal |

#### Radial Profiles

| Field | Type | Description |
|-------|------|-------------|
| `presf` | 1D array | Pressure on full mesh |
| `pres` | 1D array | Pressure on half mesh |
| `phi` | 1D array | Toroidal flux |
| `phipf` | 1D array | d(phi)/ds on full mesh |
| `chi` | 1D array | Poloidal flux |
| `chipf` | 1D array | d(chi)/ds on full mesh |
| `iotas` | 1D array | Rotational transform on half mesh |
| `iotaf` | 1D array | Rotational transform on full mesh |
| `q_factor` | 1D array | Safety factor (1/iota) |
| `jcuru` | 1D array | Poloidal current density |
| `jcurv` | 1D array | Toroidal current density |
| `buco` | 1D array | Covariant B_theta (Boozer I) |
| `bvco` | 1D array | Covariant B_zeta (Boozer G) |

#### Spectral Geometry

| Field | Type | Description |
|-------|------|-------------|
| `rmnc` | 2D array (ns x mnmax) | R cosine Fourier coefficients |
| `zmns` | 2D array (ns x mnmax) | Z sine Fourier coefficients |
| `lmns` | 2D array (ns x mnmax) | Lambda sine Fourier coefficients |
| `bmnc` | 2D array (ns x mnmax) | |B| cosine Fourier coefficients |
| `gmnc` | 2D array (ns x mnmax) | Jacobian sqrt(g) cosine coefficients |
| `bsubumnc` | 2D array (ns x mnmax) | B_theta cosine coefficients |
| `bsubvmnc` | 2D array (ns x mnmax) | B_zeta cosine coefficients |
| `bsubsmns` | 2D array (ns x mnmax) | B_s sine coefficients |
| `currumnc` | 2D array (ns x mnmax) | J_theta cosine coefficients |
| `currvmnc` | 2D array (ns x mnmax) | J_zeta cosine coefficients |

#### Python API Objects (`vmex` / `VMEC++`)

| Object | Description |
|--------|-------------|
| `wout` | Full wout data structure |
| `threed1_volumetrics` | 3D volume integrals |
| `jxbout` | J x B force-balance diagnostics |
| `mercier` | Mercier stability criterion |

### Subset Handed to Next Stage

Stage 2 (`BOOZ_XFORM` / `booz_xform_jax`) needs the **full** equilibrium spectrum and profiles in `wout_*.nc`. `GX`, `Trinity3D`, and `NEOPAX` geometry readers also consume wout-level data for field-line geometry, rotational transform, and surface metrics.

### Outputs Used as Objectives

- Aspect ratio, volume, beta, target iota(s): direct design objectives
- Mercier criterion: stability objective
- Residuals `fsqr`, `fsqz`, `fsql`: QA convergence signals, not physics design objectives

### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

---

## Governing Equations

The equilibrium satisfies ideal-MHD force balance:

$$\nabla p = \mathbf{J} \times \mathbf{B}, \quad \nabla \cdot \mathbf{B} = 0, \quad \mathbf{J} = \frac{1}{\mu_0} \nabla \times \mathbf{B}$$

`VMEC++` finds the stationary point of the energy functional (Hirshman & Whitson 1983):

$$W = \frac{1}{(2\pi)^2} \int \left( \frac{B^2}{2} + \frac{p}{\gamma - 1} \right) dV$$

In `VMEC++` flux coordinates with the stream function lambda:

$$u = \theta + \lambda(s, \theta, \zeta), \quad \frac{du}{d\zeta} = \iota(s)$$

The contravariant field components are:

$$B^\zeta = \frac{\Phi'(s) + \text{lamscale} \cdot \partial_\theta \lambda}{\text{signgs} \cdot \sqrt{g} \cdot 2\pi}$$

$$B^\theta = \frac{\chi'(s) - \text{lamscale} \cdot \partial_\zeta \lambda}{\text{signgs} \cdot \sqrt{g} \cdot 2\pi}$$

`DESC` solves the same physics in a pseudo-spectral formulation:

$$\mathbf{B} = \frac{\partial_\rho \psi}{2\pi\sqrt{g}} \left[ \left(\iota - \frac{\partial\lambda}{\partial\zeta}\right) \mathbf{e}_\theta + \left(1 + \frac{\partial\lambda}{\partial\theta}\right) \mathbf{e}_\zeta \right]$$

Reference: `stellarator_workflow.tex`, Sections 4.1-4.2.

---

## Convergence & Validity

> [!TODO]
> Document convergence behavior, known failure modes, and recommended production tolerances for this revision.

---

## API Documentation

From the repository root, use `stages/stage1-equilibrium/run_vmex.py` with required `--input` and `--output` paths and optional `--device`, which defaults to `auto` and is forwarded unchanged to VMEX. Snakemake explicitly selects `cpu` or `gpu` because VMEX's `auto` mode can select CPU for small resolutions even in a GPU image.

A solve that does not converge exits nonzero and fails the stage. Failed WOUT files are discarded, and the solver log remains available for diagnosis.

```sh
pixi run --manifest-path stages/pixi.toml -e stage-1-vmex python \
  stages/stage1-equilibrium/run_vmex.py \
  --input "inputs/quick_run/vmec_input.HSX_vacuum_ns201_quickrun" \
  --output "outputs/custom/stage1_equilibrium/wout_custom.nc" \
  --device cpu
```

---

## Scripts & Workflows

Run the VMEX task from the `stages/` directory.

```
pixi run stage-1-vmex
```

**Input:** `inputs/quick_run/vmec_input.HSX_vacuum_ns201_quickrun`
**Output:** `outputs/quick_run/stage1_equilibrium/wout_HSX_vacuum_ns201_quickrun.nc`

See `docs/mvp-pipeline.md` for full I/O details.

> [!TODO]
> Add standalone run scripts and debugging workflows for `VMEC++` and `DESC`.

---

## W&B Tracking

**Project:** `driftless-star-stage1-equilibrium`

> [!TODO]
> Set up W&B tracking.

---

## Container Specification (Phase 2)

VMEX images use the shared `stages/Dockerfile`. Run these commands from the repository root.

```
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-1-vmex --platform linux/amd64 --tag ghcr.io/driftless-star/driftless-star:stage-1-vmex-cpu stages/  # CPU
docker build --file stages/Dockerfile --build-arg CUDA_VERSION=12 --build-arg ENVIRONMENT=stage-1-vmex-gpu --platform linux/amd64 --tag ghcr.io/driftless-star/driftless-star:stage-1-vmex-gpu stages/  # GPU
```

CI builds and publishes the `ghcr.io/driftless-star/driftless-star:stage-1-vmex-cpu` and `stage-1-vmex-gpu` tags through `.github/workflows/containers.yml`. Apptainer tags add the `apptainer-` prefix.

See [guide](../guide.md#container-architecture) for full architecture details.

> [!TODO]
> Define container specifications for `VMEC++` and `DESC`.

---

## Tests (Phase 2)

> [!TODO]
> Write unit, regression, and integration tests. See [guide](../guide.md#writing-tests) for examples.

---

## Claude Skills

> [!TODO]
> Create development and operational Claude skills. See [guide](../guide.md#step-7-create-claude-skills) for skill types.
