# Stage 2: Boozer Transform

## Overview

Transforms a VMEC-style equilibrium into Boozer coordinates. Boozer coordinates are the standard system for neoclassical analysis because the magnetic field takes a particularly simple covariant form: the covariant components $I(\psi)$ and $G(\psi)$ are flux-surface functions, not angle-dependent. This stage is a coordinate-transformation step, not a physics solver.

**Position in pipeline:** Receives `wout_*.nc` from Stage 1 (Equilibrium). Outputs `boozmn_*.nc` consumed by Stage 3 (Neoclassical) and indirectly by Stage 4/5.

**Reference:** `stellarator_workflow.tex`, Section 4.3.

---

## Codes

### booz_xform_jax (Primary JAX)

- **Repository:** <https://github.com/uwplasma/boozx>
- **Language:** Python/JAX
- **Notes:** Also accepts in-memory wout-like objects, enabling a fully differentiable Stage 1 -> Stage 2 path without writing an intermediate NetCDF file.

### BOOZ_XFORM (Legacy)

- **Repository:** <https://github.com/hiddenSymmetries/booz_xform>
- **Language:** Fortran/Python
- **Notes:** Standard legacy tool used by the stellarator-optimization community.

### Installation & Platform

**`booz_xform` (legacy Fortran/Python):** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-2-booz
```

**`booz_xform_jax`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-2-booz-jax
```

See `docs/mvp-pipeline.md` for run commands and I/O details.

---

## Input Specification

Reference: `stellarator_io_reference.tex`, Section 3.2.

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `wout_*.nc` | NetCDF file | Full VMEC equilibrium output | Stage 1 |

The JAX version (`booz_xform_jax`) does not use a separate `in_booz.*` control file. Resolution (mboz, nboz) and surface selection are specified via the Python API. The legacy `BOOZ_XFORM` uses an `in_booz.*` control file.

`booz_xform_jax` also accepts in-memory wout-like objects directly from Stage 1, bypassing file I/O.

### Input Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

---

## Output Specification

Reference: `stellarator_io_reference.tex`, Section 3.2.

### Primary Output: `boozmn_*.nc` (NetCDF)

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `bmnc_b` | 2D array (ns_b x mnboz) | $\lvert B \rvert$ in Boozer coordinates (cosine coefficients) | Downstream input |
| `rmnc_b` | 2D array | $R$ in Boozer coordinates | Geometry |
| `zmns_b` | 2D array | $Z$ in Boozer coordinates | Geometry |
| `pmns_b` | 2D array | Toroidal angle shift (sine) | Geometry |
| `gmn_b` | 2D array | Jacobian harmonics | Geometry |
| `iota_b` | 1D array | Rotational transform in Boozer coords | Downstream input |
| `pres_b` | 1D array | Pressure in Boozer coords | Downstream input |
| `beta_b` | 1D array | Beta in Boozer coords | Diagnostic |
| `phip_b` | 1D array | $d\phi/ds$ in Boozer coords | Downstream input |
| `phi_b` | 1D array | Toroidal flux in Boozer coords | Downstream input |
| `bvco_b` | 1D array | Boozer $G$ (covariant $B_\zeta$) | Downstream input |
| `buco_b` | 1D array | Boozer $I$ (covariant $B_\theta$) | Downstream input |
| `jlist` | 1D array (int) | Surface indices computed | Metadata |
| `ixm_b` | 1D array (int) | Poloidal mode numbers | Metadata |
| `ixn_b` | 1D array (int) | Toroidal mode numbers | Metadata |

### Subset Handed to Next Stage

`NEO` and `NEO_JAX` need the Boozer spectrum and radial profiles. The forward-pass DKX solver uses VMEC geometry. Its radial scan reads the Boozer output to reconstruct analytical profiles on NEOPAX's minor radius. `NEOPAX` uses the Boozer spectrum through direct field or file readers.

### Outputs Used as Objectives

Non-target Boozer harmonics in `bmnc_b` or symmetry-breaking measures built from them. These are **geometry diagnostics**, not transport state variables. Typical objectives include:

- Quasi-symmetry residual: sum of unwanted harmonics relative to the dominant mode.
- Mirror ratio and helical content derived from the Boozer spectrum.

### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

---

## Governing Equations

Boozer coordinates satisfy the following dual representation of the magnetic field.

**Contravariant form:**

$$\mathbf{B} = \nabla\psi \times \nabla\theta_B + \iota(\psi)\,\nabla\zeta_B \times \nabla\psi$$

**Covariant form:**

$$\mathbf{B} = \beta\,\nabla\psi + I(\psi)\,\nabla\theta_B + G(\psi)\,\nabla\zeta_B$$

The key property of Boozer coordinates is that $I$ and $G$ are flux-surface functions (depend only on $\psi$, not on the angles). This simplifies the magnetic field strength to a pure Fourier series in $(\theta_B, \zeta_B)$ on each surface.

The Boozer angles are defined relative to the VMEC angles by:

$$\zeta_B = \zeta_0 + \nu$$

$$\theta_B = \theta_0 + \lambda + \iota\,\nu$$

The angle-shift field $\nu$ is determined from the covariant field components. The implementation uses:

$$\nu = \frac{w - I\,\lambda}{G + \iota\, I}$$

where $w$ is reconstructed from the original covariant field harmonics.

**Reference:** `stellarator_workflow.tex`, Section 4.3.

---

## Convergence & Validity

> [!TODO]
> Document convergence behavior vs. mboz/nboz resolution, known failure modes, and recommended settings for production vs. screening.

---

## API Documentation

> [!TODO]
> Document the booz_xform_jax Python API: entry points, file-based and in-memory calling conventions, and configuration options.

---

## Scripts & Workflows

**`booz_xform` (via Pixi):** From the `stages`/ directory:

```
pixi run stage-2-booz
```

Smoke-test task: verifies the `booz_xform` package is importable. A full end-to-end example for the legacy code is TBD.

**`booz_xform_jax` (via Pixi):** From the `stages`/ directory:

```
pixi run -e stage-2-booz-jax stage-2-booz
```

> [!NOTE]
> The Stage 2 JAX driver now takes explicit `--wout` and `--output` paths. Populate `outputs/quick_run/stage1_equilibrium/` by running `pixi run stage-1-vmex` first.

**Input:** `outputs/quick_run/stage1_equilibrium/wout_HSX_vacuum_ns201_quickrun.nc` (from Stage 1)
**Output:** `outputs/quick_run/stage2_boozer/boozmn_HSX_vacuum_ns201_quickrun.nc`

See `docs/mvp-pipeline.md` for full I/O details.

---

## W&B Tracking

**Project:** `driftless-star-stage2-boozer`

> [!TODO]
> Document W&B metrics, artifacts, run naming, cross-code comparison dashboards, and Stage 1 integration.

---

## Container Specification (Phase 2)

**`booz_xform` (legacy Fortran/Python):** Built from the single templated `stages/Dockerfile` using build arguments:

```
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-2-booz stages/  # CPU
```

Published to GHCR as `ghcr.io/driftless-star/driftless-star:stage-2-booz-cpu`.

**`booz_xform_jax`:** Built from the single templated `stages/Dockerfile` using build arguments:

```
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-2-booz-jax stages/                                            # CPU
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-2-booz-jax-gpu --build-arg CUDA_VERSION=12 stages/  # GPU
```

Published to GHCR as `ghcr.io/driftless-star/driftless-star:stage-2-booz-jax-cpu` and `stage-2-booz-jax-gpu`. CI builds via `.github/workflows/containers.yml`.

See [guide](../guide.md#container-architecture) for full architecture details.

---

## Tests (Phase 2)

> [!TODO]
> Document unit, regression, convergence, cross-code, integration, and round-trip tests for the Boozer transform.
> See [guide](../guide.md#writing-tests) for examples.

---

## Claude Skills

> [!TODO]
> Document dev and operational Claude Code skills for running, debugging, and validating the Boozer transform.
> See [guide](../guide.md#step-7-create-claude-skills) for skill types.
