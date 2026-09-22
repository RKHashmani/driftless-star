# Supplied equilibrium

To recover the [T3D+GX](https://t3d.readthedocs.io/en/latest/QuickGX.html) results, we will skip running Stage 1 (VMEX) and use the same [VMEC output](https://bitbucket.org/gyrokinetics/t3d/src/v0.1.0/tests/data/wout_w7x.nc) used in T3D+GX.

Run these commands from the repository root to move the NetCDF to where Stage 1 would normally generate it to skip running Stage 1.

```sh
mkdir -p outputs/w7-x/t3d_benchmark/loop/iter_1/output/stage1_equilibrium
cp inputs/w7-x/t3d_benchmark/wout_w7x_t3d_reconstruction.nc \
  outputs/w7-x/t3d_benchmark/loop/iter_1/output/stage1_equilibrium/wout_w7x_t3d_reconstruction.nc
```