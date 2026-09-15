<p align="center">
  <img src="docs/_static/driftless-star.png" alt="Driftless Star logo, a river winding around a stellarator-shaped landscape" width="300">
</p>

# Driftless Star

Driftless Star is an open-source workflow that computes transport-consistent plasma profiles for a given stellarator boundary and plasma conditions in a unified, reproducible pipeline. It takes in a plasma pressure profile and boundary shape for the magnetic confinement and returns the generated power output. Along the way, it determines the nested flux surfaces needed to achieve magnetic equilibrium, calculates the heat and particle fluxes between the nested surfaces, and uses the fluxes to evolve the pressure profiles which can be used to repeat the loop until the pressure and equilibrium is consistent with each other.

The pipeline also enables AI-accelerated fusion device design by allowing neural surrogates to replace costly physics stages and large language model agents to orchestrate the chain, make judgment calls between stages, and autonomously search for improved geometries.

## How it works

We use [Snakemake](https://snakemake.github.io/) to turn a suite of complicated, individual software into an easily runnable set of software connected by a directed acyclic graph. 
[![Horizontal workflow with the supplied figure’s wording, showing inputs, equilibrium, orbit confinement, neoclassical and turbulent transport, global transport, and pressure feedback.](docs/_static/workflow.svg)](docs/_static/workflow.svg)

[Pixi](https://pixi.prefix.dev/) prepares the workflow environment, [Docker](https://www.docker.com/) downloads the [physics containers](https://github.com/driftless-star/driftless-star/pkgs/container/driftless-star) as needed, and each software runs inside it's own docker container. Driftless Star also supports [Apptainer](https://apptainer.org/) as an alternative to Docker.

## Quick start

1. Install [Pixi](https://pixi.prefix.dev/latest/installation/).
2. Install [Docker Engine for Linux](https://docs.docker.com/engine/install/) or [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/) (or alternatively, [Apptainer](https://apptainer.org/docs/user/main/quick_start.html)).
3. With Docker running:

```bash
git clone https://github.com/driftless-star/driftless-star.git
cd driftless-star
```

For a single forward run:

```bash
pixi run driftless-star-fwd --configfile inputs/quick_run/config.yaml --cores 4
```

It computes geometry and fluxes, evolves the profiles once, and stops after Stage 5, giving you the power output.

To loop until convergence (or the max number of iterations):

```bash
pixi run driftless-star --config inputs/quick_run/config.yaml --max-iters 3 --cores 4
```

The updated pressure profiles is fed back into Stage 1 to restart the loop for another iteration until a stop is triggered. Each iteration writes to `outputs/quick_run/loop/iter_N/`.

Results are saved under `outputs/quick_run/`. The main transport result is `outputs/quick_run/stage5_transport/transport_solution.h5`, containing the evolved density, temperature, radial electric-field profiles, and computed net heating power. The quick_run case uses reduced resolution to introduce the workflow.
### Run with Apptainer

Install [Apptainer](https://apptainer.org/docs/user/main/quick_start.html) and make sure the `apptainer` command is available on your `PATH`.

Add this top-level setting to `inputs/quick_run/config.yaml`, then you can use either of the commands above:

```yaml
container_runtime: apptainer
```

You can also select Apptainer as a command line argument without editing the config:

```bash
pixi run driftless-star-fwd --configfile inputs/quick_run/config.yaml --config container_runtime=apptainer --cores 4
```

```bash
pixi run driftless-star --config inputs/quick_run/config.yaml --container-runtime apptainer --max-iters 3 --cores 4
```

### Run the W7-X case

To replicate the plasma profile evolution from the [T3D+GX example](https://t3d.readthedocs.io/en/latest/QuickGX.html), we have an input file that uses identical initial profiles.

For a single forward run, use

```bash
pixi run driftless-star-fwd --configfile inputs/w7-x_t3d_validation/config.yaml --cores 4
```

For the feedback loop, use

```bash
pixi run driftless-star --config inputs/w7-x_t3d_validation/config.yaml --max-iters 3 --cores 4
```


## Currently Supported Software

Driftless Star supports modularity, allowing different software to be containerized and used as replacements for individual stages. We currently support these software but are expanding rapidly. If you would like a specific software to be supported and integrated, please contact us or create a feature request issue.

| Stage | What it does           | Software                                                     |
| ----- | ---------------------- | ------------------------------------------------------------ |
| 1     | Magnetic Equilibrium   | [VMEX](https://github.com/uwplasma/vmex)                     |
| 2     | Boozer Transform       | [booz_xform_jax](https://github.com/uwplasma/booz_xform_jax) |
| 3     | Neoclassical Transport | [DKX](https://github.com/uwplasma/DKX)                       |
| 4     | Turbulent Transport    | [GKX](https://github.com/uwplasma/GKX)                       |
| 5     | Global Transport       | [NEOPAX](https://github.com/uwplasma/NEOPAX)                 |


## Learn more

The project is under active development. For help, bug reports, or feature requests, open an issue in the [public repository](https://github.com/driftless-star/driftless-star/issues). Please include your command, configuration, and relevant logs when reporting a failed run.

## Authors

<p align="left">
  R. K. Hashmani<sup>1</sup>, E. Neto<sup>1</sup>, Y. Zhang<sup>2</sup>, M. Feickert<sup>3</sup>, A. Wright<sup>4</sup>, S. Wright<sup>2</sup>, Q. Li<sup>5</sup>, M. Khodak<sup>2</sup>, K. Cranmer<sup>1</sup>, R. Jorge<sup>1</sup>
</p>
<p align="left">
  <sup>1</sup> Department of Physics, University of Wisconsin-Madison, WI, USA<br>
  <sup>2</sup> Department of Computer Sciences, University of Wisconsin-Madison, WI, USA<br>
  <sup>3</sup> Data Science Institute, University of Wisconsin-Madison, WI, USA<br>
  <sup>4</sup> Department of Nuclear Engineering &amp; Engineering Physics, University of Wisconsin-Madison, WI, USA<br>
  <sup>5</sup> Department of Mathematics, University of Wisconsin-Madison, WI, USA
</p>

## License

Driftless Star is available under the [MIT license](LICENSE).
