# Run the feedback loop

In this tutorial, you will launch the bundled CPU case through the feedback loop, find the records for each iteration, and read the signal that controls whether another pass runs. You will also see how the evolved plasma profiles become the inputs to the next iteration.

## Before you start

Follow the [README quick start](../../README.md#quick-start) to install Pixi and Docker and obtain the repository. Run the commands below from the repository root with Docker running. You can skip the standalone forward-pass command. The loop starts its own first pass from `inputs/quick_run/`.

Use a checkout where you have not already run this case in loop mode. The driver begins at iteration 1 on each invocation, so a later invocation is not an instruction to resume after the highest existing iteration. For another experiment, [create a separate run directory](../mvp-pipeline.md#how-to-run) with its own output path.

## 1. Understand what repeats

A **forward pass** computes the magnetic equilibrium, transforms its geometry into Boozer coordinates, calculates neoclassical and turbulent fluxes in parallel, and uses those fluxes to evolve the plasma profiles. Stage 5 can take many numerical time steps inside this single pass. The magnetic equilibrium and externally calculated fluxes are not recomputed after every such time step.

The standalone `driftless-star-fwd` command stops when Stage 5 produces `transport_solution.h5`. It does not run the pressure-feedback or convergence-signal steps.

A **loop iteration** performs that forward pass and adds post-processing. Post-processing prepares the next equilibrium pressure input, writes the evolved kinetic profiles, and checks the stop condition. When another iteration runs, it recomputes the equilibrium and transport calculations using the feedback. The bundled case has all five stages set to rerun.

The boundary shape stays fixed. What changes between iterations is the pressure supplied to the equilibrium solver and the plasma profiles supplied to the transport calculations. The transport clock advances from the previous solution's final time.

## 2. Launch the loop

```bash
pixi run driftless-star --config inputs/quick_run/config.yaml --max-iters 3 --cores 4
```

The loop uses `--config`. The standalone forward-pass command uses `--configfile`.

Here, `--cores 4` sets the execution concurrency passed to Snakemake. `--max-iters 3` allows up to three loop iterations. It does not set the number of Stage 5 time steps or require all three iterations to run.

Look for a log message containing `Iteration 1 of 3`. Stage logs will follow, and parallel jobs may interleave their output. The first run can include container downloads and compilation. A later `Iteration 2 of 3` message means the first iteration requested another pass.

The loop writes its own results under `outputs/quick_run/loop/`. It does not take a previous standalone result from `outputs/quick_run/stage5_transport/` as its starting state.

## 3. Find the first iteration

After the command finishes, list the iteration's output directories.

```bash
ls outputs/quick_run/loop/iter_1/output
```

A completed iteration has directories for the five stages and `stage5_post_processing`. Its main files are listed below.

| File relative to `outputs/quick_run/loop/iter_1/` | What it records |
| --- | --- |
| `input/effective_config.yaml` | Run settings including command-line overrides and this iteration's paths |
| `output/stage5_transport/transport_solution.h5` | Evolved density, temperature, pressure, electric field, and transport clock |
| `output/stage5_post_processing/converge_status.json` | The signal used to decide whether another iteration runs |

Inspect the post-processing directory to see the pressure and profile feedback files beside the signal.

```bash
ls outputs/quick_run/loop/iter_1/output/stage5_post_processing
```

If the simulation fails before post-processing finishes, the signal may be absent. Read the failing stage's log before interpreting that iteration as complete.

## 4. Read the stop signal

Use Python from the existing Pixi environment to display the first iteration's signal.

```bash
pixi run -e pipeline python -m json.tool outputs/quick_run/loop/iter_1/output/stage5_post_processing/converge_status.json
```

The JSON object contains a `status` field. Its value determines the next action.

| Status | Meaning |
| --- | --- |
| `continue` | Another iteration may run, provided the iteration limit has not been reached. |
| `converged` | The pressure-change criterion is satisfied. The loop stops. |
| `horizon` | The transport clock reached the configured end time. The loop stops even if pressure has not converged. |
| `halted` | The total pressure is non-positive at a checked radial point. The loop stops and the initial conditions need investigation. |

The bundled case checks the maximum pointwise relative change in total pressure between the initial and final time slices of an iteration. It requires that change to be below `1.0e-2`. This is the configured pressure criterion, rather than a guarantee that every aspect of the simulation is physically converged.

The same case sets `t_final = 1.4e-6` seconds. It can reach this short transport horizon in its first iteration. A single completed iteration is therefore a valid outcome even with `--max-iters 3`.

At the end, the driver logs `Loop finished after` followed by the number of iterations. If more than one completed, replace `iter_1` in the inspection commands with the last completed iteration. Its signal explains the final outcome. If the last status is `continue` and three iterations completed, the iteration cap stopped this invocation. Increasing the cap alone does not extend `t_final`.

## 5. Follow the feedback into the next pass

If a second iteration ran, inspect its inputs.

```bash
ls outputs/quick_run/loop/iter_2/input
```

The equilibrium input now contains the pressure fit from iteration 1. The copied `common_input.toml` contains prescribed density, temperature, and electric-field profiles from that iteration's final state, together with the advanced transport clock. The generated `loop_overrides.yaml` tells the neoclassical and turbulence stages to use those prescribed profiles.

Each later iteration has the same `input/` and `output/` layout. The original files in `inputs/quick_run/` remain the starting case. If the loop stopped after its first iteration, there is no second-iteration directory to inspect.

You have now launched the loop and identified its result and stop condition. For other cases, use the [loop reference](../mvp-pipeline.md#closing-the-loop) to choose the pressure criterion, transport horizon, iteration limit, or stage-reuse settings.
