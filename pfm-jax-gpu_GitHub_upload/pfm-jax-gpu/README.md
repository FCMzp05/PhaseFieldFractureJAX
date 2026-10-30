# Matrix-Free JAX Phase-Field Fracture

Standalone JAX/GPU and Julia/CPU phase-field fracture solvers.
Each solver in `code/` is a self-contained single file. Meshes are generated automatically unless `--mesh` is supplied.

## Install

```bash
python -m pip install --upgrade "jax[cuda12]==0.6.2" numpy scipy gmsh meshio matplotlib pandas
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Use Linux or WSL for NVIDIA JAX execution.

## Run

```bash
python code/elastic_fracture_SENT_jax_gpu.py --platform gpu
julia --project=. code/elastic_fracture_SENT_julia_cpu.jl --no-visualization
python code/kalthoff_winkler_dynamic_jax_gpu.py --platform gpu
```

Use `--help` for case-specific options. New results are written under `outputs/`. Frozen reference curves are stored in `data/`.
