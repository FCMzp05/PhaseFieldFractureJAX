# Reference data

Frozen QoI used to validate the Python/JAX solvers.

| Path | Meaning |
|---|---|
| `julia_reference/elastic_fracture_sent/` | Elastic SENT `F–u` (spectral, η=1e-6) |
| `julia_reference/plastic_fatigue_sent/` | Julia gold SENT elastoplastic fatigue |
| `julia_reference/plastic_fatigue_sent_l0_0p018/` | SENT fatigue length-scale sample |
| `julia_reference/plastic_fatigue_ct_cycle_jump/` | CT cycle-jump sweep summary |
| `julia_reference/plastic_sent_mesh_convergence/` | Mesh-ratio sweep summary |
| `python_cpu/sent_plastic_fatigue/` | NumPy/SciPy FE port outputs |

Do not overwrite these folders with new solver runs. Write new results under `outputs/`.
