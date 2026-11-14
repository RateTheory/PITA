# PITA: Polynomial Interpolative Tensor Approximation

PITA estimates mass-weighted, unprojected Hessians along a reaction path from a small number of explicitly computed Hessians. You can then use the recovered Hessians in variational transition state theory with multidimensional tunneling (VTST-MT), bypassing the need to compute full Hessian for every geometry along the path.

## Repository 

| File | Purpose |
| --- | --- |
| `PITA_slices.py` | Main PITA algorithm. Reconstructs Hessians along the reaction path. |
| `RateConstantCalculator.py` | Computes tunneling coefficients from a PITA-recovered `.pkl` file. |
| `SCT.py` | Contains supporting functions for `RateConstantCalculator.py`. |

## Input data

`PITA_slices.py` reads the ground truth `.pkl` files. Each `.pkl` file should contain the following quantities along the converged intrinsic reaction coordinate (IRC) *s*:

- Electronic energy
- Eigenvalues of reaction-path projected Hessian
- Eigenvectors of reaction-path projected Hessian
- Mass-weighted gradients along IRC
- Unprojected, mass-weighted Hessians


## Usage

1. **Reconstruct the Hessians.** Run `PITA_slices.py` on the input `.pkl` file:

   ```bash
   python PITA_slices.py
   ```

   This writes a recovered `*PITA.pkl` file.

2. **Compute the tunneling coefficients.** Run `RateConstantCalculator.py` on the recovered file:

   ```bash
   python RateConstantCalculator.py
   ```

## Requirements

numpy, scipy, pandas, tensorly


## Citation

If you use PITA, please cite:

<!-- Add the reference and DOI once the paper is published -->

## License

<!-- Add a license, e.g. MIT -->
