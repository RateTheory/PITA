Polynomial Interpolative Tensor Approximation (PITA) is designed to estimate mass-weighted unprojected Hessians along the reaction paths.

The main algorithm is contained in the PITA_slices file.
The PITA_slices.py file operates on the .pkl files containing Eigenvalues, Eigenvectors, Gradient, and unprojected mass-weighted Hessians along reaction path s.

Upon obtaining a recovered *PITA.pkl file, execute RateConstantCalculator.py to estimate tunneling coefficients.
