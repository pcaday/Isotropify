# Isotropify
Any two-dimensional Riemannian metric is equivalent to an (essentially unique) conformally Euclidean metric under a suitable change of coordinates. In imaging, this means any anisotropic 2D wave speed can be made isotropic by changing coordinates.

Isotropify is a Julia implementation of a straightforward algorithm for finding an equivalent conformally Euclidean metric to a given 2D metric which is anisotropic only in the _interior_. Check out the Jupyter notebook (`TestIsotropify.jl`) for an example and pictures.
