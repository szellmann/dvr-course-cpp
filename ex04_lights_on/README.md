# Ex. 02: Woodcock tracking

This example replaces the ray marching loop with Woodcock tracking to also
compute absorption + emission. Appearance-wise the (converged) result is rather
similar to ordinary sci-vis ray marching. We use Woodcock in a sci-vis style
manner, i.e., the data values do not directly become the density--instead we
perform a lookup into an RGBA transfer function (as the ray marching samples
did as well) and then (in stark contrast to production style rendering)
interpret RGB as albedo and alpha as extinction coefficient.

## TODOs:
- [x] Implement CPU code path (first light)
- [x] Implement OWL code path
