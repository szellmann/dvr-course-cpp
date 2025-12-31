# Ex. 05: Hey ICON..

This example demonstrates implementing a more complex volume element type than
voxels, namely the icosahedron shape common in climate and weather models such
as ICON. The example uses OptiX and OWL for cell location; also, a very
simplistic traversal accelerator for Woodcock tracking ICON data inside a
spherical shell is implemented, in lieu of the more general traversal
structures discussed in later chapters. The sample generally works on the CPU,
but doesn't use a BVH so will be very slow.

## TODOs:
- [ ] No _known_ TODOs at this point (remove this comment at the end..)
