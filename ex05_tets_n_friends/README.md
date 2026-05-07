# Ex. 05: Tets N' friends

This example demonstrates implementing custom volume element types replacing
voxels. The example uses OptiX and OWL for cell location with tetrahedra cells.
The sample generally works on the CPU, but doesn't use a BVH so will be very
slow.

## TODOs:
- [ ] Implement tet sampling (with aabb geom? with triangles? With cuBQL even?)
- [ ] And friends...: maybe INR with OptiX vec-coop? Depends how much time we
      have..
