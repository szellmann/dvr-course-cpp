# Ex. 07: Render Graph

The render graph example is the one where all the techniques discussed so far
come together. The scene now consists of multiple volumes and meshes, advanced
scattering effects, tet-meshes with BVH-based sampling, RT-Core acceleration,
space skipping optimizations, etc. The example is representative of that used
by production-grade, ray tracing-based, GPU sci-vis renderers.

## TODOs:
- [x] Triangle meshes
- [ ] Implement tet sampling (what do we want to convey? RTX on the leaf level?
      or only for  TLAS)
- [ ] Single scattering and AO
- [ ] Animation (?)
- [ ] ~~Volumes sit underneath a TLAS (be OptiX or cuBQL)~~ we *won't* do that;
      instead the volumes will go in a linear array that we iterate over; that
      way we can use hardware ray tracing for sampling
- [ ] Space skipping with DDA (from ex06)
- [ ] Instance transforms? Via command line? Do we need this?
