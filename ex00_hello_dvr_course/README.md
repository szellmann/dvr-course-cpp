# Ex. 00: Hello DVR Course!

This example shows a simple marcher and a single homogeneous medium. This is
the "status quo" in sci-vis style volume rendering from the time of the
original [real-time volume graphics Eurographics course][1], yet using a GPGPU
programming model instead of raster hardware or shaders.

## TODOs:
- [x] Lay out simple host/device architecture, marcher in ray gen prog
- [x] Implement CPU code path
- [ ] Implement OWL code path
- [ ] Common: allow for building with an interactive pipeline (optional)
      that has a viewing window and ImGui TFE
- [x] Common: named pipelines, used as basename for PNGs
- [ ] Common: allow setting the number of convergence frames on the
      pipeline object
- [x] Ray entry point scrambling using random offsets


[1]:   http://www.real-time-volume-graphics.org/?page_id=28
