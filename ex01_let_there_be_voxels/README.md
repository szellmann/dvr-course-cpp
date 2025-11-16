# Ex. 01: Let there be voxels..

This example extends ex. 00 with a more useful volume type using NanoVDB to
represent voxels. Course outline: create VDB volume with Blender, export to
OpenVDB, use `nanovdb_convert` to convert it to .nvdb, import in this example.

## TODOs:
- [x] Implement CPU code path (first light)
- [ ] Implement OWL code path
- [ ] Allow for passing transfer functions as files (especially with the
      non-interactive mode that won't let the user edit the TF
