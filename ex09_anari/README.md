# Ex. 09: ANARI

This example provides an extensible recipe to integrate the volume renderer
developed during this course into a production app system. This is accomplished
using [ANARI](https://registry.khronos.org/ANARI/specs/1.0/ANARI-1.0.html). The
sample is intended convey both how to reorganize the volume renderer into a
library (the "ANARI device") that can be used by ANARI apps such as VTK or
ParaView, and also provides a very simple app itself to, both to make the
sample self-contained, and to show what the application side looks like. To
compile and run this example, the [ANARI
SDK](https://github.com/KhronosGroup/ANARI-SDK) must be installed.

The example is organized into a separate library and executable:

## ANARI device
The device implements the volume rendering logic. Other ANARI libraries exist
(e.g., [VisRTX](https://github.com/NVIDIA/VisRTX) or
[Barney](https://github.com/NVIDIA/barney)) that expose different feature
sets. The core component of those libraries is the "device", which is loaded by
an ANARI app using dynamic linking. The device presented here is by intention
very simplistic.

Implementing ANARI devices involves writing lots of boiler plate, much of which
is moved to different files. The relevant CPU/host part, as in the other
examples, is defined in the file
[hostCode.cpp](/ex09_anari/device/hostCode.cpp), while the relevant GPU/device
code resides in [deviceCode.cu](/ex09_anari/device/deviceCode.cu). The
remaining compilation units contain mostly boiler plate.

The device was tested with VTK and ParaView; at the time of writing a
development branch of VTK must be used so the VTK-Anari unstructured mesh
mapper becomes available.

## ANARI app
The ANARI app is, by intention, very simplistic. It can only load and render
tetrahedral meshes as in prior examples, and for that requires an ANARI device
that supports the ANARI extension `KHR_SPATIAL_FIELD_UNSTRUCTURED` added with
spec v1.1. The device included here supports this extension; so do, e.g.,
[Barney](https://github.com/NVIDIA/barney) or
[Visionaray](https://github.com/szellmann/anari-visionaray). When calling the
app without specific environment variables, the example device (`ex09_anari`, e.g.,
`libanari_library_ex09_anari.so`) will be used. Other devices can be used by
setting the environment variable `ANARI_LIBRARY` accordingly.

## TODOs:
- [ ] Code cleanup, lots of dangling comments, the code would benefit from some
      refactoring, and potential boiler plate moved to a place where it
      obfuscates what is really important
- [ ] Comment the host code
- [ ] Fix AO and shadows
- [ ] Maybe add triangle geometry back in?
- [ ] Add AO/lighting and unit distance to ImGui
- [x] SZ: Test in VTK and ParaView, should work using  my umesh mapper, but I
      haven't actually tested it yet (this works, draft MR for umesh mapper
      created)
