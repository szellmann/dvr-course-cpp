#pragma once

// common
#include <vecmath.h>

using namespace vecmath;

namespace ex09_anari {

struct Tet { vec4f v0, v1, v2, v3; };

static std::vector<Tet> loadTets(std::ifstream &in) {
  // TODO: error checking for data loader we assume the data to be in the exact
  // format below, but the file format has builtin support for an arbitrary
  // number of data arrays. We currently just assume there are at least two
  // scalar data arrays, and that dataArray[0] is per-cell and dataArra[1] is
  // per-vertex

  auto loadVector = [](std::ifstream &in, auto &vec) {
    uint64_t size;
    in.read((char *)&size,sizeof(size));
    vec.resize(size);
    in.read((char *)vec.data(),vec.size()*sizeof(vec[0]));
  };

  uint64_t numDataArrays=0;
  std::vector<vec3f> vertices;
  std::vector<int> cellTypes;
  std::vector<int> cellIndices;
  std::vector<int> connectivity;
  std::vector<float> cellValues, vertexValues;

  // vertex positions:
  loadVector(in,vertices);
  // topology:
  loadVector(in,cellTypes);
  loadVector(in,cellIndices);
  loadVector(in,connectivity);
  // data arrays:
  in.read((char *)&numDataArrays,sizeof(numDataArrays));
  loadVector(in,cellValues);
  loadVector(in,vertexValues);

  // assemble tets:
  std::vector<Tet> tets;
  for (size_t i=0; i<cellTypes.size(); ++i) {
    constexpr int VTK_TET_ = 10;
    if (cellTypes[i] != VTK_TET_) continue;
    int i0 = connectivity[cellIndices[i]];
    int i1 = connectivity[cellIndices[i]+1];
    int i2 = connectivity[cellIndices[i]+2];
    int i3 = connectivity[cellIndices[i]+3];
    vec3f v0 = vertices[i0];
    vec3f v1 = vertices[i1];
    vec3f v2 = vertices[i2];
    vec3f v3 = vertices[i3];
    float s0, s1, s2, s3;
    if (!vertexValues.empty()) {
      s0 = vertexValues[i0];
      s1 = vertexValues[i1];
      s2 = vertexValues[i2];
      s3 = vertexValues[i3];
    } else {
      s0 = cellValues[cellIndices[i]/4];
      s1 = cellValues[cellIndices[i]/4];
      s2 = cellValues[cellIndices[i]/4];
      s3 = cellValues[cellIndices[i]/4];
    }

    // Store tets in our simple, flattened format, i.e., values are encoded in
    // the 'w' coordinate of the positional vectors
    Tet tet;
    tet.v0 = vec4f(v0,s0);
    tet.v1 = vec4f(v1,s1);
    tet.v2 = vec4f(v2,s2);
    tet.v3 = vec4f(v3,s3);
    tets.push_back(tet);
  }
  return tets;
}

} // ex09_anari



