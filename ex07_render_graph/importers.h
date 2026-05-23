#pragma once

// std
#include <fstream>
// nanovdb
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
// tinyobjloader
#include <tiny_obj_loader.h>
// ours
#include "Params.h"

namespace ex07_render_graph {

using namespace dvr_course;

// ========================================================
// imported data, with handle, and on-device storage
// ========================================================
struct NVDB_import {
  bool isValid{false};
  Volume volume;
  Buffer<uint8_t> onDevice;
};

struct TetMesh_import {
  bool isValid{false};
  Volume volume;
  Buffer<Tet> onDevice;
};

struct OBJ_import {
  bool isValid{false};
  TriangleMesh mesh;
  std::pair<Buffer<vec3f>,Buffer<vec3i>> onDevice;
};

// ========================================================
// loadXXX functions
// ========================================================
static NVDB_import loadNvdb(std::string filePath) {
  nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle;

  try {
    auto grid = nanovdb::io::readGrid(filePath);
    auto hostbuffer = nanovdb::HostBuffer::create(grid.bufferSize());
    std::memcpy(hostbuffer.data(), grid.data(), grid.bufferSize());
    gridHandle = std::move(hostbuffer);
  } catch (...) {
    return {false,{},{}};
  }

  auto boundsMin = gridHandle.gridMetaData()->worldBBox().min();
  auto boundsMax = gridHandle.gridMetaData()->worldBBox().max();
  box3f volbounds({(float)boundsMin[0], (float)boundsMin[1], (float)boundsMin[2]},
                  {(float)boundsMax[0], (float)boundsMax[1], (float)boundsMax[2]});

  box1f valueRange = {gridHandle.grid<float>()->tree().root().minimum(),
                      gridHandle.grid<float>()->tree().root().maximum()};
  Volume volume;
  volume.type = Volume::NVDB;
  volume.asNvdb.filterLinear = true;
  volume.bounds = volbounds;
  volume.dataRange = valueRange;

  Buffer deviceGrid(gridHandle.bufferSize(), (uint8_t *)gridHandle.data());

  return {true,volume,deviceGrid};
}

static TetMesh_import loadTetMesh(std::string filePath) {
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

  std::ifstream in(filePath);
  if (!in.good()) {
    return {false,{},{}};
  }

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
    #define VTK_TET_ 10
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

  Volume volume;
  volume.type = Volume::TET;
  volume.bounds = box3f(
    {INFINITY,INFINITY,INFINITY},
    {-INFINITY,-INFINITY,-INFINITY}
  );

  volume.dataRange = box1f(INFINITY, -INFINITY);

  for (size_t i=0; i<tets.size(); ++i) {
    volume.bounds.extend(tets[i].v0.xyz); volume.dataRange.extend(tets[i].v0.w);
    volume.bounds.extend(tets[i].v1.xyz); volume.dataRange.extend(tets[i].v1.w);
    volume.bounds.extend(tets[i].v2.xyz); volume.dataRange.extend(tets[i].v2.w);
    volume.bounds.extend(tets[i].v3.xyz); volume.dataRange.extend(tets[i].v3.w);
  }

  Buffer deviceTets(tets.size(), tets.data());

  volume.asTetMesh.tets = deviceTets.data();
  volume.asTetMesh.numTets = (int)deviceTets.size();

  return {true,volume,deviceTets};
}

static OBJ_import loadObj(std::string filePath) {
  tinyobj::ObjReaderConfig reader_config;
  tinyobj::ObjReader reader;

  if (!reader.ParseFromFile(filePath, reader_config)) {
    return {false,{},{}};
  }

  std::vector<vec3f> vertices;
  std::vector<vec3i> indices;

  auto &attrib = reader.GetAttrib();
  auto &shapes = reader.GetShapes();

  vertices.reserve(attrib.vertices.size()/3);
  for (size_t i=0; i<attrib.vertices.size(); i+=3) {
    vertices.push_back({attrib.vertices[i+0],
                        attrib.vertices[i+1],
                        attrib.vertices[i+2]});
  }

  for (const auto &s : shapes) {
    for (size_t i=0; i<s.mesh.indices.size(); i+=3) {
      indices.push_back({s.mesh.indices[i+0].vertex_index,
                         s.mesh.indices[i+1].vertex_index,
                         s.mesh.indices[i+2].vertex_index});
    }
  }

  Buffer deviceVertices(vertices.size(), vertices.data());
  Buffer deviceIndices(indices.size(), indices.data());

  TriangleMesh mesh;
  mesh.vertices = deviceVertices.data();
  mesh.indices = deviceIndices.data();

  mesh.bounds = box3f(
    {INFINITY,INFINITY,INFINITY},
    {-INFINITY,-INFINITY,-INFINITY}
  );

  for (size_t i=0; i<indices.size(); ++i) {
    mesh.bounds.extend(vertices[indices[i].x]);
    mesh.bounds.extend(vertices[indices[i].y]);
    mesh.bounds.extend(vertices[indices[i].z]);
  }

  return {true,mesh,{deviceVertices,deviceIndices}};
}

} // namespace ex07_render_graph



