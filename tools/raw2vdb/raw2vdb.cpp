// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

// std
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>
// openvdb
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/DenseSparseTools.h>
#include <openvdb/math/Math.h>
// ours
#include <dvr_course-common.h>
#include <vecmath.h>

using namespace dvr_course;

enum class DataType { U8, U16, F32, Unspecified, };

inline
size_t sizeInBytes(DataType type) {
  if (type == DataType::U8) return 1ull;
  else if (type == DataType::U16) return 2ull;
  else if (type == DataType::F32) return 4ull;
  else return ~0ull;
}

struct {
  struct {
    std::string fileName;
    vec3i dims{-1};
    DataType type{DataType::Unspecified};
  } input, output;
} g_appState;

static bool parseCommandLine(int argc, char **argv)
{
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "-o")
      g_appState.output.fileName = argv[++i];
    else if (arg == "-dims") {
      g_appState.input.dims.x = atoi(argv[++i]);
      g_appState.input.dims.y = atoi(argv[++i]);
      g_appState.input.dims.z = atoi(argv[++i]);
    } else if (arg == "-outdims") {
      g_appState.output.dims.x = atoi(argv[++i]);
      g_appState.output.dims.y = atoi(argv[++i]);
      g_appState.output.dims.z = atoi(argv[++i]);
    } else if (arg == "-type") {
      std::string type = argv[++i];
      if (type == "uchar")   g_appState.input.type = DataType::U8;
      if (type == "ushort")  g_appState.input.type = DataType::U16;
      if (type == "float32") g_appState.input.type = DataType::F32;
    } else if (arg == "-outtype" ) {
      std::string type = argv[++i];
      if (type == "uchar")   g_appState.output.type = DataType::U8;
      if (type == "ushort")  g_appState.output.type = DataType::U16;
      if (type == "float32") g_appState.output.type = DataType::F32;
    } else if (arg[0] != '-')
      g_appState.input.fileName = arg;
    else return false;
  }

  return true;
}

static bool validateInput()
{
  if (g_appState.input.fileName.empty())
    return false;

  if (g_appState.output.fileName.empty())
    return false;

  if (!endsWith(g_appState.output.fileName, ".vdb"))
    return false;

  if (g_appState.input.dims.x <= 0 || g_appState.input.dims.y <= 0 ||
      g_appState.input.dims.z <= 0)
    return false;

  if (g_appState.input.type == DataType::Unspecified)
    return false;

  return true;
}

inline
float getValue(int x, int y, int z, const char *input, vec3i dims, DataType type)
{
  float value = 0.f;
  if (type == DataType::U8) {
    unsigned char *inVoxels = (unsigned char *)input;
    value = inVoxels[x+y*dims.x+z*size_t(dims.x)*dims.y]/255.f;
  }
  else if (type == DataType::U16) {
    unsigned short *inVoxels = (unsigned short *)input;
    value = inVoxels[x+y*dims.x+z*size_t(dims.x)*dims.y]/65535.f;
  }
  else if (type == DataType::F32) {
    float *inVoxels = (float *)input;
    value = inVoxels[x+y*dims.x+z*size_t(dims.x)*dims.y];
  }
  return value;
}

class FloatRule
{
public:
  typedef openvdb::FloatTree            ResultTreeType;
  typedef ResultTreeType::LeafNodeType  ResultLeafNodeType;

  typedef float                                  ResultValueType;
  typedef float                                  DenseValueType;

  FloatRule(const DenseValueType &value, const DenseValueType &tolerance = DenseValueType(0.0))
    : mMaskValue(value),
      mTolerance(tolerance)
  {}

  template <typename IndexOrCoord>
  void operator()(const DenseValueType& a, const IndexOrCoord& offset,
                  ResultLeafNodeType* leaf) const
  {
    if (a <= mMaskValue-mTolerance || a >= mMaskValue+mTolerance) {
      leaf->setValueOn(offset, a);
    }
  }

private:
  const DenseValueType mMaskValue;
  const DenseValueType mTolerance;
};

template<typename VDB_T> void convertToVDB(
  const std::vector<char> &input, vec3i dims, DataType type, vec3i outdims)
{
  openvdb::initialize();
  openvdb::math::CoordBBox domain(openvdb::math::Coord(0, 0, 0),
                                  openvdb::math::Coord(dims.x, dims.y, dims.z));
  openvdb::tools::Dense<VDB_T> *dense = new openvdb::tools::Dense<VDB_T>(domain, 0.f);

  for (int z=0; z<dims.z; ++z) {
    for (int y=0; y<dims.y; ++y) {
      for (int x=0; x<dims.x; ++x) {
        vec3f xyz(x,y,z);
        xyz /= vec3f(dims-1);
        xyz *= vec3f(outdims-1);
        openvdb::math::Coord ijk(xyz.x,xyz.y,xyz.z);
        const float value = getValue(x,y,z,input.data(),dims,type);
        dense->setValue(ijk,value);
      }
    }
  }

  float backgroundValue = 0.f;
  FloatRule rule(backgroundValue);
  openvdb::FloatTree::Ptr result
      = openvdb::tools::extractSparseTree(*dense, rule, backgroundValue);
  result->prune();

  openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(result);
  grid->setName("density");

  openvdb::GridPtrVec grids;
  grids.push_back(grid);

  try {
    openvdb::io::File file(g_appState.output.fileName);
    file.write(grids);
    file.close();
    std::cout << "Output written to: " << g_appState.output.fileName << '\n';
  } catch (const openvdb::Exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  delete dense;
}

static void printUsage()
{
  std::cerr << "Usage: ./raw2vdb in.XXX -o out.vdb -dims vx vy vz "
      << "-type {uchar|ushort|float32} [-outdims vx vy vz] "
      << "[-outtype {uchar|ushort|float32}]\n";
}

int main(int argc, char **argv)
{
  if (!parseCommandLine(argc, argv)) {
    printUsage();
    exit(1);
  }

  if (!validateInput()) {
    printUsage();
    exit(1);
  }

  std::ifstream in(g_appState.input.fileName, std::ios::binary);
  if (!in.good()) {
    printUsage();
    exit(1);
  }

  if (g_appState.output.dims.x <= 0 || g_appState.output.dims.y <= 0 ||
      g_appState.output.dims.z <= 0)
    g_appState.output.dims = g_appState.input.dims;

  if (g_appState.output.type == DataType::Unspecified)
    g_appState.output.type = g_appState.input.type;

  size_t size = g_appState.input.dims.x*size_t(g_appState.input.dims.y)*
      g_appState.input.dims.z*sizeInBytes(g_appState.input.type);
  std::vector<char> input(size);
  in.read(input.data(), size);

  if (g_appState.output.type == DataType::U8)
    convertToVDB<unsigned char>(
        input, g_appState.input.dims, g_appState.input.type, g_appState.output.dims);
  else if (g_appState.output.type == DataType::U16)
    convertToVDB<unsigned short>(
        input, g_appState.input.dims, g_appState.input.type, g_appState.output.dims);
  else if (g_appState.output.type == DataType::F32)
    convertToVDB<float>(
        input, g_appState.input.dims, g_appState.input.type, g_appState.output.dims);
}



