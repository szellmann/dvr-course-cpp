// ======================================================================== //
// Copyright 2025-2026 Stefan Zellmann                                      //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// std
#include <string>
// openvdb
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/math/Math.h>
// vtk
#include <vtkCell.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkIdList.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridReader.h>
// ours
#include <dvr_course-common.h>
#include <vecmath.h>

using namespace dvr_course;

struct {
  std::string inFileName;
  std::string outFileName{"out.vdb"};
  vec3i dims{256}; // number of voxels in VDB
} g_appState;

static bool isVolumeCell(int type)
{
  return type == VTK_TETRA || type == VTK_VOXEL || type == VTK_HEXAHEDRON
      || type == VTK_WEDGE || type == VTK_PYRAMID;
}

static std::vector<float> firstScalarArray(vtkDataSetAttributes *data)
{
  std::vector<float> result;

  for (int i=0; i<data->GetNumberOfArrays(); ++i) {
    vtkDataArray *array = data->GetArray(i);
    if (!array)
      continue;

    int nComp = array->GetNumberOfComponents();
    if (nComp != 1)
      continue;

    vtkIdType numTuples = array->GetNumberOfTuples();

    for (vtkIdType j=0; j<numTuples; ++j) {
      result.push_back(static_cast<float>(array->GetTuple1(j)));
    }
  }

  return result;
}

static vec3i projectToGrid(const vec3f V, const box3f &worldBounds, const vec3i &dims)
{
  const vec3f V01 = (V-worldBounds.lower)/worldBounds.size();
  const vec3i Vi(V01.x*dims.x,V01.y*dims.y,V01.z*dims.z);
  return clamp(Vi,vec3i(0),dims-1);
}

static void rasterize(openvdb::FloatGrid::Accessor acc,
                      const box4f &cell,
                      const box3f &worldBounds,
                      const vec3i &dims)
{
  vec3i lo = projectToGrid(cell.lower.xyz,worldBounds,dims);
  vec3i up = projectToGrid(cell.upper.xyz,worldBounds,dims);

  for (int z=lo.z; z<=up.z; ++z) {
    for (int y=lo.y; y<=up.y; ++y) {
      for (int x=lo.x; x<=up.x; ++x) {
        float t = length((vec3f(x,y,z)-vec3f(lo))/vec3f(up-lo+1))/length(vec3f(1,1,1));
        acc.setValue(openvdb::Coord(x,y,z),lerp(cell.lower.w,cell.upper.w,t));
      }
    }
  }
}

static bool parseCommandLine(int argc, char **argv)
{
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "-o")
      g_appState.outFileName = argv[++i];
    else if (arg == "-dims") {
      g_appState.dims.x = atoi(argv[++i]);
      g_appState.dims.y = atoi(argv[++i]);
      g_appState.dims.z = atoi(argv[++i]);
    } else if (arg[0] != '-')
      g_appState.inFileName = arg;
    else return false;
  }

  return true;
}

static bool validateInput()
{
  if (!(endsWith(g_appState.inFileName, ".vtk") ||
        endsWith(g_appState.inFileName, ".vtu"))) {
    return false;
  }

  return true;
}

static void printUsage()
{
  std::cerr << "Usage: ./vtu2vdb in.{vtk|vtu} -o out.vdb [-dims vx vy vz]\n";
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

  // Read VTK unstructured grid data
  auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  auto legacyReader = vtkSmartPointer<vtkUnstructuredGridReader>::New();

  vtkUnstructuredGrid *grid{nullptr};
  if (reader->CanReadFile(g_appState.inFileName.c_str())) {
    reader->SetFileName(g_appState.inFileName.c_str());
    reader->Update();
    grid = reader->GetOutput();
  } else {
    legacyReader->SetFileName(g_appState.inFileName.c_str());
    legacyReader->Update();
    grid = legacyReader->GetOutput();
  }

  if (!grid) {
    printUsage();
    std::cerr << "Error, failed to load VTK file: " << g_appState.inFileName << '\n';
    exit(1);
  }

  // Assemble VTK input:
  vtkIdType numPoints = grid->GetNumberOfPoints();
  vtkIdType numCells = grid->GetNumberOfCells();

  std::vector<vec3f> vertices;
  for (vtkIdType i = 0; i < numPoints; ++i) {
    double *pt = grid->GetPoint(i);
    vertices.push_back({(float)pt[0],(float)pt[1],(float)pt[2]});
  }

  auto *cellData = grid->GetCellData();
  auto *pointData = grid->GetPointData();
  std::vector<float> cellValues = firstScalarArray(cellData);
  std::vector<float> vertexValues = firstScalarArray(pointData);

  // Compute world bounds:
  box3f worldBounds{1e20f,-1e20f};
  for (vtkIdType i=0; i<numCells; ++i) {
    int type = grid->GetCellType(i);
    if (!isVolumeCell(type))
      continue;
    vtkCell *cell = grid->GetCell(i);
    int n = cell->GetNumberOfPoints();
    for (int j=0; j<n; ++j) {
      vtkIdType index = cell->GetPointId(j);
      worldBounds.extend(vertices[index]);
    }
  }

  vec3f origin = worldBounds.lower;
  vec3f spacing = (worldBounds.upper-worldBounds.lower)/vec3f(g_appState.dims);

  // VDB to fill:
  openvdb::initialize();

  using FloatTree = openvdb::tree::Tree4<float, 5, 4, 3>::Type;
  openvdb::FloatGrid::Ptr vdbGrid = openvdb::FloatGrid::create(0.f);
  vdbGrid->setName("density");
  openvdb::math::Mat4d matrix = openvdb::math::Mat4d::identity();
  matrix.setTranslation(openvdb::math::Vec3d(origin.x,origin.y,origin.z));
  matrix.setToScale(openvdb::math::Vec3d(spacing.x,spacing.y,spacing.z));
  auto xfm = openvdb::math::Transform::createLinearTransform(matrix);
  vdbGrid->setTransform(xfm);
  auto &tree = vdbGrid->tree();
  auto acc = vdbGrid->getAccessor();

  // Splat grid cells to vdb using cell bounding boxes
  for (vtkIdType i=0; i<numCells; ++i) {
    int type = grid->GetCellType(i);
    if (!isVolumeCell(type))
      continue;
    vtkCell *cell = grid->GetCell(i);
    int n = cell->GetNumberOfPoints();
    float cellValue = cellValues.empty() ? NAN : cellValues[i];
    box4f cellBounds{1e20f,-1e20f};
    for (int j=0; j<n; ++j) {
      vtkIdType index = cell->GetPointId(j);
      float value = vertexValues.empty() ? cellValue : vertexValues[index];
      assert(!isnan(value));
      cellBounds.extend(vec4f(vertices[index],value));
    }
    rasterize(acc,cellBounds,worldBounds,g_appState.dims);
  }

  tree.prune();

  openvdb::GridPtrVec grids;
  grids.push_back(vdbGrid);
  openvdb::io::File file(g_appState.outFileName);
  file.write(grids);
  file.close();
}
