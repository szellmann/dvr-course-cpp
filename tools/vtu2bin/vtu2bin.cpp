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
#include <fstream>
#include <string>
#include <vector>
// vtk
#include <vtkCell.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDataSetTriangleFilter.h>
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
  std::string outFileName{"out.bin"};
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

static bool parseCommandLine(int argc, char **argv)
{
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "-o")
      g_appState.outFileName = argv[++i];
    else if (arg[0] != '-')
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
  std::cerr << "Usage: ./vtu2bin in.{vtk|vtu} -o out.bin\n";
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

  vtkUnstructuredGrid *inputGrid{nullptr};
  if (reader->CanReadFile(g_appState.inFileName.c_str())) {
    reader->SetFileName(g_appState.inFileName.c_str());
    reader->Update();
    inputGrid = reader->GetOutput();
  } else {
    legacyReader->SetFileName(g_appState.inFileName.c_str());
    legacyReader->Update();
    inputGrid = legacyReader->GetOutput();
  }

  if (!inputGrid) {
    printUsage();
    std::cerr << "Error, failed to load VTK file: " << g_appState.inFileName << '\n';
    exit(1);
  }

  auto triangleFilter = vtkSmartPointer<vtkDataSetTriangleFilter>::New();
  triangleFilter->SetInputData(inputGrid);
  triangleFilter->Update();
  vtkUnstructuredGrid *grid
      = vtkUnstructuredGrid::SafeDownCast(triangleFilter->GetOutput());

  // Assemble VTK input:
  vtkIdType numPoints = grid->GetNumberOfPoints();
  vtkIdType numCells = grid->GetNumberOfCells();

  std::vector<vec3f> vertices;
  for (vtkIdType i=0; i<numPoints; ++i) {
    double *pt = grid->GetPoint(i);
    vertices.push_back({(float)pt[0],(float)pt[1],(float)pt[2]});
  }

  auto *cellData = grid->GetCellData();
  auto *pointData = grid->GetPointData();
  std::vector<float> cellValues = firstScalarArray(cellData);
  std::vector<float> vertexValues = firstScalarArray(pointData);

  std::vector<int> cellTypes;
  std::vector<int> cellIndices;
  std::vector<int> connectivity;

  for (vtkIdType i=0; i<numCells; ++i) {
    int type = grid->GetCellType(i);
    if (!isVolumeCell(type))
      continue;
    cellTypes.push_back(type);
    cellIndices.push_back((int)connectivity.size());
    vtkCell *cell = grid->GetCell(i);
    int n = cell->GetNumberOfPoints();
    for (int j=0; j<n; ++j) {
      vtkIdType index = cell->GetPointId(j);
      connectivity.push_back(index);
    }
  }

  auto saveVector = [](std::ofstream &of, auto &vec) {
    uint64_t size = vec.size();
    of.write((const char *)&size,sizeof(size));
    of.write((const char *)vec.data(),vec.size()*sizeof(vec[0]));
  };

  uint64_t numDataArrays=2;
  std::ofstream of(g_appState.outFileName,std::ios::binary);

  // vertex positions:
  saveVector(of,vertices);
  // topology
  saveVector(of,cellTypes);
  saveVector(of,cellIndices);
  saveVector(of,connectivity);
  // data arrays:
  of.write((const char *)&numDataArrays,sizeof(numDataArrays));
  saveVector(of,cellValues);
  saveVector(of,vertexValues);
}
