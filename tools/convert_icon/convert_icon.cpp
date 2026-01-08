// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <netcdf.h>
#ifdef WITH_UMESH
# include "umesh/umesh.h"
#endif

struct {
  bool convertToIC{true};
  bool convertToUMesh{false};
} g_appState;

static void printHelp() {
  std::cout << "SYNOPSIS\n\n";
  std::cout << "Convert DWD ICON data to internal format used by our tool chain.\n";
  std::cout << "Given data from the DWD, we require the appropriate \"horizontal grid file\",\n";
  std::cout << "e.g., \"icon_grid_0026_R03B07_G.nc\" from http://icon-downloads.mpimet.mpg.de/dwd_grids.xml,\n";
  std::cout << "the time-invariant grid containing HSURF, e.g.:\n";
  std::cout << "\"icon_global_icosahedral_time-invariant_2026010300_HSURF.nc\",";
  std::cout << "the level height grids containing HHL from here:\n";
  std::cout << "https://opendata.dwd.de/weather/nwp/icon/grib/00/hhl/\n";
  std::cout << "and the grid files for the variable of interest in NetCDF format.\n";
  std::cout << "Data files can, e.g., be found here: https://opendata.dwd.de/weather/nwp/icon/grib/00/\n";
  std::cout << "Files in grib2 format must first be converted to NetCDF using:\n";
  std::cout << "cdo -f nc copy <in.grib2> <out.nc>\n";
  std::cout << "We assume that certain NetCDF dims and variables are present, such as \"height\".\n";
  std::cout << "In case this these are not present this script should be adapted accordingly....\n";
}

inline umesh::vec3f toCartesian(const umesh::vec3f spherical)
{
  const float r = spherical.x;
  const float lat = spherical.y;
  const float lon = spherical.z;

  float x = r * cosf(lat) * cosf(lon);
  float y = r * cosf(lat) * sinf(lon);
  float z = r * sinf(lat);
  return {x,y,z};
}

static size_t readDimLength(int ncid, std::string name) {
  int retval, dimid;
  if ((retval != nc_inq_dimid(ncid, name.c_str(), &dimid)) != NC_NOERR) {
    fprintf(stderr, "dim %s not found\n", name.c_str());
    return ~0ull;
  }

  size_t result;
  if ((retval = nc_inq_dimlen(ncid, dimid, &result)) != NC_NOERR) {
    fprintf(stderr, "variable %s found but size mismatch\n", name.c_str());
    return ~0ull;
  }

  return result;
}

static std::vector<int> readIntVar(int ncid, std::string name, size_t len) {
  int retval, varid;
  if ((retval = nc_inq_varid(ncid, name.c_str(), &varid)) != NC_NOERR) {
    fprintf(stderr, "variable %s not found\n", name.c_str());
    return {};
  }

  std::vector<int> result(len);

  if ((retval = nc_get_var_int(ncid, varid, result.data())) != NC_NOERR) {
    fprintf(stderr, "cannot read from variable %s\n", name.c_str());
    return {};
  }

  if (result.size() != len) {
    fprintf(stderr, "variable %s found but size mismatch\n", name.c_str());
    return {};
  }

  return result;
}

static std::vector<double> readDoubleVar(int ncid, std::string name, size_t len) {
  int retval, varid;
  if ((retval = nc_inq_varid(ncid, name.c_str(), &varid)) != NC_NOERR) {
    fprintf(stderr, "variable %s not found\n", name.c_str());
    return {};
  }

  std::vector<double> result(len);

  if ((retval = nc_get_var_double(ncid, varid, result.data())) != NC_NOERR) {
    fprintf(stderr, "cannot read from variable %s\n", name.c_str());
    return {};
  }

  if (result.size() != len) {
    fprintf(stderr, "variable %s found but size mismatch\n", name.c_str());
    return {};
  }

  return result;
}



int main(int argc, char *argv[]) {
  if (argc < 3 || std::string(argv[1]) == "help" ) {
    printHelp();
    return 1;
  }

  int ncid, retval;
  if ((retval = nc_open(argv[1], NC_NOWRITE, &ncid)) != NC_NOERR) {
    printf("Error opening file: %s\n", nc_strerror(retval));
    return 1;
  }

  // read number of cells:
  size_t cell = readDimLength(ncid, "cell");
  printf("number of cells: %i\n",(int)cell);

  // read number of vertices:
  size_t vertex = readDimLength(ncid, "vertex");
  printf("number of vertices: %i\n",(int)vertex);

  // read clon_vertices & clat_vertices:

  auto clon_vertices = readDoubleVar(ncid, "clon_vertices", cell*3);
  auto clat_vertices = readDoubleVar(ncid, "clat_vertices", cell*3);

  nc_close(ncid);

  if (clon_vertices.empty() || clat_vertices.empty()) {
    fprintf(stderr, "%s\n", "Cannot proceed as lon/lat coordinates missing");
    nc_close(ncid);
    return 1;
  }

  // Data files:

  int numLayers = 0;
  std::vector<float> heights, values;
  for (int i=2; i<argc; ++i) {
    if ((retval = nc_open(argv[i], NC_NOWRITE, &ncid)) != NC_NOERR) {
      printf("Error opening file: %s\n", nc_strerror(retval));
      return 1;
    }

    // read number of cells:
    size_t ncells = readDimLength(ncid, "ncells");
    printf("number of cells IN DATA FILE: %i\n",(int)ncells);

    auto height = readDoubleVar(ncid, "height", 1);
    if (height.empty()) {
      fprintf(stderr, "No height found in %s, aborting...\n", argv[i]);
      nc_close(ncid);
      return 1;
    }
    heights.push_back(height[0]);

    // read VARIABLE
    const char *varname = "pres";
    auto var = readDoubleVar(ncid, varname, ncells);
    if (var.empty()) {
      fprintf(stderr, "Error reading variable %s, error: %s\n",
              varname, nc_strerror(retval));
      nc_close(ncid);
      return 1;
    }

#if 1
    double minValue(DBL_MAX);
    double maxValue(-DBL_MAX);
    for (int j=0; j<ncells; ++j) {
      minValue = fmin(minValue,var[j]);
      maxValue = fmax(maxValue,var[j]);
    }
    for (int j=0; j<ncells; ++j) {
      var[j] -= minValue;
      var[j] /= maxValue-minValue;
    }
#endif
    for (int j=0; j<ncells; ++j) {
      values.push_back((float)var[j]);
    }

    numLayers++;
  }

  printf("num layers: %i\n",numLayers);

  std::vector<std::pair<float,int>> height_to_index;
  for (int i=0; i<numLayers; ++i) {
    height_to_index.push_back({heights[i],i});
  }
  std::sort(height_to_index.begin(),height_to_index.end(),
    [](auto p1, auto p2) { return p1.first < p2.first; });
  for (int i=0; i<numLayers; ++i) {
    std::cout << height_to_index[i].second << '\n';
  }


  if (numLayers > 30) {
    std::cerr << "Only loading the first 30 layers..\n";
    numLayers = 30;
  }

  if (g_appState.convertToIC) {
    std::ofstream out("out.ic",std::ios::binary);
    for (int i=0; i<cell; ++i) {
      float lat[3]{(float)clat_vertices[i*3],(float)clat_vertices[i*3+1],(float)clat_vertices[i*3+2]};
      float lon[3]{(float)clon_vertices[i*3],(float)clon_vertices[i*3+1],(float)clon_vertices[i*3+2]};;
      float H[32];
      H[0] = 6.371229f;
      for (int j=0; j<numLayers; ++j) {
        //H[j+1] = H[j]+height_to_index[j].first/100000.f;
        H[j+1] = H[j]+height_to_index[j].first/1000.f;
      }
      float value[32];
      for (int j=0; j<numLayers; ++j) {
        int h = height_to_index[j].second;
        if (i==0) std::cout << "layer " << j << " is at index " << h << ", height value is " << height_to_index[j].first << '\n';
        value[j] = values[h*cell+i];
      }
      out.write((const char *)lat,sizeof(lat));
      out.write((const char *)lon,sizeof(lon));
      out.write((const char *)&numLayers,sizeof(numLayers));
      out.write((const char *)H,sizeof(H));
      out.write((const char *)value,sizeof(value));
    }
    out.close();
  }

  if (g_appState.convertToUMesh) {
    using namespace umesh;
    auto output = std::make_shared<UMesh>();
    output->perVertex = std::make_shared<Attribute>();
    for (int i=0; i<cell; ++i) {
      float lat[3]{(float)clat_vertices[i*3],(float)clat_vertices[i*3+1],(float)clat_vertices[i*3+2]};
      float lon[3]{(float)clon_vertices[i*3],(float)clon_vertices[i*3+1],(float)clon_vertices[i*3+2]};;
      //float H[32];
      //H[0] = 6.371229f;
      //for (int j=0; j<numLayers; ++j) {
      //  H[j+1] = H[j]+height_to_index[j].first/100000.f;
      //}
      //float value[32];
      float h1 = 6.371229f;
      for (int j=0; j<numLayers; ++j) {
        float h2 = h1 + height_to_index[j].first/1000.f;

        int h = height_to_index[j].second;

        // bottom triangle vertices
        vec3f bv1 = toCartesian({h1,lat[0],lon[0]});
        vec3f bv2 = toCartesian({h1,lat[1],lon[1]});
        vec3f bv3 = toCartesian({h1,lat[2],lon[2]});
        // bottom value
        float bot = values[h*cell+i]; // TODO: interpolate

        // top triangle vertices
        vec3f tv1 = toCartesian({h2,lat[0],lon[0]});
        vec3f tv2 = toCartesian({h2,lat[1],lon[1]});
        vec3f tv3 = toCartesian({h2,lat[2],lon[2]});
        float top = values[h*cell+i]; // TODO: interpolate

        output->vertices.push_back(bv1); output->perVertex->values.push_back(bot);
        output->vertices.push_back(bv2); output->perVertex->values.push_back(bot);
        output->vertices.push_back(bv3); output->perVertex->values.push_back(bot);
        output->vertices.push_back(tv1); output->perVertex->values.push_back(top);
        output->vertices.push_back(tv2); output->perVertex->values.push_back(top);
        output->vertices.push_back(tv3); output->perVertex->values.push_back(top);

        UMesh::Wedge wedge;
        wedge[0] = (int)output->vertices.size()-6;
        wedge[1] = (int)output->vertices.size()-5;
        wedge[2] = (int)output->vertices.size()-4;
        wedge[3] = (int)output->vertices.size()-3;
        wedge[4] = (int)output->vertices.size()-2;
        wedge[5] = (int)output->vertices.size()-1;

        output->wedges.push_back(wedge);

        h1 = h2;
      }
    }

    output->finalize();
    std::cout << output->vertices.size() << '\n';
    std::cout << output->wedges.size() << '\n';
    output->saveTo("out.umesh");
  }
}
