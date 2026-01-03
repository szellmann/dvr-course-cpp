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

static void printHelp() {
  std::cout << "SYNOPSIS\n\n";
  std::cout << "Convert DWD ICON data to internal format used by our tool chain.\n";
  std::cout << "Given data from the DWD, we require the appropriate \"horizontal grid file\",\n";
  std::cout << "e.g., \"icon_grid_0026_R03B07_G.nc\" from http://icon-downloads.mpimet.mpg.de/dwd_grids.xml\n";
  std::cout << "and the grid files for the variable of interest in NetCDF format.\n";
  std::cout << "Data files can, e.g., be found here: https://opendata.dwd.de/weather/nwp/icon/grib/00/\n";
  std::cout << "Files in grib2 format must first be converted to NetCDF using:\n";
  std::cout << "cdo -f nc copy <in.grib2> <out.nc>\n";
  std::cout << "We assume that certain NetCDF dims and variables are present, such as \"height\".\n";
  std::cout << "In case this these are not present this script should be adapted accordingly....\n";
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
  int cell_id;
  if ((retval = nc_inq_dimid(ncid, "cell", &cell_id)) != NC_NOERR) {
    printf("Error finding dimension: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  size_t cell;
  if ((retval = nc_inq_dimlen(ncid, cell_id, &cell)) != NC_NOERR) {
    printf("Error reading dimension: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }
  printf("number of cells: %i\n",(int)cell);

  // read triangle vertex IDs
  int vertex_of_cell_id;
  if ((retval = nc_inq_varid(ncid, "vertex_of_cell", &vertex_of_cell_id)) != NC_NOERR) {
    printf("Error finding variable: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  int *vertex_of_cell = new int[cell*3]; // these are one-based!! (Fortran......)
  if ((retval = nc_get_var_int(ncid, vertex_of_cell_id, vertex_of_cell)) != NC_NOERR) {
    printf("Error reading data: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  // read number of vertices:
  int vertex_id;
  if ((retval = nc_inq_dimid(ncid, "vertex", &vertex_id)) != NC_NOERR) {
    printf("Error finding dimension: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  size_t vertex;
  if ((retval = nc_inq_dimlen(ncid, vertex_id, &vertex)) != NC_NOERR) {
    printf("Error reading dimension: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }
  printf("number of vertices: %i\n",(int)vertex);

  // read vlon & vlat:
  int vlon_id;
  if ((retval = nc_inq_varid(ncid, "vlon", &vlon_id)) != NC_NOERR) {
    printf("Error finding dimension: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  double *vlon = new double[vertex];
  if ((retval = nc_get_var_double(ncid, vlon_id, vlon)) != NC_NOERR) {
    printf("Error reading data: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  int vlat_id;
  if ((retval = nc_inq_varid(ncid, "vlat", &vlat_id)) != NC_NOERR) {
    printf("Error finding dimension: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  double *vlat = new double[vertex];
  if ((retval = nc_get_var_double(ncid, vlat_id, vlat)) != NC_NOERR) {
    printf("Error reading data: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  // read clon_vertices & clat_vertices:
  int clon_vertices_id;
  if ((retval = nc_inq_varid(ncid, "clon_vertices", &clon_vertices_id)) != NC_NOERR) {
    printf("Error finding dimension: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  double *clon_vertices = new double[cell*3];
  if ((retval = nc_get_var_double(ncid, clon_vertices_id, clon_vertices)) != NC_NOERR) {
    printf("Error reading data: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  int clat_vertices_id;
  if ((retval = nc_inq_varid(ncid, "clat_vertices", &clat_vertices_id)) != NC_NOERR) {
    printf("Error finding dimension: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  double *clat_vertices = new double[cell*3];
  if ((retval = nc_get_var_double(ncid, clat_vertices_id, clat_vertices)) != NC_NOERR) {
    printf("Error reading data: %s\n", nc_strerror(retval));
    nc_close(ncid);
    return 1;
  }

  nc_close(ncid);

  // Data files:

  int numLayers = 0;
  std::vector<float> heights, values;
  for (int i=2; i<std::min(argc,32); ++i) {
    if ((retval = nc_open(argv[i], NC_NOWRITE, &ncid)) != NC_NOERR) {
      printf("Error opening file: %s\n", nc_strerror(retval));
      return 1;
    }

    // read number of cells:
    int ncells_id;
    if ((retval = nc_inq_dimid(ncid, "ncells", &ncells_id)) != NC_NOERR) {
      printf("Error finding dimension: %s\n", nc_strerror(retval));
      nc_close(ncid);
      return 1;
    }

    size_t ncells;
    if ((retval = nc_inq_dimlen(ncid, ncells_id, &ncells)) != NC_NOERR) {
      printf("Error reading dimension: %s\n", nc_strerror(retval));
      nc_close(ncid);
      return 1;
    }
    printf("number of cells IN DATA FILE: %i\n",(int)ncells);

    // read number height layers:
    int height_id;
    if ((retval = nc_inq_dimid(ncid, "height", &height_id)) != NC_NOERR) {
      printf("Error finding dimension: %s\n", nc_strerror(retval));
      nc_close(ncid);
      return 1;
    }

    size_t height;
    if ((retval = nc_inq_dimlen(ncid, height_id, &height)) != NC_NOERR) {
      printf("Error reading dimension: %s\n", nc_strerror(retval));
      nc_close(ncid);
      return 1;
    }
    printf("height IN DATA FILE: %i (MUST BE 1!)\n",(int)height);

    // read height as variable
    int height_as_var_id;
    if ((retval = nc_inq_varid(ncid, "height", &height_as_var_id)) != NC_NOERR) {
      printf("Error finding dimension: %s\n", nc_strerror(retval));
      nc_close(ncid);
      return 1;
    }

    double *height_var = new double[height];
    if ((retval = nc_get_var_double(ncid, height_as_var_id, height_var)) != NC_NOERR) {
      printf("Error reading data: %s\n", nc_strerror(retval));
      nc_close(ncid);
      return 1;
    }
    for (int j=0; j<height; ++j) {
      heights.push_back((float)height_var[j]);
    }
    delete[] height_var;

    const char *varname = "pres";

    // read VARIABLE
    int var_id;
    if ((retval = nc_inq_varid(ncid, varname, &var_id)) != NC_NOERR) {
      printf("Error finding dimension: %s\n", nc_strerror(retval));
      nc_close(ncid);
      return 1;
    }

    double *var = new double[ncells*height];
    if ((retval = nc_get_var_double(ncid, var_id, var)) != NC_NOERR) {
      printf("Error reading data: %s\n", nc_strerror(retval));
      nc_close(ncid);
      return 1;
    }
#if 1
    double minValue(DBL_MAX);
    double maxValue(-DBL_MAX);
    for (int j=0; j<ncells*height; ++j) {
      minValue = fmin(minValue,var[j]);
      maxValue = fmax(maxValue,var[j]);
    }
    for (int j=0; j<ncells*height; ++j) {
      var[j] -= minValue;
      var[j] /= maxValue-minValue;
    }
#endif
    for (int j=0; j<ncells*height; ++j) {
      values.push_back((float)var[j]);
    }
    delete[] var;

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



  std::ofstream out("out.ic",std::ios::binary);
  for (int i=0; i<cell; ++i) {
    float lat[3]{(float)clat_vertices[i*3],(float)clat_vertices[i*3+1],(float)clat_vertices[i*3+2]};
    float lon[3]{(float)clon_vertices[i*3],(float)clon_vertices[i*3+1],(float)clon_vertices[i*3+2]};;
    float H[32];
    assert(numLayers<sizeof(H)/sizeof(H[0]));
    H[0] = 6.371229f;
    for (int j=0; j<numLayers; ++j) {
      H[j+1] = H[j]+height_to_index[j].first/100000.f;
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
