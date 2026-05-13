// std
#include <fstream>

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
// anari-math
#include <anari/anari_cpp/ext/linalg.h>

// Header with common resources; .h: host, .cuh: device
#include <dvr_course-common.h>

#include "importers.h"


using box3_t = std::array<anari::math::float3, 2>;
namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(box3_t, ANARI_FLOAT32_BOX3);
ANARI_TYPEFOR_DEFINITION(box3_t);
} // namespace anari

using namespace dvr_course;

struct {
  std::string filepath;
  Transfunc transfunc;
  float unitDistance{1.f};
  anari::SpatialField field{nullptr};
  anari::Volume volume{nullptr};
} g_appState;

namespace ex09_anari {

// ========================================================
// Log ANARI errors
// ========================================================
static void statusFunc(const void * /*userData*/,
    ANARIDevice /*device*/,
    ANARIObject source,
    ANARIDataType /*sourceType*/,
    ANARIStatusSeverity severity,
    ANARIStatusCode /*code*/,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
  } else if (severity == ANARI_SEVERITY_WARNING) {
    fprintf(stderr, "[WARN ][%p] %s\n", source, message);
  } else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING) {
    fprintf(stderr, "[PERF ][%p] %s\n", source, message);
  }
  // Ignore INFO/DEBUG messages
}

static void updateAnariTransfunc(anari::Device device)
{
  std::vector<anari::math::float3> colors;
  std::vector<float> opacities;

  std::vector<vec4f> rgbaLUT = g_appState.transfunc.getLUT();
  for (auto rgba: rgbaLUT) {
    colors.emplace_back(rgba.r, rgba.g, rgba.b);
    opacities.emplace_back(rgba.a);
  }

  anari::setAndReleaseParameter(device,
      g_appState.volume,
      "color",
      anari::newArray1D(device, colors.data(), colors.size()));
  anari::setAndReleaseParameter(device,
      g_appState.volume,
      "opacity",
      anari::newArray1D(device, opacities.data(), opacities.size()));
  anari::math::float2 voxelRange(g_appState.transfunc.valueRange.lower,
                                 g_appState.transfunc.valueRange.upper);
  anariSetParameter(
      device, g_appState.volume, "valueRange", ANARI_FLOAT32_BOX1, &voxelRange);
  anariSetParameter(
      device, g_appState.volume, "unitDistance", ANARI_FLOAT32, &g_appState.unitDistance);

  anari::commitParameters(device, g_appState.volume);
}

static anari::World generateScene(anari::Device device)
{
  // Load input:
  std::ifstream in(g_appState.filepath);
  if (!in.good()) {
    return nullptr;
  }

  auto loadVector = [](std::ifstream &in, auto &vec) {
    uint64_t size;
    in.read((char *)&size,sizeof(size));
    vec.resize(size);
    in.read((char *)vec.data(),vec.size()*sizeof(vec[0]));
  };

  struct {
    uint64_t numDataArrays=0;
    std::vector<vec3f> vertices;
    std::vector<int> cellTypes;
    std::vector<int> cellIndices;
    std::vector<int> connectivity;
    std::vector<float> cellValues, vertexValues;
  } input;

  // vertex positions:
  loadVector(in,input.vertices);
  // topology:
  loadVector(in,input.cellTypes);
  loadVector(in,input.cellIndices);
  loadVector(in,input.connectivity);
  // data arrays:
  in.read((char *)&input.numDataArrays,sizeof(input.numDataArrays));
  loadVector(in,input.cellValues);
  loadVector(in,input.vertexValues);

  // Convert to ANARI:
  std::vector<anari::math::float3> vertexPosition;
  std::vector<uint8_t> cellType;
  std::vector<uint32_t> index;
  std::vector<uint32_t> cellIndex;
  std::vector<float> vertexData;
  std::vector<float> cellData;

  for (size_t i=0; i<input.vertices.size(); ++i) {
    vertexPosition.push_back({input.vertices[i].x,input.vertices[i].y,input.vertices[i].z});
    vertexData.push_back(input.vertexValues[i]);
  }

  for (size_t i=0; i<input.connectivity.size(); ++i) {
    index.push_back((uint32_t)input.connectivity[i]);
  }

  for (size_t i=0; i<input.cellIndices.size(); ++i) {
    cellType.push_back((uint8_t)input.cellTypes[i]);
    cellIndex.push_back((uint32_t)input.cellIndices[i]);
  }

  g_appState.field = anari::newObject<anari::SpatialField>(device, "unstructured");

  anari::setParameterArray1D(device,
      g_appState.field,
      "vertex.position",
      ANARI_FLOAT32_VEC3,
      vertexPosition.data(),
      vertexPosition.size());

  anari::setParameterArray1D(device,
      g_appState.field,
      "vertex.data",
      ANARI_FLOAT32,
      vertexData.data(),
      vertexData.size());

  anari::setParameterArray1D(device,
      g_appState.field,
      "cell.type",
      ANARI_UINT8,
      cellType.data(),
      cellType.size());

  anari::setParameterArray1D(device,
      g_appState.field,
      "index",
      ANARI_UINT32,
      index.data(),
      index.size());

  anari::setParameterArray1D(device,
      g_appState.field,
      "cell.index",
      ANARI_UINT32,
      cellIndex.data(),
      cellIndex.size());

  anari::setParameterArray1D(device,
      g_appState.field,
      "cell.data",
      ANARI_FLOAT32,
      cellData.data(),
      cellData.size());

  anari::commitParameters(device, g_appState.field);

  g_appState.volume = anari::newObject<anari::Volume>(device, "transferFunction1D");
  anari::setParameter(device, g_appState.volume, "value", g_appState.field);

  anari::commitParameters(device, g_appState.volume);

  updateAnariTransfunc(device);

  // Create World //

  auto group = anari::newObject<anari::Group>(device);
  anari::setAndReleaseParameter(
      device, group, "volume", anari::newArray1D(device, &g_appState.volume));
  anari::commitParameters(device, group);

  auto inst = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, inst, "group", group);
  anari::commitParameters(device, inst);

  anari::World world = anari::newObject<anari::World>(device);
  anari::setAndReleaseParameter(
      device, world, "instance", anari::newArray1D(device, &inst));

  anari::commitParameters(device, world);

  return world;
}

void printUsage() {
  fprintf(stderr, "%s", "Usage: ex07_anari_app file.bin\n");
}

static void parseCommandLine(int argc, char *argv[]) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-' && endsWith(arg,".bin"))
      g_appState.filepath = arg;
  }
}

extern "C" int main(int argc, char *argv[]) {

  if (argc < 2) {
    printUsage();
    exit(-1);
  }

  parseCommandLine(argc, argv);

  Pipeline pl(argc, argv, "ex09_anari");

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(&fb);

  if (!pl.transfuncValid()) {
    auto &tf = g_appState.transfunc;
    std::vector<vec4f> tfValues({
      {0.f,0.f,1.f,0.1f },
      {0.f,1.f,0.f,0.1f }
    });
    tf.valueRange = {0.f,1.f};
    tf.setLUT(tfValues);
    pl.setTransfunc(&tf);
  }

  std::string libName = "environment";
  if (!getenv("ANARI_LIBRARY")) libName = "ex09_anari";
  auto library = anari::loadLibrary(libName.c_str(), statusFunc);
  auto device = anari::newDevice(library, "default");

  auto world = generateScene(device);
  if (!world) {
    printUsage();
    exit(-1);
  }

  auto renderer = anari::newObject<anari::Renderer>(device, "default");
  const anari::math::float4 backgroundColor = {0.1f, 0.1f, 0.1f, 1.f};
  anari::setParameter(device, renderer, "background", backgroundColor);
  anari::setParameter(device, renderer, "pixelSamples", 1);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);

  anari::math::uint2 imageSize = {(unsigned)imgWidth, (unsigned)imgHeight};
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);

  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "renderer", renderer);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");

  anari::commitParameters(device, camera);
  anari::setParameter(device, frame, "camera", camera);
  anari::commitParameters(device, frame);

  box3_t bounds;
  anari::getProperty(device, world, "bounds", bounds, ANARI_WAIT);

  Camera cam;
  cam.viewAll(*(box3f *)&bounds);
  pl.setCamera(&cam);

  do {
    struct {
      vec3f lower_left, horizontal, vertical;
    } screen;
    cam.getScreen(screen.lower_left,screen.horizontal,screen.vertical);

    anari::math::float3 eye(cam.getPosition().x,cam.getPosition().y,cam.getPosition().z);
    anari::math::float3 dir(cam.getPOI().x,cam.getPOI().y,cam.getPOI().z);
    dir -= eye;
    anari::math::float3 up(cam.getUp().x,cam.getUp().y,cam.getUp().z);

    anari::setParameter(device, camera, "position", eye);
    anari::setParameter(device, camera, "direction", dir);
    anari::setParameter(device, camera, "up", up);
    anari::setParameter(device, camera, "fovy", cam.getFovyInRadians());
    anari::setParameter(device, camera, "aspect", (float)fb.width/fb.height);
    anari::commitParameters(device, camera);

    anari::math::uint2 imageSize = {(unsigned)fb.width, (unsigned)fb.height};
    anari::setParameter(device, frame, "size", imageSize);
    anari::commitParameters(device, frame);

    updateAnariTransfunc(device);

    anari::render(device, frame);
    anari::wait(device, frame);

    auto af = anari::map<uint32_t>(device, frame, "channel.color");

    fb.uploadColor(af.data);
  
    anari::unmap(device, frame, "channel.color");

    pl.launch();
    pl.present();
  } while (pl.isRunning());
}

} // namespace ex09_anari



