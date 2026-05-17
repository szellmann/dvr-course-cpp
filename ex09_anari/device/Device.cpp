// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

// helium
#include "helium/array/Array1D.h"
#include "helium/array/Array2D.h"
#include "helium/array/Array3D.h"
#include "helium/array/ObjectArray.h"

// ours
#include "Device.h"
#include "hostCode.h"

#include "anari_library_ex09_anari_queries.h"

namespace ex09_anari {

// Data Arrays ////////////////////////////////////////////////////////////////

void *DVRCourseDevice::mapArray(ANARIArray a)
{
  return helium::BaseDevice::mapArray(a);
}

void DVRCourseDevice::unmapArray(ANARIArray a)
{
  helium::BaseDevice::unmapArray(a);
}

// API Objects ////////////////////////////////////////////////////////////////

ANARIArray1D DVRCourseDevice::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
  initDevice();

  helium::Array1DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems = numItems;

  if (anari::isObject(type))
    return (ANARIArray1D) new helium::ObjectArray(deviceState(), md);
  else
    return (ANARIArray1D) new helium::Array1D(deviceState(), md);
}

ANARIArray2D DVRCourseDevice::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  initDevice();

  helium::Array2DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;

  return (ANARIArray2D) new helium::Array2D(deviceState(), md);
}

ANARIArray3D DVRCourseDevice::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  initDevice();

  helium::Array3DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;
  md.numItems3 = numItems3;

  return (ANARIArray3D) new helium::Array3D(deviceState(), md);
}

ANARICamera DVRCourseDevice::newCamera(const char * /*subtype*/)
{
  initDevice();
  return (ANARICamera) new Camera(deviceState());
}

ANARIFrame DVRCourseDevice::newFrame()
{
  initDevice();
  return (ANARIFrame) new Frame(deviceState());
}

ANARIGroup DVRCourseDevice::newGroup()
{
  initDevice();
  return (ANARIGroup) new Group(deviceState());
}

ANARILight DVRCourseDevice::newLight(const char * /*subtype*/)
{
  return {};
}

ANARIMaterial DVRCourseDevice::newMaterial(const char * /*subtype*/)
{
  return {};
}

ANARIGeometry DVRCourseDevice::newGeometry(const char * /*subtype*/)
{
  return {};
}

ANARIInstance DVRCourseDevice::newInstance(const char * /*subtype*/)
{
  initDevice();
  return (ANARIInstance) new Instance(deviceState());
}

ANARIRenderer DVRCourseDevice::newRenderer(const char * /*subtype*/)
{
  initDevice();
  return (ANARIRenderer) new Renderer(deviceState());
}

ANARISampler DVRCourseDevice::newSampler(const char * /*subtype*/)
{
  return {};
}

ANARISpatialField DVRCourseDevice::newSpatialField(const char * /*subtype*/)
{
  initDevice();
  return (ANARISpatialField) new SpatialField(deviceState());
}

ANARISurface DVRCourseDevice::newSurface()
{
  return {};
}

ANARIVolume DVRCourseDevice::newVolume(const char * /*subtype*/)
{
  initDevice();
  return (ANARIVolume) new TF1D(deviceState());
}

ANARIWorld DVRCourseDevice::newWorld()
{
  initDevice();
  return (ANARIWorld) new World(deviceState());
}

// Query functions ////////////////////////////////////////////////////////////

const char **DVRCourseDevice::getObjectSubtypes(ANARIDataType objectType)
{
  return ex09_anari::query_object_types(objectType);
}

const void *DVRCourseDevice::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  return ex09_anari::query_object_info(
      objectType, objectSubtype, infoName, infoType);
}

const void *DVRCourseDevice::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  return ex09_anari::query_param_info(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

// Other Device definitions ///////////////////////////////////////////////

DVRCourseDevice::DVRCourseDevice(ANARIStatusCallback cb, const void *ptr)
    : helium::BaseDevice(cb, ptr)
{
  m_state = std::make_unique<GlobalState>(this_device());
  deviceCommitParameters();
}

DVRCourseDevice::DVRCourseDevice(ANARILibrary l) : helium::BaseDevice(l)
{
  m_state = std::make_unique<GlobalState>(this_device());
  deviceCommitParameters();
}

DVRCourseDevice::~DVRCourseDevice()
{
  auto &state = *deviceState();

  state.commitBuffer.clear();

  reportMessage(ANARI_SEVERITY_DEBUG, "destroying ex09_anari device (%p)", this);

  // TODO: clear context?!
}

void DVRCourseDevice::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing ex09_anari device (%p)", this);
  auto &state = *deviceState();

  //state.anariDevice = (anari::Device)this;

  m_initialized = true;
}

void DVRCourseDevice::deviceCommitParameters()
{
  auto &state = *deviceState();

  helium::BaseDevice::deviceCommitParameters();
}

int DVRCourseDevice::deviceGetProperty(
    const char *name, ANARIDataType type, void *mem, uint64_t size, uint32_t mask)
{
  std::string_view prop = name;
  if (prop == "extension" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_extensions());
    return 1;
  } else if (prop == "ex09_anari" && type == ANARI_BOOL) {
    helium::writeToVoidP(mem, true);
    return 1;
  }
  return 0;
}

GlobalState *DVRCourseDevice::deviceState() const
{
  return (GlobalState *)helium::BaseDevice::m_state.get();
}

} // namespace ex09_anari




