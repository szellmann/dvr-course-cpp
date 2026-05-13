// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

// helium
#include "helium/array/Array1D.h"
#include "helium/array/Array2D.h"
#include "helium/array/Array3D.h"
#include "helium/array/ObjectArray.h"

// ours
#include "Device.h"
#include "Objects.h"

#include "anari_library_ex09_anari_queries.h"

namespace ex09_anari {

// Data Arrays ////////////////////////////////////////////////////////////////

void *Device::mapArray(ANARIArray a)
{
  return helium::BaseDevice::mapArray(a);
}

void Device::unmapArray(ANARIArray a)
{
  helium::BaseDevice::unmapArray(a);
}

// API Objects ////////////////////////////////////////////////////////////////

ANARIArray1D Device::newArray1D(const void *appMemory,
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

ANARIArray2D Device::newArray2D(const void *appMemory,
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

ANARIArray3D Device::newArray3D(const void *appMemory,
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

ANARICamera Device::newCamera(const char * /*subtype*/)
{
  initDevice();
  return (ANARICamera) new Camera(deviceState());
}

ANARIFrame Device::newFrame()
{
  initDevice();
  return (ANARIFrame) new Frame(deviceState());
}

ANARIGroup Device::newGroup()
{
  initDevice();
  return (ANARIGroup) new Group(deviceState());
}

ANARILight Device::newLight(const char * /*subtype*/)
{
  return {};
}

ANARIMaterial Device::newMaterial(const char * /*subtype*/)
{
  return {};
}

ANARIGeometry Device::newGeometry(const char * /*subtype*/)
{
  return {};
}

ANARIInstance Device::newInstance(const char * /*subtype*/)
{
  initDevice();
  return (ANARIInstance) new Instance(deviceState());
}

ANARIRenderer Device::newRenderer(const char * /*subtype*/)
{
  initDevice();
  return (ANARIRenderer) new Renderer(deviceState());
}

ANARISampler Device::newSampler(const char * /*subtype*/)
{
  return {};
}

ANARISpatialField Device::newSpatialField(const char * /*subtype*/)
{
  initDevice();
  return (ANARISpatialField) new SpatialField(deviceState());
}

ANARISurface Device::newSurface()
{
  return {};
}

ANARIVolume Device::newVolume(const char * /*subtype*/)
{
  initDevice();
  return (ANARIVolume) new Volume(deviceState());
}

ANARIWorld Device::newWorld()
{
  initDevice();
  return (ANARIWorld) new World(deviceState());
}

// Query functions ////////////////////////////////////////////////////////////

const char **Device::getObjectSubtypes(ANARIDataType objectType)
{
  return ex09_anari::query_object_types(objectType);
}

const void *Device::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  return ex09_anari::query_object_info(
      objectType, objectSubtype, infoName, infoType);
}

const void *Device::getParameterInfo(ANARIDataType objectType,
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

Device::Device(ANARIStatusCallback cb, const void *ptr)
    : helium::BaseDevice(cb, ptr)
{
  m_state = std::make_unique<helium::BaseGlobalDeviceState>(this_device());
  deviceCommitParameters();
}

Device::Device(ANARILibrary l) : helium::BaseDevice(l)
{
  m_state = std::make_unique<helium::BaseGlobalDeviceState>(this_device());
  deviceCommitParameters();
}

Device::~Device()
{
  auto &state = *deviceState();

  state.commitBuffer.clear();

  reportMessage(ANARI_SEVERITY_DEBUG, "destroying ex09_anari device (%p)", this);

  // TODO: clear context?!
}

void Device::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing ex09_anari device (%p)", this);
  auto &state = *deviceState();

  //state.anariDevice = (anari::Device)this;

  m_initialized = true;
}

void Device::deviceCommitParameters()
{
  auto &state = *deviceState();

  helium::BaseDevice::deviceCommitParameters();
}

int Device::deviceGetProperty(
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

helium::BaseGlobalDeviceState *Device::deviceState() const
{
  return (helium::BaseGlobalDeviceState *)helium::BaseDevice::m_state.get();
}

} // namespace ex09_anari




