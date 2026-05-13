// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Device.h"
#include "anari/backend/LibraryImpl.h"
#include "anari_library_ex09_anari_export.h"

namespace ex09_anari {

const char **query_extensions();

struct Library : public anari::LibraryImpl
{
  Library(
      void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr);

  ANARIDevice newDevice(const char *subtype) override;
  const char **getDeviceExtensions(const char *deviceType) override;
};

// Definitions ////////////////////////////////////////////////////////////////

Library::Library(
    void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr)
    : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
{}

ANARIDevice Library::newDevice(const char * /*subtype*/)
{
  return (ANARIDevice) new Device(this_library());
}

const char **Library::getDeviceExtensions(const char * /*deviceType*/)
{
  return query_extensions();
}

} // namespace ex09_anari

// Define library entrypoint //////////////////////////////////////////////////

extern "C" DVR_COURSE_DEVICE_INTERFACE ANARI_DEFINE_LIBRARY_ENTRYPOINT(
    ex09_anari, handle, scb, scbPtr)
{
  return (ANARILibrary) new ex09_anari::Library(handle, scb, scbPtr);
}

extern "C" DVR_COURSE_DEVICE_INTERFACE ANARIDevice anariNewDVRCourseDevice(
    ANARIStatusCallback defaultCallback, const void *userPtr)
{
  return (ANARIDevice) new ex09_anari::Device(defaultCallback, userPtr);
}
