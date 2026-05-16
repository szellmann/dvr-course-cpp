// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include "helium/BaseObject.h"
#include "helium/BaseFrame.h"
// ours
#include <dvr_course-common.h>

namespace ex09_anari {

typedef helium::BaseGlobalDeviceState GlobalState;

struct Object : public helium::BaseObject
{
  Object(ANARIDataType type, GlobalState *s);
  virtual ~Object() = default;

  virtual bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  virtual void commitParameters() override;
  virtual void finalize() override;

  virtual bool isValid() const override;

  GlobalState *deviceState() const;
};

struct Camera : public Object
{
  Camera(GlobalState *s);
  virtual ~Camera() = default;
};

struct Frame : public helium::BaseFrame
{
  Frame(GlobalState *s);
  ~Frame() = default;

  bool isValid() const override;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;

  void renderFrame() override;

  void *map(std::string_view channel,
      uint32_t *width,
      uint32_t *height,
      ANARIDataType *pixelType) override;
  void unmap(std::string_view channel) override;
  int frameReady(ANARIWaitMask m) override;
  void discard() override;
};

struct Renderer : public Object
{
  Renderer(GlobalState *s);
  virtual ~Renderer() = default;
};

struct World : public Object
{
  World(GlobalState *s);
  virtual ~World() = default;
};

struct Group : public Object
{
  Group(GlobalState *s);
  virtual ~Group() = default;
};

struct Instance : public Object
{
  Instance(GlobalState *s);
  virtual ~Instance() = default;
};

struct Volume : public Object
{
  Volume(GlobalState *s);
  virtual ~Volume() = default;
};

struct SpatialField : public Object
{
  SpatialField(GlobalState *s);
  virtual ~SpatialField() = default;
};

} // ex09_anari



