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

struct Renderer : public Object
{
  Renderer(GlobalState *s);
  virtual ~Renderer() = default;
};

struct Camera : public Object
{
  Camera(GlobalState *s);
  virtual ~Camera() = default;

  void commitParameters() override;
  void finalize() override;

  dvr_course::Camera getCamera() const;
 private:
  anari::math::float3 m_pos;
  anari::math::float3 m_dir;
  anari::math::float3 m_up;
  float m_fovy{0.f};
};

struct World : public Object
{
  World(GlobalState *s);
  virtual ~World() = default;
};

//=========================================================
// In ANARI, the frame is the object connecting world,
// renderer, and camera; so that's where we put the
// "pipeline" object from the previous samples. The
// frame is also involved when calling renderFrame()
// and will this be the central object for the lib's
// control flow on the host:
//=========================================================
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
 private:
  helium::IntrusivePtr<Renderer> m_renderer;
  helium::IntrusivePtr<Camera> m_camera;
  helium::IntrusivePtr<World> m_world;

  anari::math::uint2 m_size{0u,0u};
  anari::DataType m_colorType{ANARI_UNKNOWN};
  anari::DataType m_depthType{ANARI_UNKNOWN};
  int m_frameID{0};

  struct {
    dvr_course::Pipeline pipeline;
    dvr_course::Camera   camera;
    dvr_course::Frame    frame;
  } m_impl;
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

// macros to make a type known to ANARI as Object:
#define DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)              \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define DVR_COURSE_ANARI_TYPEFOR_DEFINITION(type)                              \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(ex09_anari::Camera *, ANARI_CAMERA);
DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(ex09_anari::Renderer *, ANARI_RENDERER);
DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(ex09_anari::World *, ANARI_WORLD);


