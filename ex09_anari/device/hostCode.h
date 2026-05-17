// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include "helium/array/Array1D.h"
#include "helium/array/ObjectArray.h"
#include "helium/utility/ChangeObserverPtr.h"
#include "helium/BaseObject.h"
#include "helium/BaseFrame.h"
// ours
#include <dvr_course-common.h>
// ex09:
#include "Params.h"

namespace ex09_anari {

struct GlobalState : public helium::BaseGlobalDeviceState
{
  GlobalState(ANARIDevice d) : helium::BaseGlobalDeviceState(d)
  { m_pipeline.setHeadless(true); }

  ~GlobalState() = default;

  dvr_course::Pipeline &pipeline()
  { return m_pipeline; }

  LaunchParams &parms()
  { return m_parms; }
 private:
  dvr_course::Pipeline m_pipeline;
  LaunchParams         m_parms;
};

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
  dvr_course::Pipeline &pipeline();
  LaunchParams &parms();
};

struct SpatialField : public Object
{
  SpatialField(GlobalState *s);
  virtual ~SpatialField() = default;

  void commitParameters() override;
  void finalize() override;

#ifdef RTCORE
  OWLGroup getTLAS()
  { return m_TLAS; }
#endif

  Volume getVolume()
  { return m_volume; }

 private:
  struct {
    helium::IntrusivePtr<helium::Array1D> vertexPosition;
    helium::IntrusivePtr<helium::Array1D> vertexData;
    helium::IntrusivePtr<helium::Array1D> index;
    helium::IntrusivePtr<helium::Array1D> cellIndex;
    helium::IntrusivePtr<helium::Array1D> cellType;
    helium::IntrusivePtr<helium::Array1D> cellData;
  } m_params;

#ifdef RTCORE
  OWLGroup m_TLAS;
#endif

  dvr_course::Buffer<Tet> m_tets;
  Volume m_volume;
};

struct TF1D : public Object
{
  TF1D(GlobalState *s);
  virtual ~TF1D() = default;

  void commitParameters() override;
  void finalize() override;

  SpatialField *getField()
  { return m_params.field.ptr; }

  dvr_course::Transfunc *getTransfunc()
  { return m_transfunc.get(); }

 private:
  struct Params {
    Params(TF1D *parent) : colorData(parent), opacityData(parent) {}
    box1f valueRange{0.f, 1.f};
    float unitDistance{1.f};
    vec4f uniformColor{1.f, 1.f, 1.f, 1.f};
    float uniformOpacity{1.f};

    helium::IntrusivePtr<SpatialField> field;
    helium::ChangeObserverPtr<helium::Array1D> colorData;
    helium::ChangeObserverPtr<helium::Array1D> opacityData;
  } m_params;

  std::unique_ptr<dvr_course::Transfunc> m_transfunc;
};

struct Group : public Object
{
  Group(GlobalState *s);
  virtual ~Group() = default;

  void commitParameters() override;
  void finalize() override;

  std::vector<TF1D *> &volumes()
  { return m_volumes; }

 private:
  helium::IntrusivePtr<helium::ObjectArray> m_volumeData;
  std::vector<TF1D *> m_volumes;
};

struct Instance : public Object
{
  Instance(GlobalState *s);
  virtual ~Instance() = default;

  void commitParameters() override;
  void finalize() override;

  Group *group()
  { return m_group.ptr; }

 private:
  helium::IntrusivePtr<Group> m_group;
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

  dvr_course::Camera *getCamera()
  { return m_camera.get(); }

 private:
  anari::math::float3 m_pos;
  anari::math::float3 m_dir;
  anari::math::float3 m_up;
  float m_fovy{0.f};

  std::unique_ptr<dvr_course::Camera> m_camera;
};

struct World : public Object
{
  World(GlobalState *s);
  virtual ~World() = default;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;

  std::vector<TF1D *> volumes()
  { return m_volumes; }

 private:
  helium::IntrusivePtr<helium::ObjectArray> m_volumeData;
  helium::IntrusivePtr<helium::ObjectArray> m_instanceData;

  std::vector<TF1D *> m_volumes;
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

  GlobalState *deviceState() const;
  dvr_course::Pipeline &pipeline();
  LaunchParams &parms();

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

  dvr_course::Buffer<Transfunc> m_TFs;

  std::unique_ptr<dvr_course::Frame> m_frame;
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

DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(ex09_anari::Object *, ANARI_OBJECT);
DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(ex09_anari::SpatialField *, ANARI_SPATIAL_FIELD);
DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(ex09_anari::TF1D *, ANARI_VOLUME);
DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(ex09_anari::Group *, ANARI_GROUP);
DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(ex09_anari::World *, ANARI_WORLD);
DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(ex09_anari::Camera *, ANARI_CAMERA);
DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(ex09_anari::Renderer *, ANARI_RENDERER);
DVR_COURSE_ANARI_TYPEFOR_SPECIALIZATION(ex09_anari::Frame *, ANARI_FRAME);


