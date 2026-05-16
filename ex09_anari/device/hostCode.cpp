// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "hostCode.h"

using namespace vecmath;

namespace ex09_anari {

// Object definitions /////////////////////////////////////////////////////////

Object::Object(ANARIDataType type, GlobalState *s)
    : helium::BaseObject(type, s)
{
  helium::BaseObject::markParameterChanged();
  s->commitBuffer.addObjectToCommit(this);
}

void Object::commitParameters()
{
  // no-op
}

void Object::finalize()
{
  // no-op
}

bool Object::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint64_t size, uint32_t flags)
{
  if (name == "valid" && type == ANARI_BOOL) {
    helium::writeToVoidP(ptr, isValid());
    return true;
  }

  return false;
}

bool Object::isValid() const
{
  return true;
}

GlobalState *Object::deviceState() const
{
  return (GlobalState *)helium::BaseObject::m_state;
}

// Nodes  /////////////////////////////////////////////////////////////////////

Instance::Instance(GlobalState *s) : Object(ANARI_INSTANCE, s)
{}

Group::Group(GlobalState *s) : Object(ANARI_GROUP, s)
{}

// Structural  ////////////////////////////////////////////////////////////////

World::World(GlobalState *s) : Object(ANARI_WORLD, s)
{}

// Renderer ///////////////////////////////////////////////////////////////////

Renderer::Renderer(GlobalState *s) : Object(ANARI_RENDERER, s)
{}

// Perspective camera /////////////////////////////////////////////////////////

Camera::Camera(GlobalState *s) : Object(ANARI_CAMERA, s)
{}

void Camera::commitParameters()
{
  m_pos = getParam<anari::math::float3>("position", anari::math::float3(0.f));
  m_dir = normalize(getParam<anari::math::float3>("direction", anari::math::float3(0.f, 0.f, 1.f)));
  m_up = normalize(getParam<anari::math::float3>("up", anari::math::float3(0.f, 1.f, 0.f)));
  if (!getParam("fovy", ANARI_FLOAT32, &m_fovy))
    m_fovy = 60.f * float(M_PI)/180.f;
}

void Camera::finalize()
{
}

dvr_course::Camera Camera::getCamera() const
{
  dvr_course::Camera cam;
  auto poi = m_pos+m_dir;
  cam.setOrientation(vec3f(m_pos.x,m_pos.y,m_pos.z),
                     vec3f(poi.x,poi.y,poi.z),
                     vec3f(m_up.x,m_up.y,m_up.z),
                     m_fovy);
  return cam;
}

// Frame //////////////////////////////////////////////////////////////////////

Frame::Frame(GlobalState *s) : helium::BaseFrame(s)
{
  m_impl.pipeline.setFrame(&m_impl.frame);
  m_impl.pipeline.setCamera(&m_impl.camera);
}

bool Frame::isValid() const
{
}

void Frame::commitParameters()
{
  m_renderer = getParamObject<Renderer>("renderer");
  m_camera = getParamObject<Camera>("camera");
  m_world = getParamObject<World>("world");

  m_size = getParam<anari::math::uint2>("size", anari::math::uint2(10));
  m_colorType = getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
  m_depthType = getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);
}

void Frame::finalize()
{
  if (!m_renderer) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'renderer' on frame");
  }

  if (!m_camera) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
  }

  if (!m_world) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
  }

  if (m_colorType != ANARI_UFIXED8_RGBA_SRGB) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "Unsupported color type on frame");
  }

  if (m_colorType != ANARI_FLOAT32) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "Unsupported color type on frame");
  }

  m_impl.frame.resize(m_size.x,m_size.y);
  auto cam = m_camera->getCamera();
  m_impl.camera.setOrientation(cam.getPosition(),
                               cam.getPOI(),
                               cam.getUp(),
                               cam.getFovyInRadians());
}

bool Frame::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint64_t size, uint32_t flags)
{
}

void *Frame::map(std::string_view channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  *width = m_impl.frame.width;
  *height = m_impl.frame.height;

  if (channel == "color" || channel == "channel.color") {
    *pixelType = ANARI_UFIXED8_RGBA_SRGB;
    *width = m_impl.frame.width;
    return m_impl.frame.fbPointer;
  } else if (channel == "depth" || channel == "channel.depth") {
    *pixelType = ANARI_FLOAT32;
    return m_impl.frame.fbDepth;
  } else {
    *width = 0;
    *height = 0;
    *pixelType = ANARI_UNKNOWN;
    return nullptr;
  }
}

void Frame::unmap(std::string_view channel)
{
  // no-op
}
int Frame::frameReady(ANARIWaitMask m)
{
  return 1;
}

void Frame::discard()
{
  // no-op
}

void Frame::renderFrame()
{
  // only launch (ANARI takes over the 'present()' part):
  m_impl.pipeline.launch();
}

// TF1D volume ////////////////////////////////////////////////////////////////

Volume::Volume(GlobalState *s) : Object(ANARI_VOLUME, s)
{}

// Unstructured field /////////////////////////////////////////////////////////

SpatialField::SpatialField(GlobalState *s) : Object(ANARI_SPATIAL_FIELD, s)
{}

} // namespace ex09_anari



