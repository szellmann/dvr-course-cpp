// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "hostCode.h"

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

// Structural  ////////////////////////////////////////////////////////////////

Instance::Instance(GlobalState *s) : Object(ANARI_INSTANCE, s)
{}

Group::Group(GlobalState *s) : Object(ANARI_GROUP, s)
{}

World::World(GlobalState *s) : Object(ANARI_WORLD, s)
{}

// Frame //////////////////////////////////////////////////////////////////////

Frame::Frame(GlobalState *s) : helium::BaseFrame(s)
{}

bool Frame::isValid() const
{
}

void Frame::commitParameters()
{
}

void Frame::finalize()
{
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
  *width = 0;
  *height = 0;

  if (channel == "color" || channel == "channel.color") {
    *pixelType = ANARI_UFIXED8_RGBA_SRGB;
    return 0;//mapColorBuffer();
  } else if (channel == "depth" || channel == "channel.depth") {
    *pixelType = ANARI_FLOAT32;
    return 0;//mapDepthBuffer();
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
}

// Renderer ///////////////////////////////////////////////////////////////////

Renderer::Renderer(GlobalState *s) : Object(ANARI_RENDERER, s)
{}

// Perspective camera /////////////////////////////////////////////////////////

Camera::Camera(GlobalState *s) : Object(ANARI_CAMERA, s)
{}

// TF1D volume ////////////////////////////////////////////////////////////////

Volume::Volume(GlobalState *s) : Object(ANARI_VOLUME, s)
{}

// Unstructured field /////////////////////////////////////////////////////////

SpatialField::SpatialField(GlobalState *s) : Object(ANARI_SPATIAL_FIELD, s)
{}

} // namespace ex09_anari



