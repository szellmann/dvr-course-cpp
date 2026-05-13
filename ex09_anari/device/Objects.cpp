// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include "Objects.h"

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



