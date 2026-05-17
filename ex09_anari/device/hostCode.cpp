// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

// ex09:
#include "hostCode.h"
#ifdef RTCORE
#include "Params-owl.h"
#endif

// common namespace for helper classes:
// Camera, FB, wrappers for RTX execution model, etc. etc.
using namespace dvr_course;

DECL_LAUNCH_PARAMS(ex09_anari::LaunchParams)

namespace ex09_anari {

#ifdef RTCORE
extern "C" char ptxCode[];
#else
extern void directLighting();
#endif

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

// TF1D volume ////////////////////////////////////////////////////////////////

TF1D::TF1D(GlobalState *s)
  : Object(ANARI_VOLUME, s),
    m_colorData(this), // observe changes
    m_opacityData(this) // observe changes
{}

void TF1D::commitParameters()
{
  m_field = getParamObject<SpatialField>("value");

  double valueRange_f[2] = {0.f, 1.f};
  double valueRange_d[2] = {0.0, 1.0};
  if (getParam("valueRange", ANARI_FLOAT32_BOX1, &valueRange_f[0])) {
    m_valueRange.lower = valueRange_f[0];
    m_valueRange.upper = valueRange_f[1];
  }
  if (getParam("valueRange", ANARI_FLOAT64_BOX1, &valueRange_d[0])) {
    m_valueRange.lower = float(valueRange_d[0]);
    m_valueRange.upper = float(valueRange_d[1]);
  }

  m_colorData = getParamObject<helium::Array1D>("color");
  m_uniformColor = vec4f(1.f);
  getParam("color", ANARI_FLOAT32_VEC3, &m_uniformColor);
  getParam("color", ANARI_FLOAT32_VEC4, &m_uniformColor);
  m_opacityData = getParamObject<helium::Array1D>("opacity");
  m_uniformOpacity = getParam<float>("opacity", 1.f) * m_uniformColor.w;
}

void TF1D::finalize()
{
  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no spatial field provided to transferFunction1D volume");
    return;
  }

  if (!m_field->isValid()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid spatial field provided to transferFunction1D volume");
    return;
  }

  size_t numColorChannels{4};
  if (m_colorData) { // TODO: more types
    if (m_colorData->elementType() == ANARI_FLOAT32_VEC3)
      numColorChannels = 3;
  }

  float *colorData = m_colorData ? (float *)m_colorData->data() : nullptr;
  float *opacityData = m_opacityData ? (float *)m_opacityData->data() : nullptr;

  size_t numColors = m_colorData ? m_colorData->size() : 1;
  size_t numOpacities = m_opacityData ? m_opacityData->size() : 1;
  size_t tfSize = max(numColors, numOpacities);

  // combine color and opacity data to single array:
  std::vector<vec4f> rgbaLUT(tfSize);
  for (size_t i=0; i<tfSize; ++i) {
    float colorPos = tfSize > 1 ? (float(i)/(tfSize-1))*(numColors-1) : 0.f;
    float colorFrac = colorPos-floorf(colorPos);

    vec4f color0(m_uniformColor.xyz, m_uniformOpacity);
    vec4f color1(m_uniformColor.xyz, m_uniformOpacity);
    if (colorData) {
      if (numColorChannels == 3) {
        vec3f *colors = (vec3f *)colorData;
        color0 = vec4f(colors[int(floorf(colorPos))], m_uniformOpacity);
        color1 = vec4f(colors[int(ceilf(colorPos))], m_uniformOpacity);
      }
      else if (numColorChannels == 4) {
        vec4f *colors = (vec4f *)colorData;
        color0 = colors[int(floorf(colorPos))];
        color1 = colors[int(ceilf(colorPos))];
      }
    }

    vec4f color = lerp(color0, color1, colorFrac);

    if (opacityData) {
      float alphaPos = tfSize > 1 ? (float(i)/(tfSize-1))*(numOpacities-1) : 0.f;
      float alphaFrac = alphaPos-floorf(alphaPos);

      float alpha0 = opacityData[int(floorf(alphaPos))];
      float alpha1 = opacityData[int(ceilf(alphaPos))];

      color.w *= lerp(alpha0, alpha1, alphaFrac);
    }

    rgbaLUT[i] = color;
  }

  m_impl.tf.valueRange = m_valueRange;
  m_impl.tf.setLUT(rgbaLUT);
}

// Unstructured field /////////////////////////////////////////////////////////

SpatialField::SpatialField(GlobalState *s) : Object(ANARI_SPATIAL_FIELD, s)
{}

// Nodes  /////////////////////////////////////////////////////////////////////

Group::Group(GlobalState *s) : Object(ANARI_GROUP, s)
{}

void Group::commitParameters()
{
  m_volumeData = getParamObject<helium::ObjectArray>("volume");
}

void Group::finalize()
{
  m_volumes.clear();

  if (m_volumeData) {
    std::for_each(m_volumeData->handlesBegin(),
        m_volumeData->handlesEnd(),
        [&](auto *o) {
          if (o && o->isValid()) {
            auto *vol = (TF1D *)o;
            m_volumes.push_back(vol);
          }
        });
  }
}

Instance::Instance(GlobalState *s) : Object(ANARI_INSTANCE, s)
{}

void Instance::commitParameters()
{
  m_group = getParamObject<Group>("group");
}

void Instance::finalize()
{
  if (!m_group)
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'group' on ANARIInstance");
}

// Structural  ////////////////////////////////////////////////////////////////

World::World(GlobalState *s) : Object(ANARI_WORLD, s)
{}

void World::commitParameters()
{
  m_volumeData = getParamObject<helium::ObjectArray>("volume");
  m_instanceData = getParamObject<helium::ObjectArray>("instance");
}

void World::finalize()
{
  m_volumes.clear();

  // volume data set on the world directly:
  if (m_volumeData) {
    std::for_each(m_volumeData->handlesBegin(),
        m_volumeData->handlesEnd(),
        [&](auto *o) {
          if (o && o->isValid()) {
            auto *vol = (TF1D *)o;
            m_volumes.push_back(vol);
          }
        });
  }

  // volume data coming through instances:

  // we don't support real instancing but just traverse the instances given and
  // grab the volume underneath (if any). A real ANARI device would implement
  // some instantation logic here:
  if (m_instanceData) {
    std::for_each(m_instanceData->handlesBegin(),
        m_instanceData->handlesEnd(),
        [&](auto *o) {
          if (o && o->isValid()) {
            auto *inst = (Instance *)o;
            if (inst->group()) {
              std::for_each(inst->group()->volumes().begin(),
                  inst->group()->volumes().end(),
                  [&](auto *v) {
                    m_volumes.push_back(v);
                  });
            }
          }
        });
  }
}

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
#ifdef RTCORE
  m_impl.pipeline.setRayGen(ptxCode, "directLighting");
  m_impl.pipeline.setLaunchParamsDecl(launchParams_owl, sizeof(LaunchParams));
#else
  m_impl.pipeline.setRayGen(directLighting);
#endif
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

  for (auto &vol: m_world->volumes()) {
  }

  struct {
    vec3f lower_left, horizontal, vertical;
  } screen;
  cam.getScreen(screen.lower_left,screen.horizontal,screen.vertical);

  // update camera:
  m_impl.pipeline.launchParam("camera.org", m_impl.parms.camera.org) = cam.getPosition();
  m_impl.pipeline.launchParam("camera.dir_00", m_impl.parms.camera.dir_00) = screen.lower_left;
  m_impl.pipeline.launchParam("camera.dir_du", m_impl.parms.camera.dir_du) = screen.horizontal / m_size.x;
  m_impl.pipeline.launchParam("camera.dir_dv", m_impl.parms.camera.dir_dv) = screen.vertical / m_size.y;
  // update framebuffer:
  m_impl.pipeline.launchParam("fbPointer", (RawPointer &)m_impl.parms.fbPointer) = m_impl.frame.fbPointer;
  m_impl.pipeline.launchParam("fbDepth", (RawPointer &)m_impl.parms.fbDepth) = m_impl.frame.fbDepth;
  m_impl.pipeline.launchParam("accumBuffer", (RawPointer &)m_impl.parms.accumBuffer) = m_impl.frame.accumBuffer;
  // update renderer params:
  m_impl.pipeline.launchParam("backgroundColor", m_impl.parms.backgroundColor) = vec4f(0.f);
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
  // set params:
  SET_LAUNCH_PARAMS(m_impl.parms);

  // only launch (ANARI takes over the 'present()' part):
  m_impl.pipeline.launch();
}

} // namespace ex09_anari

DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::Object *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::SpatialField *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::TF1D *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::Group *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::World *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::Camera *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::Renderer *);




