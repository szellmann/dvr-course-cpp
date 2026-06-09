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

GlobalState::GlobalState(ANARIDevice d) : helium::BaseGlobalDeviceState(d)
{
  Pipeline::RTConfig conf;
#ifdef RTCORE
  conf.ptxCode = ptxCode;
  conf.launchParamsDecl = launchParams_owl;
  conf.sizeOfLaunchParamsStruct = sizeof(LaunchParams);
#else
  conf.rayGens.push_back({"directLighting",directLighting});
#endif
  m_pipeline.initRT(conf);
  m_pipeline.setRayGen("directLighting");
  m_pipeline.setHeadless(true);
}

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

Pipeline &Object::pipeline()
{
  return deviceState()->pipeline();
}

LaunchParams &Object::parms()
{
  return deviceState()->parms();
}

// UnknownObject definitions //////////////////////////////////////////////////

UnknownObject::UnknownObject(ANARIDataType type, GlobalState *s)
    : Object(type, s)
{}

bool UnknownObject::isValid() const
{
  return false;
}

// Unstructured field /////////////////////////////////////////////////////////

SpatialField::SpatialField(GlobalState *s) : Object(ANARI_SPATIAL_FIELD, s)
{}

void SpatialField::commitParameters()
{
  m_params.vertexPosition = getParamObject<helium::Array1D>("vertex.position");
  m_params.vertexData = getParamObject<helium::Array1D>("vertex.data");
  m_params.index = getParamObject<helium::Array1D>("index");
  m_params.cellIndex = getParamObject<helium::Array1D>("cell.index");
  m_params.cellType = getParamObject<helium::Array1D>("cell.type");
  m_params.cellData = getParamObject<helium::Array1D>("cell.data");
}

void SpatialField::finalize()
{
  if (!m_params.vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on unstructured spatial field");
    return;
  }

  if (!(m_params.vertexData || m_params.cellData)) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.data' (or 'cellData') on unstructured spatial field");
    return;
  }

  if (!m_params.index) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'index' on unstructured spatial field");
    return;
  }

  if (!m_params.cellIndex) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter(s) 'cell.index' on unstructured spatial field");
    return;
  }

  if (!m_params.cellType) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter(s) 'cell.type' on unstructured spatial field");
    return;
  }

  auto *cellTypes = m_params.cellType->beginAs<uint8_t>();
  auto *vertices = m_params.vertexPosition->beginAs<anari::math::float3>();
  auto *vertexValues = m_params.vertexData? m_params.vertexData->beginAs<float>(): nullptr;
  auto *connectivity = m_params.index->beginAs<uint32_t>();
  auto *cellIndices = m_params.cellIndex->beginAs<uint32_t>();
  auto *cellValues = m_params.cellData? m_params.cellData->beginAs<float>(): nullptr;

  // assemble tets:
  std::vector<Tet> tets;
  box3f bounds(INFINITY,-INFINITY);
  box1f range(INFINITY,-INFINITY);
  for (size_t i=0; i<m_params.cellType->size(); ++i) {
    #define VTK_TET_ 10
    if (cellTypes[i] != VTK_TET_) continue;
    uint32_t i0 = connectivity[cellIndices[i]];
    uint32_t i1 = connectivity[cellIndices[i]+1];
    uint32_t i2 = connectivity[cellIndices[i]+2];
    uint32_t i3 = connectivity[cellIndices[i]+3];
    auto v0 = vertices[i0];
    auto v1 = vertices[i1];
    auto v2 = vertices[i2];
    auto v3 = vertices[i3];
    float s0, s1, s2, s3;
    if (vertexValues != nullptr) {
      s0 = vertexValues[i0];
      s1 = vertexValues[i1];
      s2 = vertexValues[i2];
      s3 = vertexValues[i3];
    } else {
      s0 = cellValues[cellIndices[i]/4];
      s1 = cellValues[cellIndices[i]/4];
      s2 = cellValues[cellIndices[i]/4];
      s3 = cellValues[cellIndices[i]/4];
    }

    vec3f vv0(v0.x,v0.y,v0.z);
    vec3f vv1(v1.x,v1.y,v1.z);
    vec3f vv2(v2.x,v2.y,v2.z);
    vec3f vv3(v3.x,v3.y,v3.z);

    bounds.extend(vv0); bounds.extend(vv1); bounds.extend(vv2); bounds.extend(vv3);
    range.extend(s0); range.extend(s1); range.extend(s2); range.extend(s3);

    // Store tets in our simple, flattened format, i.e., values are encoded in
    // the 'w' coordinate of the positional vectors
    Tet tet;
    tet.v0 = vec4f(vv0,s0);
    tet.v1 = vec4f(vv1,s1);
    tet.v2 = vec4f(vv2,s2);
    tet.v3 = vec4f(vv3,s3);
    tets.push_back(tet);
  }

  m_tets = dvr_course::Buffer(tets.size(),tets.data());

  m_volume.type = Volume::TET;
  m_volume.asTetMesh.tets = m_tets.data();
  m_volume.asTetMesh.numTets = (int)m_tets.size();
  m_volume.bounds = bounds;
  m_volume.dataRange = range;

#ifdef RTCORE
  OWLVarDecl tetsGeomVars[]
  = {
     { "tets",  OWL_BUFPTR, OWL_OFFSETOF(TetMesh,tets)},
     { "numTets",  OWL_INT, OWL_OFFSETOF(TetMesh,numTets)},
     { nullptr /* sentinel to mark end of list */ }
  };
  OWLGeomType userGeomType = owlGeomTypeCreate(pipeline().owlContext(),
                                               OWL_GEOM_USER,
                                               sizeof(TetMesh),
                                               tetsGeomVars, -1);
  owlGeomTypeSetBoundsProg(userGeomType, pipeline().owlModule(), "TetBounds");
  owlGeomTypeSetIntersectProg(userGeomType, 0, pipeline().owlModule(), "TetIntersect");
  owlGeomTypeSetClosestHit(userGeomType, 0, pipeline().owlModule(), "TetClosestHit");

  OWLGeom userGeom = owlGeomCreate(pipeline().owlContext(), userGeomType);
  owlGeomSetPrimCount(userGeom, tets.size());

  OWLBuffer tetBuffer = owlDeviceBufferCreate(pipeline().owlContext(),
                                              OWL_USER_TYPE(Tet{}),
                                              tets.size(),
                                              tets.data());
  owlGeomSetBuffer(userGeom, "tets", tetBuffer);
  owlGeomSet1i(userGeom, "numTets", (int)tets.size());

  owlBuildPrograms(pipeline().owlContext());

  OWLGroup userGeomBLAS = owlUserGeomGroupCreate(pipeline().owlContext(), 1, &userGeom);
  owlGroupBuildAccel(userGeomBLAS);

  m_TLAS = owlInstanceGroupCreate(pipeline().owlContext(), 1);
  owlInstanceGroupSetChild(m_TLAS, 0, userGeomBLAS);

  owlGroupBuildAccel(m_TLAS);

  m_volume.asTetMesh.handle = owlGroupGetTraversable(m_TLAS, 0);
#endif
}

// TF1D volume ////////////////////////////////////////////////////////////////

TF1D::TF1D(GlobalState *s)
  : Object(ANARI_VOLUME, s), m_params(this), m_transfunc(new dvr_course::Transfunc)
{}

void TF1D::commitParameters()
{
  m_params.field = getParamObject<SpatialField>("value");

  float valueRange_f[2] = {0.f, 1.f};
  double valueRange_d[2] = {0.0, 1.0};
  if (getParam("valueRange", ANARI_FLOAT32_BOX1, &valueRange_f[0])) {
    m_params.valueRange.lower = valueRange_f[0];
    m_params.valueRange.upper = valueRange_f[1];
  }
  if (getParam("valueRange", ANARI_FLOAT64_BOX1, &valueRange_d[0])) {
    m_params.valueRange.lower = float(valueRange_d[0]);
    m_params.valueRange.upper = float(valueRange_d[1]);
  }

  m_params.colorData = getParamObject<helium::Array1D>("color");
  m_params.uniformColor = vec4f(1.f);
  getParam("color", ANARI_FLOAT32_VEC3, &m_params.uniformColor);
  getParam("color", ANARI_FLOAT32_VEC4, &m_params.uniformColor);
  m_params.opacityData = getParamObject<helium::Array1D>("opacity");
  m_params.uniformOpacity = getParam<float>("opacity", 1.f) * m_params.uniformColor.w;
}

void TF1D::finalize()
{
  if (!m_params.field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no spatial field provided to transferFunction1D volume");
    return;
  }

  if (!m_params.field->isValid()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "invalid spatial field provided to transferFunction1D volume");
    return;
  }

  size_t numColorChannels{4};
  if (m_params.colorData) { // TODO: more types
    if (m_params.colorData->elementType() == ANARI_FLOAT32_VEC3)
      numColorChannels = 3;
  }

  float *colorData = m_params.colorData ? (float *)m_params.colorData->data() : nullptr;
  float *opacityData = m_params.opacityData ? (float *)m_params.opacityData->data() : nullptr;

  size_t numColors = m_params.colorData ? m_params.colorData->size() : 1;
  size_t numOpacities = m_params.opacityData ? m_params.opacityData->size() : 1;
  size_t tfSize = max(numColors, numOpacities);

  // combine color and opacity data to single array:
  std::vector<vec4f> rgbaLUT(tfSize);
  for (size_t i=0; i<tfSize; ++i) {
    float colorPos = tfSize > 1 ? (float(i)/(tfSize-1))*(numColors-1) : 0.f;
    float colorFrac = colorPos-floorf(colorPos);

    vec4f color0(m_params.uniformColor.xyz, m_params.uniformOpacity);
    vec4f color1(m_params.uniformColor.xyz, m_params.uniformOpacity);
    if (colorData) {
      if (numColorChannels == 3) {
        vec3f *colors = (vec3f *)colorData;
        color0 = vec4f(colors[int(floorf(colorPos))], m_params.uniformOpacity);
        color1 = vec4f(colors[int(ceilf(colorPos))], m_params.uniformOpacity);
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

  m_transfunc->valueRange = m_params.valueRange;
  m_transfunc->setLUT(rgbaLUT);

  pipeline().markTransfuncUpdate();
}

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

bool World::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint64_t size, uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    box3f bounds(INFINITY,-INFINITY);
    for (auto vol: volumes()) {
      auto field = vol->getField();
      if (!field) continue;
      bounds.extend(field->getVolume().bounds);
    }
    std::memcpy(ptr, &bounds, sizeof(bounds));
    return true;
  }

  return Object::getProperty(name, type, ptr, size, flags);
}

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

Camera::Camera(GlobalState *s) : Object(ANARI_CAMERA, s), m_camera(new dvr_course::Camera)
{
  pipeline().setCamera(m_camera.get());
}

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
  auto poi = m_pos+m_dir;
  m_camera->setOrientation(vec3f(m_pos.x,m_pos.y,m_pos.z),
                           vec3f(poi.x,poi.y,poi.z),
                           vec3f(m_up.x,m_up.y,m_up.z),
                           m_fovy);
  pipeline().markCameraUpdate();
}

// Frame //////////////////////////////////////////////////////////////////////

Frame::Frame(GlobalState *s) : helium::BaseFrame(s), m_frame(new dvr_course::Frame)
{
  pipeline().setFrame(m_frame.get());
}

bool Frame::isValid() const
{
  return true;
}

GlobalState *Frame::deviceState() const
{
  return (GlobalState *)helium::BaseObject::m_state;
}

Pipeline &Frame::pipeline()
{
  return deviceState()->pipeline();
}

LaunchParams &Frame::parms()
{
  return deviceState()->parms();
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

  m_frame->resize(m_size.x,m_size.y);

  std::vector<Volume> volumes;
  for (size_t i=0, validID=0; i<m_world->volumes().size(); ++i) {
    auto vol = m_world->volumes()[i];
    auto field = vol->getField();
    if (!field) continue;
    volumes.push_back(field->getVolume());
    auto tf = vol->getTransfunc();
    pipeline().setTransfunc(tf,validID);
    validID++;
  }
  m_volumes = Buffer(volumes.size(),volumes.data());

  pipeline().markFrameUpdate();

  // volumes
  pipeline().launchParam("volumes", (RawPointer &)parms().volumes) = (Volume *)m_volumes.data();
  pipeline().launchParam("numVolumes", parms().numVolumes) = (int)m_volumes.size();
  // transfuncs
  // update framebuffer:
  pipeline().launchParam("fbPointer", (RawPointer &)parms().fbPointer) = m_frame->fbPointer;
  pipeline().launchParam("fbDepth", (RawPointer &)parms().fbDepth) = m_frame->fbDepth;
  pipeline().launchParam("accumBuffer", (RawPointer &)parms().accumBuffer) = m_frame->accumBuffer;
  // update DVR params:
  pipeline().launchParam("unitDistance", parms().unitDistance) = 1.f;
  // update renderer params:
  pipeline().launchParam("backgroundColor", parms().backgroundColor) = vec4f(0.f);
}

bool Frame::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint64_t size, uint32_t flags)
{
  return true;
}

void *Frame::map(std::string_view channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  *width = m_frame->width;
  *height = m_frame->height;

  if (channel == "color" || channel == "channel.color") {
    *pixelType = ANARI_UFIXED8_RGBA_SRGB;
    *width = m_frame->width;
    return m_frame->fbPointer;
  } else if (channel == "depth" || channel == "channel.depth") {
    *pixelType = ANARI_FLOAT32;
    return m_frame->fbDepth;
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
  deviceState()->commitBuffer.flush();

  auto cam = *m_camera->getCamera();
  struct {
    vec3f lower_left, horizontal, vertical;
  } screen;
  cam.getScreen(screen.lower_left,screen.horizontal,screen.vertical);

  // update camera:
  pipeline().launchParam("camera.org", parms().camera.org) = cam.getPosition();
  pipeline().launchParam("camera.dir_00", parms().camera.dir_00) = screen.lower_left;
  pipeline().launchParam("camera.dir_du", parms().camera.dir_du) = screen.horizontal / m_size.x;
  pipeline().launchParam("camera.dir_dv", parms().camera.dir_dv) = screen.vertical / m_size.y;

  // update transfuncs:
  std::vector<ex09_anari::Transfunc> transfuncs;
  for (size_t i=0, validID=0; i<m_world->volumes().size(); ++i) {
    auto vol = m_world->volumes()[i];
    auto field = vol->getField();
    if (!field) continue;
    auto tf = vol->getTransfunc();
    transfuncs.push_back({tf->valueRange,tf->rgbaLUT,tf->size});
  }
  m_TFs = Buffer(transfuncs.size(),transfuncs.data());
  pipeline().launchParam("transfuncs", (RawPointer &)parms().transfuncs) = m_TFs.data();

  // lighting
  pipeline().launchParam("ambientColor", parms().ambientColor) = vec3f(1.f);
  pipeline().launchParam("ambientRadiance", parms().ambientRadiance) = 1.f;
  pipeline().launchParam("ambientSamples", parms().ambientSamples) = 2;
  pipeline().launchParam("occlusionDistance", parms().occlusionDistance) = 2.f;

  // update accum:
  pipeline().launchParam("accumID", parms().accumID) = pipeline().frameID;

  // set params:
  SET_LAUNCH_PARAMS(parms());

  // only launch (ANARI takes over the 'present()' part):
  pipeline().launch();

  pipeline().isRunning();
}

} // namespace ex09_anari

DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::Object *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::SpatialField *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::TF1D *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::Group *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::World *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::Camera *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::Renderer *);
DVR_COURSE_ANARI_TYPEFOR_DEFINITION(ex09_anari::Frame *);




