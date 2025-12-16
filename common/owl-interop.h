// ======================================================================== //
// Copyright 2025-2025 Stefan Zellmann                                      //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

// Import OWL types here--so that infrastructure classes (like Buffer) don't
// depend on owl, but can at the same time interop on the interface (we might
// eventually just try to use OWL on the host, but for now that seems a little
// too complicated):
typedef enum
  {
   OWL_INVALID_TYPE = 0,

   OWL_BUFFER=10,
   /*! a 64-bit int representing the number of elemnets in a buffer */
   OWL_BUFFER_SIZE,
   OWL_BUFFER_ID,
   OWL_BUFFER_POINTER,
   OWL_BUFPTR=OWL_BUFFER_POINTER,

   OWL_GROUP=20,

   /*! implicit variable of type integer that specifies the *index*
     of the given device. this variable type is implicit in the
     sense that it only gets _declared_ on the host, and gets set
     automatically during SBT creation */
   OWL_DEVICE=30,

   /*! texture(s) */
   OWL_TEXTURE=40,
   OWL_TEXTURE_2D=OWL_TEXTURE,


   /* all types that are naively copyable should be below this value,
      all that aren't should be above */
   _OWL_BEGIN_COPYABLE_TYPES = 1000,
   
   
   OWL_FLOAT=1000,
   OWL_FLOAT2,
   OWL_FLOAT3,
   OWL_FLOAT4,

   OWL_INT=1010,
   OWL_INT2,
   OWL_INT3,
   OWL_INT4,
   
   OWL_UINT=1020,
   OWL_UINT2,
   OWL_UINT3,
   OWL_UINT4,
   
   OWL_LONG=1030,
   OWL_LONG2,
   OWL_LONG3,
   OWL_LONG4,

   OWL_ULONG=1040,
   OWL_ULONG2,
   OWL_ULONG3,
   OWL_ULONG4,

   OWL_DOUBLE=1050,
   OWL_DOUBLE2,
   OWL_DOUBLE3,
   OWL_DOUBLE4,
    
   OWL_CHAR=1060,
   OWL_CHAR2,
   OWL_CHAR3,
   OWL_CHAR4,

   /*! unsigend 8-bit integer */
   OWL_UCHAR=1070,
   OWL_UCHAR2,
   OWL_UCHAR3,
   OWL_UCHAR4,

   OWL_SHORT=1080,
   OWL_SHORT2,
   OWL_SHORT3,
   OWL_SHORT4,

   /*! unsigend 8-bit integer */
   OWL_USHORT=1090,
   OWL_USHORT2,
   OWL_USHORT3,
   OWL_USHORT4,

   OWL_BOOL,
   OWL_BOOL2,
   OWL_BOOL3,
   OWL_BOOL4,
   
   /*! just another name for a 64-bit data type - unlike
     OWL_BUFFER_POINTER's (which gets translated from OWLBuffer's
     to actual device-side poiners) these OWL_RAW_POINTER types get
     copied binary without any translation. This is useful for
     owl-cuda interaction (where the user already has device
     pointers), but should not be used for logical buffers */
   OWL_RAW_POINTER=OWL_ULONG,
   OWL_BYTE = OWL_UCHAR,
   // OWL_BOOL = OWL_UCHAR,
   // OWL_BOOL2 = OWL_UCHAR2,
   // OWL_BOOL3 = OWL_UCHAR3,
   // OWL_BOOL4 = OWL_UCHAR4,


   /* matrix formats */
   OWL_AFFINE3F=1300,

   /*! at least for now, use that for buffers with user-defined types:
     type then is "OWL_USER_TYPE_BEGIN+sizeof(elementtype). Note
     that since we always _add_ the user type's size to this value
     this MUST be the last entry in the enum */
   OWL_USER_TYPE_BEGIN=10000
  }
  OWLDataType;

#define OWL_USER_TYPE(userType) ((OWLDataType)(OWL_USER_TYPE_BEGIN+sizeof(userType)))
