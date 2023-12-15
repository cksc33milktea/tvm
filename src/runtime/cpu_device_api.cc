/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file cpu_device_api.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <cstdlib>
#include <cstring>

#include "workspace_pool.h"

#include <iostream>
using namespace std;

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include <assert.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#if defined(__arm__) || defined(__aarch64__)
#include <libxlnk_cma.h>
#else
void* cma_alloc(size_t size, int cached);
void cma_free(void* buf);
uint32_t cma_get_phy_addr(void* buf);
void cma_flush_cache(void* buf, unsigned int phys_addr, int size);
void cma_invalidate_cache(void* buf, unsigned int phys_addr, int size);
#endif

void *VTAMapRegister(uint32_t addr);
void VTAUnmapRegister(void *vta);
void VTAWriteMappedReg(void* base_addr, uint32_t offset, uint32_t val);
uint32_t VTAReadMappedReg(void* base_addr, uint32_t offset);

/*! \brief VTA configuration register start value */
#define VTA_START 0x1
/*! \brief VTA configuration register auto-restart value */
#define VTA_AUTORESTART 0x81
/*! \brief VTA configuration register done value */
#define VTA_DONE 0x1

#ifdef __cplusplus
}
#endif



size_t total=0;

namespace tvm {
namespace runtime {
class CPUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final {}
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    void* ptr;
	
	if(nbytes<400000000)total+=nbytes;
#if _MSC_VER
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr) throw std::bad_alloc();
#elif defined(__ANDROID__) && __ANDROID_API__ < 17
    ptr = memalign(alignment, nbytes);
    if (ptr == nullptr) throw std::bad_alloc();
#else
    // posix_memalign is available in android ndk since __ANDROID_API__ >= 17
    //int ret = posix_memalign(&ptr, alignment, nbytes);
	int ret=0;
	
	if(nbytes>400000000)ptr=malloc(nbytes);
	else ptr=cma_alloc(nbytes,1);
    if (ret != 0) throw std::bad_alloc();
#endif
	//cout<<"Alloc : "<<ptr<<" "<<nbytes<<" "<<total<<endl;
    return ptr;
  }

  void FreeDataSpace(Device dev, void* ptr) final {
	  //cout<<"free : "<<ptr<<endl;
	  cma_free(ptr);
#if _MSC_VER
    //_aligned_free(ptr);
#else
    //free(ptr);
#endif
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {}

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* data) final;

  static CPUDeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new CPUDeviceAPI();
    return inst;
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
  }
};

struct CPUWorkspacePool : public WorkspacePool {
  CPUWorkspacePool() : WorkspacePool(kDLCPU, CPUDeviceAPI::Global()) {}
};

void* CPUDeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  return dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()->AllocWorkspace(dev, size);
}

void CPUDeviceAPI::FreeWorkspace(Device dev, void* data) {
  dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()->FreeWorkspace(dev, data);
}

TVM_REGISTER_GLOBAL("device_api.cpu").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = CPUDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});
}  // namespace runtime
}  // namespace tvm
