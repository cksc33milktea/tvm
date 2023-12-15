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
 *
 * \file pynq_driver.c
 * \brief VTA driver for Zynq SoC boards with Pynq support (see pynq.io).
 */

#include <vta/driver.h>
#include <thread>
#include <time.h>
#include "pynq_driver.h"
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
using namespace std;

typedef std::chrono::high_resolution_clock Clock;



void* VTAMemAlloc1(size_t size, int cached) {
  assert(size <= VTA_MAX_XFER);
  
  //cout<<size<<" "<<cached<<endl;
  // Rely on the pynq-specific cma library
  return cma_alloc(size, cached);
}

void VTAMemFree1(void* buf) {
  // Rely on the pynq-specific cma library
  cma_free(buf);
}

vta_phy_addr_t VTAMemGetPhyAddr1(void* buf) {
  return cma_get_phy_addr(buf);
}

void VTAMemCopyFromHost1(void* dst, const void* src, size_t size) {
  // For SoC-based FPGAs that used shared memory with the CPU, use memcopy()
  memcpy(dst, src, size);
}

void VTAMemCopyToHost1(void* dst, const void* src, size_t size) {
  // For SoC-based FPGAs that used shared memory with the CPU, use memcopy()
  memcpy(dst, src, size);
}

void VTAFlushCache1(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
  // Call the cma_flush_cache on the CMA buffer
  // so that the FPGA can read the buffer data.
  cout<<"flush!"<<endl;
  cma_flush_cache(vir_addr, phy_addr, size);
}

void VTAInvalidateCache1(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
  // Call the cma_invalidate_cache on the CMA buffer
  // so that the host needs to read the buffer data.
  cma_invalidate_cache(vir_addr, phy_addr, size);
}

void *VTAMapRegister(uint32_t addr) {
  // Align the base address with the pages
  uint32_t virt_base = addr & ~(getpagesize() - 1);
  // Calculate base address offset w.r.t the base address
  uint32_t virt_offset = addr - virt_base;
  // Open file and mmap
  uint32_t mmap_file = open("/dev/mem", O_RDWR|O_SYNC);
  return mmap(NULL,
              (VTA_IP_REG_MAP_RANGE + virt_offset),
              PROT_READ|PROT_WRITE,
              MAP_SHARED,
              mmap_file,
              virt_base);
}

void VTAUnmapRegister(void *vta) {
  // Unmap memory
  int status = munmap(vta, VTA_IP_REG_MAP_RANGE);
  assert(status == 0);
}

void VTAWriteMappedReg(void* base_addr, uint32_t offset, uint32_t val) {
  *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset)) = val;
}

uint32_t VTAReadMappedReg(void* base_addr, uint32_t offset) {
  return *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset));
}

class VTADevice {
 public:
  VTADevice() {
    // VTA stage handles
    vta_fetch_handle_ = VTAMapRegister(VTA_FETCH_ADDR);
    vta_load_handle_ = VTAMapRegister(VTA_LOAD_ADDR);
    vta_compute_handle_ = VTAMapRegister(VTA_COMPUTE_ADDR);
    vta_store_handle_ = VTAMapRegister(VTA_STORE_ADDR);
  }

  ~VTADevice() {
    // Close VTA stage handle
    VTAUnmapRegister(vta_fetch_handle_);
    VTAUnmapRegister(vta_load_handle_);
    VTAUnmapRegister(vta_compute_handle_);
    VTAUnmapRegister(vta_store_handle_);
  }

  int Run(vta_phy_addr_t insn_phy_addr,
          uint32_t insn_count,
          uint32_t wait_cycles) {/*
    VTAWriteMappedReg(vta_fetch_handle_, VTA_FETCH_INSN_COUNT_OFFSET, insn_count);
    VTAWriteMappedReg(vta_fetch_handle_, VTA_FETCH_INSN_ADDR_OFFSET, insn_phy_addr);
    VTAWriteMappedReg(vta_load_handle_, VTA_LOAD_INP_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_load_handle_, VTA_LOAD_WGT_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_compute_handle_, VTA_COMPUTE_UOP_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_compute_handle_, VTA_COMPUTE_BIAS_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_store_handle_, VTA_STORE_OUT_ADDR_OFFSET, 0);

    // VTA start
    VTAWriteMappedReg(vta_fetch_handle_, 0x0, VTA_START);
    VTAWriteMappedReg(vta_load_handle_, 0x0, VTA_AUTORESTART);
    VTAWriteMappedReg(vta_compute_handle_, 0x0, VTA_AUTORESTART);
    VTAWriteMappedReg(vta_store_handle_, 0x0, VTA_AUTORESTART);

    // Allow device to respond
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000 };
    nanosleep(&ts, &ts);

    // Loop until the VTA is done
    unsigned t, flag = 0;
    for (t = 0; t < wait_cycles; ++t) {
      flag = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET);
      if (flag == VTA_DONE) break;
      std::this_thread::yield();
    }
    // Report error if timeout
    return t < wait_cycles ? 0 : 1;*/
	return 0;
  }
  
  int conv2d(float* input1,float* input2,float* output,int H,int W,int i_ch,int o_ch,int ksize,int stride, int turn) {
	  
	  int base=100000;
	  
	  auto t1a = Clock::now();
	  
	  float* src1=(float*)VTAMemAlloc1(sizeof(float)*(i_ch*H*W+2*base),0);
	  float* src2=(float*)VTAMemAlloc1(sizeof(float)*(i_ch*o_ch*ksize*ksize+2*base),0);
	  //float* dst1=(float*)VTAMemAlloc1(sizeof(float)*(o_ch*H*W),0);
	  
	  cma_flush_cache(input1, VTAMemGetPhyAddr1(input1), i_ch*H*W*4);
	  cma_flush_cache(input2, VTAMemGetPhyAddr1(input2), i_ch*o_ch*ksize*ksize*4);
	  cma_flush_cache(output, VTAMemGetPhyAddr1(output), o_ch*H*W*4);
	  
	  auto t2a = Clock::now();
	
	std::cout <<turn<< " Alloc+Flush time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2a - t1a).count() << " ns or" \
        <<std::chrono::duration_cast<std::chrono::nanoseconds>(t2a - t1a).count()/1000000<< " ms" << std::endl;
	  
	  
	  auto t11 = Clock::now();
	  
	  VTAMemCopyFromHost1(src1+base,input1,sizeof(float)*(i_ch*H*W));
	  VTAMemCopyFromHost1(src2,input2,sizeof(float)*(i_ch*o_ch*ksize*ksize));
	  
	  auto t22 = Clock::now();
	
	std::cout <<turn<< " Copy input time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t22 - t11).count() << " ns or" \
        <<std::chrono::duration_cast<std::chrono::nanoseconds>(t22 - t11).count()/1000000<< " ms" << std::endl;
		
	uint32_t addr;
	
	if(turn==1)addr=0xA0000000;
	else if(turn==2)addr=0xA0010000;
	else if(turn==3)addr=0xA0020000;
	else if(turn==4)addr=0xA0030000;
	else if(turn==5)addr=0xA0040000;
	else if(turn==6)addr=0xA0050000;
	
	//cout<<addr<<" "<<turn<<endl;
	  
	VTAWriteMappedReg(VTAMapRegister(addr),0x10,VTAMemGetPhyAddr1(src1));
	VTAWriteMappedReg(VTAMapRegister(addr),0x1c,VTAMemGetPhyAddr1(src2));
	VTAWriteMappedReg(VTAMapRegister(addr),0x50,VTAMemGetPhyAddr1(output));
	
	VTAWriteMappedReg(VTAMapRegister(addr),0x28,W);
	VTAWriteMappedReg(VTAMapRegister(addr),0x30,H);
	VTAWriteMappedReg(VTAMapRegister(addr),0x38,i_ch);
	VTAWriteMappedReg(VTAMapRegister(addr),0x40,o_ch);
	VTAWriteMappedReg(VTAMapRegister(addr),0x48,ksize);
	VTAWriteMappedReg(VTAMapRegister(addr),0x5c,stride);
	
	
	VTAWriteMappedReg(VTAMapRegister(addr),0x00,0x01);
	
	auto t111 = Clock::now();
	
	while((VTAReadMappedReg(VTAMapRegister(addr),0x00)&0x00000001)==1){
		usleep(1000*10);
	}
	
	auto t222 = Clock::now();
	
	std::cout <<turn<< " FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t222 - t111).count() << " ns or" \
        <<std::chrono::duration_cast<std::chrono::nanoseconds>(t222 - t111).count()/1000000<< " ms" << std::endl;

	  auto t1 = Clock::now();
	  
	  //VTAMemCopyFromHost1(output,dst1,sizeof(float)*(o_ch*H*W));
	  
	  auto t2 = Clock::now();
	
	//std::cout <<turn<< " Copy output time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns or" \
        <<std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()/1000000<< " ms" << std::endl;
	
	
	auto t1b = Clock::now();
	
	VTAMemFree1(src1);
	VTAMemFree1(src2);
	//VTAMemFree1(dst1);
	
	auto t2b = Clock::now();
	
	std::cout <<turn<< " Free time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2b - t1b).count() << " ns or" \
        <<std::chrono::duration_cast<std::chrono::nanoseconds>(t2b - t1b).count()/1000000<< " ms" << std::endl;
	
    return 0;
  }
  
 private:
  // VTA handles (register maps)
  void* vta_fetch_handle_{nullptr};
  void* vta_load_handle_{nullptr};
  void* vta_compute_handle_{nullptr};
  void* vta_store_handle_{nullptr};
};

VTADeviceHandle VTADeviceAlloc1() {
  return new VTADevice();
}

void VTADeviceFree1(VTADeviceHandle handle) {
  delete static_cast<VTADevice*>(handle);
}


int VTADeviceRun1(VTADeviceHandle handle,
                 vta_phy_addr_t insn_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles) {
  return static_cast<VTADevice*>(handle)->Run(
      insn_phy_addr, insn_count, wait_cycles);
}


void VTADeviceRun_test(VTADeviceHandle device){
	
	
	cout<<"at PYNQ!..."<<endl;
	
	//static_cast<VTADevice*>(device)->add2(input1,input2,output,iterations);
	
	return;//static_cast<VTADevice*>(device)->add2(input1,input2,output,iterations);
}

void VTADeviceRun_test1(VTADeviceHandle device){
	
	
	cout<<"at PYNQ!!..."<<endl;
	
	//static_cast<VTADevice*>(device)->add2(input1,input2,output,iterations);
	
	return;//static_cast<VTADevice*>(device)->add2(input1,input2,output,iterations);
}

void VTADeviceRun_conv2d(VTADeviceHandle device,float* input1,float* input2,float* output,int H,int W,int i_ch,int o_ch,int ksize,int stride,int turn){
	
	//cout<<"conv2d..."<<endl;
	
	static_cast<VTADevice*>(device)->conv2d(input1,input2,output,H, W,i_ch, o_ch,ksize,stride,turn);
	

	return;
}