//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <vpusmm.h>
#include <sys/mman.h>
#include <dma-buf.h>
#include <sys/ioctl.h>
#include <unistd.h>


#include "kmb_allocator.h"

#include <iostream>

void *vpu::KmbPlugin::KmbAllocator::lock(void *handle, InferenceEngine::LockOp) noexcept {
    if (_allocatedMemory.find(handle) == _allocatedMemory.end())
        return nullptr;

    return handle;
}

void vpu::KmbPlugin::KmbAllocator::unlock(void *) noexcept {
}

void *vpu::KmbPlugin::KmbAllocator::alloc(size_t size) noexcept {
    long pageSize = getpagesize();
    size_t realSize = size + (size % pageSize ? (pageSize - size % pageSize) : 0);

    auto fd = vpusmm_alloc_dmabuf(realSize, VPUSMMType::VPUSMMTYPE_COHERENT);

    void *out = mmap(nullptr, realSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (out == MAP_FAILED)
        return nullptr;

    MemoryDescriptor memDesc = {
            realSize, //  size
            fd,       //  file descriptor
    };
    _allocatedMemory[out] = memDesc;

    return out;
}

bool vpu::KmbPlugin::KmbAllocator::free(void *handle) noexcept {
    auto memoryIt = _allocatedMemory.find(handle);
    if (memoryIt == _allocatedMemory.end()) {
        return false;
    }

    auto memoryDesc = memoryIt->second;

    auto out = munmap(handle, memoryDesc.size);
    if (out == -1) {
        return false;
    }
    close(memoryDesc.fd);

    _allocatedMemory.erase(handle);

    return true;
}
