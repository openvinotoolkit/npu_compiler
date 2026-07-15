//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "buffer.hpp"
#include "buffer_metadata.hpp"
#include "function.hpp"
#include "npu_interpreter_runtime/npu_vm_runtime.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace intel_npu::vm {

struct MemRef {
    const void* basePtr = nullptr;
    const void* data = nullptr;
    int64_t offset = 0;
    int64_t dimsCount = 0;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    explicit MemRef(int64_t dims): dimsCount(dims), sizes(dims, 0), strides(dims, 0) {
    }
};

std::optional<BufferHandle> extractMemRef(npu_vm_runtime_mem_ref_handle_t handle, BufferManager& bufferManager,
                                          std::unordered_map<BufferHandle, BufferMetadata>& metadata,
                                          Permission permission, const FuncParamResType& type,
                                          const std::vector<uint16_t>& typeByteSizes);

}  // namespace intel_npu::vm

using npu_vm_runtime_mem_ref = intel_npu::vm::MemRef;
