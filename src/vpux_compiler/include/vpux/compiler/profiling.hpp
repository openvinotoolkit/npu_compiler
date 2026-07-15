//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/icompiler.hpp"

#include <vector>

namespace vpux {

std::vector<ze::ze_profiling_layer_info> getLayerInfoImpl(const uint8_t* blobData, uint64_t blobSize,
                                                          const uint8_t* profData, uint64_t profSize);
std::vector<ze::ze_profiling_task_info> getTaskInfoImpl(const uint8_t* blobData, uint64_t blobSize,
                                                        const uint8_t* profData, uint64_t profSize);

}  // namespace vpux
