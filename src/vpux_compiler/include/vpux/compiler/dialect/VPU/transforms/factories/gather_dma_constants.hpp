//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {
namespace VPU {

namespace arch40xx {
// Constants
constexpr size_t DMA_MAX_INDICES_LIST_LENGTH =
        65'536;  // The maximum length of the indices list for scatter-gather addressing on NPU4.
constexpr size_t GATHER_DMA_MAX_ELEMENT_SIZE = 4096;
}  // namespace arch40xx
namespace arch50xx {
// Constants
constexpr size_t DMA_MAX_INDICES_LIST_LENGTH =
        65'536;  // The maximum length of the indices list for scatter-gather addressing on NPU5.
constexpr size_t GATHER_DMA_MAX_ELEMENT_SIZE = 4096;
}  // namespace arch50xx

size_t getGatherDMAMaxIndicesListLength(config::ArchKind arch);
size_t getGatherDMAMaxElementSize(config::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
