//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace vpux::config {
enum class ArchKind : uint64_t;
}
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

namespace arch60xx {
// Constants
constexpr size_t DMA_MAX_INDICES_LIST_LENGTH =
        65'536;  // The maximum length of the indices list for scatter-gather addressing on NPU6.
constexpr size_t GATHER_DMA_MAX_ELEMENT_SIZE = 2097151;
}  // namespace arch60xx

// indices need to be 32 byte aligned, data type is i64, so the number of indices need to be multiple of 4
constexpr size_t INDICES_ALIGNMENT = 4;

size_t getGatherDMAMaxIndicesListLength(config::ArchKind arch);
size_t getGatherDMAMaxElementSize(config::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
