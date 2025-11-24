//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/workload_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/padding_native.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

#include <cassert>

using namespace vpux;

VPUIP::WorkloadComponents VPUIP::minimizeWorkloadSize(const VPUIP::WorkloadComponents& workload,
                                                      ArrayRef<int64_t> kernelSize, bool minimizeChannels) {
    constexpr size_t dimX = 0;
    constexpr size_t dimY = 1;
    constexpr size_t dimZ = 2;

    const auto& initialInStart = workload.inStart;
    const auto& initialInEnd = workload.inEnd;
    const auto& initialOutStart = workload.outStart;
    const auto& initialOutEnd = workload.outEnd;

    assert(initialInStart.size() == 3 && "Expected inStart to have three elements");
    assert(initialInEnd.size() == 3 && "Expected inEnd to have three elements");
    assert(initialOutStart.size() == 3 && "Expected outStart to have three elements");
    assert(initialOutEnd.size() == 3 && "Expected outEnd to have three elements");
    assert(initialInStart[dimX] <= initialInEnd[dimX] && "Incorrect offsets for input dim X");
    assert(initialInStart[dimY] <= initialInEnd[dimY] && "Incorrect offsets for input dim Y");
    assert(initialInStart[dimZ] <= initialInEnd[dimZ] && "Incorrect offsets for input dim Z");
    assert(initialOutStart[dimX] <= initialOutEnd[dimX] && "Incorrect offsets for output dim X");
    assert(initialOutStart[dimY] <= initialOutEnd[dimY] && "Incorrect offsets for output dim Y");
    assert(initialOutStart[dimZ] <= initialOutEnd[dimZ] && "Incorrect offsets for output dim Z");

    // Initialize the end dimensions to match the start ones, so that a single element is initially configured
    auto newInEnd = initialInStart;
    auto newOutEnd = initialOutStart;

    // Deduce the minimal spatial dimensions of the input, to produce a single output pixel, as well as the padding
    // configuration for this workload. The right / bottom padding can be removed if the minimal input size does not
    // depend on this padding to exist
    int64_t minX = 1, minY = 1;
    if (kernelSize.size() == 2) {
        minX = kernelSize[Dims4D::Kernel::X.ind()];
        minY = kernelSize[Dims4D::Kernel::Y.ind()];
    }
    const auto initialPadLeft = workload.pad.getLeftPad();
    const auto initialPadRight = workload.pad.getRightPad();
    const auto initialPadTop = workload.pad.getTopPad();
    const auto initialPadBottom = workload.pad.getBottomPad();
    minX -= initialPadLeft;
    minY -= initialPadTop;

    const auto inputIncludesEndPad = [](int64_t start, int64_t end, int64_t minSize) {
        const auto sizeWithoutPad = end - start + 1;
        return sizeWithoutPad < minSize;
    };
    auto newPadRight = initialPadRight;
    if (inputIncludesEndPad(initialInStart[dimX], initialInEnd[dimX], minX)) {
        minX -= initialPadRight;
    } else {
        newPadRight = 0;
    }
    auto newPadBottom = initialPadBottom;
    if (inputIncludesEndPad(initialInStart[dimY], initialInEnd[dimY], minY)) {
        minY -= initialPadBottom;
    } else {
        newPadBottom = 0;
    }

    auto newPad = VPU::Padding(initialPadLeft, newPadRight, initialPadTop, newPadBottom);

    newInEnd[dimX] += minX - 1;
    newInEnd[dimY] += minY - 1;

    if (minimizeChannels) {
        // The minimum number of input / output channels is generally 16. However, when IDU / ODU autopad is used,
        // the number of channels can be even smaller. As the workload is configured to work with fewer channels
        // than 16, the newly introduced workload must use the same number of channels as the existing one
        constexpr auto channelAlignment = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
        const auto inputZSize = initialInEnd[dimZ] - initialInStart[dimZ] + 1;
        const auto outputZSize = initialOutEnd[dimZ] - initialOutStart[dimZ] + 1;
        newInEnd[dimZ] += (inputZSize % channelAlignment != 0) ? inputZSize - 1 : channelAlignment - 1;
        newOutEnd[dimZ] += (outputZSize % channelAlignment != 0) ? outputZSize - 1 : channelAlignment - 1;
    } else {
        newInEnd[dimZ] = initialInEnd[dimZ];
        newOutEnd[dimZ] = initialOutEnd[dimZ];
    }

    return VPUIP::WorkloadComponents{workload.inStart, std::move(newInEnd), workload.outStart, std::move(newOutEnd),
                                     newPad};
}
