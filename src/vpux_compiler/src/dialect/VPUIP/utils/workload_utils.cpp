//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/workload_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/padding_native.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

VPUIP::WorkloadComponents VPUIP::reduceToOneOutputPixel(const VPUIP::WorkloadComponents& workload,
                                                        ArrayRef<int64_t> kernelSize) {
    constexpr size_t dimX = 0;
    constexpr size_t dimY = 1;
    constexpr size_t dimZ = 2;

    const auto& initialInStart = workload.inStart;
    const auto& initialInEnd = workload.inEnd;
    const auto& initialOutStart = workload.outStart;
    const auto& initialOutEnd = workload.outEnd;

    // Initialize the end dimensions to match the start ones, so that a single element is initially configured
    auto newInEnd = initialInStart;
    auto newOutEnd = initialOutStart;

    // Deduce the minimal spatial dimensions of the input, to produce a single output pixel
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

    // All input and output channels need to be used for the pixel computation to be correct
    newInEnd[dimZ] = initialInEnd[dimZ];
    newOutEnd[dimZ] = initialOutEnd[dimZ];

    return VPUIP::WorkloadComponents{workload.inStart, std::move(newInEnd), workload.outStart, std::move(newOutEnd),
                                     newPad};
}
