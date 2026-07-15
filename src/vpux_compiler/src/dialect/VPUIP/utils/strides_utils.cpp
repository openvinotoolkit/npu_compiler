//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/strides_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

namespace vpux {
namespace VPUIP {

MemDimArr getStridesMemDims(vpux::NDTypeInterface tensorType) {
    const auto elemSize = tensorType.getElemTypeSize();
    const auto memShapes = tensorType.getMemShape().raw();
    const auto memStrides = tensorType.getMemStrides().raw();

    MemDimArr stridesDims;
    for (auto ind : irange(memShapes.size()) | reversed) {
        auto dim = MemDim(ind);
        if (ind == memShapes.size() - 1 && memStrides[ind] != elemSize) {
            stridesDims.push_back(dim);
        } else if (ind != memShapes.size() - 1) {
            const auto prevMemDim = ind + 1;
            if (memStrides[ind] != memStrides[prevMemDim] * memShapes[prevMemDim]) {
                stridesDims.push_back(dim);
            }
        }
    }

    return stridesDims;
}

bool isDDRCopyEfficient(vpux::NDTypeInterface tensorType, config::ArchKind arch) {
    // For one dim strided cases, if contigous width and strides achieve a certain one,
    // some strided performance seems almost same to non-strided.
    // Tests results like the following:
    // if arch == 37XX, the contiguous width should be aligned by 64 bytes
    // if arch == 40XX+, the contiguous width should be aligned by 512 bytes
    // if stride is aligned by contiguous width, the strides are efficient

    const auto elemSize = tensorType.getElemTypeSize();
    const auto memShapes = tensorType.getMemShape().raw();
    const auto memStrides = tensorType.getMemStrides().raw();

    int64_t contiguousWidth = 1, stride = 1;
    size_t stridesDimsNum = 0;
    // Calculate the contiguous width by checking from the last dimension
    // Calculate the strides by checking from the last dimension
    const auto memShapesSize = static_cast<int64_t>(memShapes.size());
    const auto lastDimInd = memShapesSize - 1;
    for (auto ind : irange(memShapesSize) | reversed) {
        if (stridesDimsNum > 1) {
            return false;
        }
        if (ind == lastDimInd && memStrides[ind] != elemSize) {
            contiguousWidth = elemSize.count();
            stride = memStrides[ind].count();
            ++stridesDimsNum;
        } else if (ind != lastDimInd) {
            const auto prevMemDim = ind + 1;
            const auto expectedStride = memStrides[prevMemDim].count() * memShapes[prevMemDim];
            const auto curMemStride = memStrides[ind].count();
            if (curMemStride != expectedStride) {
                contiguousWidth = expectedStride;
                stride = curMemStride;
                ++stridesDimsNum;
            }
        }
    }

    if (stridesDimsNum == 0) {
        return true;
    }

    const auto alignedWidthThreshold =
            arch == config::ArchKind::NPU37XX ? INPUT_DDR_CONTIGUOUS_WIDTH_37XX : INPUT_DDR_CONTIGUOUS_WIDTH_40XX;
    if (contiguousWidth % alignedWidthThreshold != 0) {
        return false;
    }
    if (stride % contiguousWidth != 0) {
        return false;
    }

    return true;
}

// Ignore size-1 dimensions only at the boundaries. A leading of size 1 does not affect the traversed memory region,
// but a middle/trailing size-1 dim is still part of the compactness relation between
// its neighboring dimensions and must preserve the expected stride chain.
bool isEffectivelyStrided(vpux::NDTypeInterface type) {
    const auto memShape = type.getMemShape();
    const auto memStrides = type.getMemStrides();
    const auto elemSize = type.getElemTypeSize();
    const auto rank = memShape.size();

    size_t first = 0;
    while (first < rank && memShape[MemDim(first)] == 1) {
        ++first;
    }
    for (size_t ind{first}; ind < rank; ++ind) {
        const auto dim = MemDim(ind);
        if (ind == rank - 1) {
            if (memStrides[dim] != elemSize) {
                return true;
            }
        } else {
            const auto nextDim = MemDim(ind + 1);
            if (memStrides[dim] != memStrides[nextDim] * memShape[nextDim]) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace VPUIP
}  // namespace vpux
