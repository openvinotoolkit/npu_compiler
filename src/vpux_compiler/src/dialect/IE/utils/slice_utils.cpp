//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux {
namespace IE {

DimArr getDiffInOutSizeDims(ShapeRef inShape, ShapeRef outShape) {
    VPUX_THROW_UNLESS(inShape.size() == outShape.size(),
                      "The size of the input '{0}' and output '{1}' tensors does not match", inShape.size(),
                      outShape.size());
    SmallVector<Dim> diffInOutSizeDims;
    const auto ioShapes = zip(inShape, outShape);
    for (const auto& ioShape : ioShapes | indexed) {
        const auto inSize = std::get<0>(ioShape.value());
        const auto outSize = std::get<1>(ioShape.value());
        if (inSize != outSize) {
            diffInOutSizeDims.push_back(Dim(ioShape.index()));
        }
    }
    return diffInOutSizeDims;
}

std::optional<vpux::Dim> getSingleDiffAxis(ShapeRef inShape, ShapeRef outShape) {
    const auto layerAxes = getDiffInOutSizeDims(inShape, outShape);
    return layerAxes.size() == 1 ? layerAxes.front() : std::optional<vpux::Dim>{};
}

SmallVector<uint64_t> getSliceAxes(IE::SliceOp sliceOp) {
    auto sliceInShape = getShape(sliceOp.getSource());
    auto sizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());

    SmallVector<uint64_t> sliceAxes;
    for (size_t dimIdx = 0; dimIdx < sizes.size(); dimIdx++) {
        if (sliceInShape[Dim(dimIdx)] != sizes[dimIdx]) {
            sliceAxes.push_back(static_cast<uint64_t>(dimIdx));
        }
    }

    return sliceAxes;
}

std::optional<bool> slicesHaveOverlap(ArrayRef<IE::SliceOp> slicesOp, const Dim& dim) {
    if (slicesOp.size() == 1) {
        return false;
    }

    auto vectorsHaveOverlap = [](std::pair<int64_t, int64_t> previous, std::pair<int64_t, int64_t> next) {
        return std::max(previous.first, next.first) <= std::min(previous.second, next.second);
    };

    SmallVector<std::pair<int64_t, int64_t>> startEnds;
    int64_t rank = 0;
    for (auto slice : slicesOp) {
        if (slice == nullptr) {
            return std::nullopt;
        }

        const auto sliceRank = mlir::cast<NDTypeInterface>(slice.getSource().getType()).getRank();
        if (rank == 0) {
            rank = sliceRank;
        } else if (rank != sliceRank) {
            return std::nullopt;
        }

        const auto sliceOffsets = parseIntArrayAttr<int64_t>(slice.getStaticOffsets());
        const auto sliceSizes = parseIntArrayAttr<int64_t>(slice.getStaticSizes());
        std::pair<int64_t, int64_t> startEnd{sliceOffsets[dim.ind()],
                                             sliceOffsets[dim.ind()] + sliceSizes[dim.ind()] - 1};
        for (auto& element : startEnds) {
            if (vectorsHaveOverlap(element, startEnd)) {
                return true;
            }
        }
        startEnds.push_back(startEnd);
    }

    return false;
}

}  // namespace IE
}  // namespace vpux
