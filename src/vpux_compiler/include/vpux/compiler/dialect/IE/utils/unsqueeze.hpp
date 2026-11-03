//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

namespace vpux {
namespace IE {

// Unsqueezes the given shape (static or dynamic) by inserting trivial dimensions (1) at the specified axes.
// Axes must be sorted and unique.
// Example: inShape=8x16, axes={0, 2, 4} => outShape=1x8x1x16x1
template <typename D, typename T, template <class> class Tag>
mlir::FailureOr<details::DimValues<D, T, Tag>> unsqueezeShape(mlir::Location loc,
                                                              details::DimValuesRef<D, T, Tag> inShape,
                                                              ArrayRef<int64_t> axes) {
    auto outShape = makeShape(inShape, inShape.size() + axes.size(), 1);

    size_t inInd = 0;
    size_t axesInd = 0;
    for (auto outInd : irange(outShape.size())) {
        if (axesInd < axes.size()) {
            const auto nextAxisInd = checked_cast<size_t>(axes[axesInd]);

            if (nextAxisInd < outInd) {
                return errorAt(loc, "Axis '{0}' occurred twice", nextAxisInd);
            }

            if (nextAxisInd == outInd) {
                outShape[D(outInd)] = 1;
                ++axesInd;
                continue;
            }
        }

        if (inInd < inShape.size()) {
            outShape[D(outInd)] = inShape[D(inInd)];
            ++inInd;
            continue;
        }
    }
    if (inInd != inShape.size() || axesInd != axes.size()) {
        return errorAt(loc, "Inconsistent parameters");
    }

    return outShape;
}

template <typename D, typename T, template <class> class Tag>
mlir::FailureOr<details::DimValues<D, T, Tag>> unsqueezeShape(mlir::Location loc,
                                                              const details::DimValues<D, T, Tag>& inShape,
                                                              ArrayRef<int64_t> axes) {
    return unsqueezeShape(loc, details::DimValuesRef<D, T, Tag>(inShape), axes);
}

template <typename UnsqueezeType>
mlir::FailureOr<SmallVector<int64_t>> getAxes(UnsqueezeType unsqueeze, mlir::Location loc) {
    if (unsqueeze.getAxes() != nullptr && unsqueeze.getAxesValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (unsqueeze.getAxes() == nullptr && !unsqueeze.getAxesValue().has_value()) {
        return errorAt(loc, "Missed axes representation");
    }

    SmallVector<int64_t> axes;
    if (unsqueeze.getAxesValue().has_value()) {
        axes = parseIntArrayAttr<int64_t>(unsqueeze.getAxesValue().value());
    } else {
        auto axesConst = unsqueeze.getAxes().template getDefiningOp<Const::DeclareOp>();
        if (axesConst == nullptr) {
            return errorAt(loc, "Only constant axes are supported");
        }

        const auto axesContent = axesConst.getContent();
        axes = to_small_vector(axesContent.template getValues<int64_t>());
    }

    const auto inType = mlir::cast<mlir::ShapedType>(unsqueeze.getInput().getType());
    const auto inRank = inType.getRank();
    const auto numAxes = checked_cast<int64_t>(axes.size());
    const auto outRank = inRank + numAxes;

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += outRank;
        }
        if (axis < 0 || axis >= outRank) {
            return errorAt(loc, "Axis '{0}' is out of bounds for output rank {1}", axis, outRank);
        }
    }

    std::sort(axes.begin(), axes.end());
    if (std::adjacent_find(axes.begin(), axes.end()) != axes.end()) {
        return errorAt(loc, "Axes must be unique");
    }
    return axes;
}

}  // namespace IE
}  // namespace vpux
