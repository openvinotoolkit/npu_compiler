//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/concat_utils.hpp"

using namespace vpux;
using namespace VPU;

mlir::DenseSet<int64_t> vpux::VPU::getConcatAxes(VPU::ConcatOp concat) {
    mlir::DenseSet<int64_t> concatAxes;
    const auto staticOffsets = concat.getStaticOffsets();
    if (staticOffsets.has_value()) {
        const auto offsets = parseIntArrayOfArrayAttr<int64_t>(staticOffsets.value());
        for (auto& offset : offsets) {
            for (size_t axis = 0; axis < offset.size(); ++axis) {
                if (offset[axis] != 0) {
                    concatAxes.insert(axis);
                }
            }
        }
    } else {
        const auto concatAttr = concat.getPerAxis();
        VPUX_THROW_UNLESS(concatAttr.has_value(),
                          "ConcatOp at '{0}' has neither static offsets nor per axis attribute");
        const auto concatAxis = concatAttr.value().getAxis().getValue().getSExtValue();
        concatAxes.insert(concatAxis);
    }

    return concatAxes;
}
