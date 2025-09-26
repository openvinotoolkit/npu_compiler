//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

bool IE::anyIDUAutopadCandidate(ArrayRef<mlir::Operation*> ops) {
    return llvm::any_of(ops, [](mlir::Operation* op) {
        auto convOp = mlir::dyn_cast_or_null<IE::ConvolutionOp>(op);
        if (convOp == nullptr) {
            return false;
        }
        if (!convOp->hasAttr(VPU::INPUT_PADDING_ATTR_NAME)) {
            return false;
        }
        auto inputPadding =
                parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(convOp->getAttr(VPU::INPUT_PADDING_ATTR_NAME)));
        if (inputPadding.size() != 4) {
            return false;
        }
        return inputPadding[Dims4D::Act::C.ind()] > 0;
    });
}
