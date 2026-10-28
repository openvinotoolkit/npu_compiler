//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTNEGATIVEPADTOSLICE
#define GEN_PASS_DEF_CONVERTNEGATIVEPADTOSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertNegativePadToSlicePass
//

class ConvertNegativePadToSlicePass final :
        public IE::impl::ConvertNegativePadToSliceBase<ConvertNegativePadToSlicePass> {
public:
    explicit ConvertNegativePadToSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertNegativePadToSlicePass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    func.walk([&](IE::PadOp padOp) {
        _log.trace("Found IE::PadOp Operation '{0}'", padOp->getLoc());

        // Skip dynamic tensors
        if (IE::hasDynamicTensors(padOp)) {
            return;
        }

        auto padsBegin = vpux::IE::extractPads(padOp.getPadsBeginAttrAttr(), _log);
        if (mlir::failed(padsBegin)) {
            return;
        }

        auto padsEnd = vpux::IE::extractPads(padOp.getPadsEndAttrAttr(), _log);
        if (mlir::failed(padsEnd)) {
            return;
        }

        const auto padsBeginValue = padsBegin.value();
        const auto padsEndValue = padsEnd.value();

        const auto inputType = mlir::cast<vpux::NDTypeInterface>(padOp.getInput().getType());
        const auto inputShape = inputType.getShape().raw();

        VPUX_THROW_UNLESS(padsBeginValue.size() == inputShape.size() && padsEndValue.size() == inputShape.size(),
                          "`IE::PadOp` {0} shape size {1} mismatch with input size {2}", padOp.getLoc(),
                          padsBeginValue.size(), inputShape.size());

        // Separate negative (slice) and non-negative (keep in pad) values
        SmallVector<int64_t> sliceOffsets(inputShape.size(), 0);
        SmallVector<int64_t> sliceSizes(inputShape.begin(), inputShape.end());
        SmallVector<int64_t> newPadsBegin(inputShape.size(), 0);
        SmallVector<int64_t> newPadsEnd(inputShape.size(), 0);
        bool hasNegativePad = false;
        bool hasPositivePad = false;

        for (size_t i = 0; i < inputShape.size(); ++i) {
            if (padsBeginValue[i] < 0) {
                sliceOffsets[i] = -padsBeginValue[i];
                sliceSizes[i] -= sliceOffsets[i];
                hasNegativePad = true;
            } else if (padsBeginValue[i] > 0) {
                newPadsBegin[i] = padsBeginValue[i];
                hasPositivePad = true;
            }

            if (padsEndValue[i] < 0) {
                sliceSizes[i] += padsEndValue[i];
                hasNegativePad = true;
            } else if (padsEndValue[i] > 0) {
                newPadsEnd[i] = padsEndValue[i];
                hasPositivePad = true;
            }
        }

        // No negative padding, nothing to do
        if (!hasNegativePad) {
            return;
        }

        // Validate slice parameters
        for (size_t i = 0; i < inputShape.size(); ++i) {
            if (sliceOffsets[i] >= inputShape[i]) {
                _log.trace("Invalid slice offset {0} at dimension {1}, exceeds input dimension size {2}",
                           sliceOffsets[i], i, inputShape[i]);
                return;
            }
            if (sliceSizes[i] <= 0) {
                _log.trace("Invalid slice size {0} at dimension {1}, negative padding exceeds input dimension",
                           sliceSizes[i], i);
                return;
            }
        }

        // Create SliceOp for negative padding
        mlir::OpBuilder builder(padOp);
        auto sliceOffsetsAttr = getIntArrayAttr(&ctx, sliceOffsets);
        auto sliceSizesAttr = getIntArrayAttr(&ctx, sliceSizes);

        auto sliceOp = IE::SliceOp::create(builder, takeOpLoc(padOp, "slice"), padOp.getInput(), sliceOffsetsAttr,
                                           sliceSizesAttr);

        if (!hasPositivePad) {
            // All pads were negative, replace Pad with just Slice
            padOp.replaceAllUsesWith(sliceOp.getResult());
        } else {
            // Mixed case: create new PadOp with non-negative pads only
            auto newPadsBeginAttr = getIntArrayAttr(&ctx, newPadsBegin);
            auto newPadsEndAttr = getIntArrayAttr(&ctx, newPadsEnd);

            auto newPadOp = IE::PadOp::create(builder, padOp->getLoc(), sliceOp.getResult(), nullptr, nullptr, nullptr,
                                              newPadsBeginAttr, newPadsEndAttr, padOp.getPadValueAttrAttr(),
                                              padOp.getModeAttr(), padOp.getOutputPaddingAttr(),
                                              padOp.getInputPaddingAttr(), nullptr, nullptr);
            padOp.replaceAllUsesWith(newPadOp.getOutput());
        }
        padOp->erase();
    });
}

}  // namespace

//
// createConvertNegativePadToSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertNegativePadToSlicePass(Logger log) {
    return std::make_unique<ConvertNegativePadToSlicePass>(log);
}
