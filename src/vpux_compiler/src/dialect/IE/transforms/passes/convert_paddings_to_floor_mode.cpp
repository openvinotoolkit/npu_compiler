//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTPADDINGSTOFLOORMODE
#define GEN_PASS_DEF_CONVERTPADDINGSTOFLOORMODE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Normaly the kernel size is smaller than input size with 2 dimensions:
// For 4d NCHW input kernel is 2d XY
// For 5d NCDHW input kernel is 3d XYZ
const size_t INPUT_AND_KERNEL_SIZE_DIFF = 2;

//
// ConvertPaddingsToFloorModePass
//

class ConvertPaddingsToFloorModePass final :
        public IE::impl::ConvertPaddingsToFloorModeBase<ConvertPaddingsToFloorModePass> {
public:
    explicit ConvertPaddingsToFloorModePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    template <class ConcreteOp>
    void updatePoolOperation(ConcreteOp op);

    template <class ConcreteOp>
    void updateAvgPoolOperation(ConcreteOp op);

    template <class ConcreteOp>
    void updateConvOperation(ConcreteOp op);
};

//
// safeRunOnFunc
//

void ConvertPaddingsToFloorModePass::safeRunOnFunc() {
    auto func = getOperation();

    const auto callback = [this](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<IE::MaxPoolOp>([this](IE::MaxPoolOp op) {
                    updatePoolOperation<IE::MaxPoolOp>(op);
                })
                .Case<IE::MaxPool8Op>([this](IE::MaxPool8Op op) {
                    updatePoolOperation<IE::MaxPool8Op>(op);
                })
                .Case<IE::AvgPoolOp>([this](IE::AvgPoolOp op) {
                    updateAvgPoolOperation<IE::AvgPoolOp>(op);
                })
                .Case<IE::AvgPool16Op>([this](IE::AvgPool16Op op) {
                    updateAvgPoolOperation<IE::AvgPool16Op>(op);
                })
                .Case<IE::ConvolutionOp>([this](IE::ConvolutionOp op) {
                    updateConvOperation<IE::ConvolutionOp>(op);
                })
                .Case<IE::GroupConvolutionOp>([this](IE::GroupConvolutionOp op) {
                    updateConvOperation<IE::GroupConvolutionOp>(op);
                });
    };

    func.walk(callback);
}

//
// updateOperation
//

int64_t inferInputSizeFloor(int64_t outputSize, int64_t padBegin, int64_t padEnd, int64_t kernel, int64_t stride) {
    return ((outputSize - 1) * stride) - padBegin - padEnd + kernel;
}

void cvtPaddingsToFloorMode(int64_t inputSize, int64_t outputSize, int64_t kernel, int64_t stride, int64_t& padBegin,
                            int64_t& padEnd) {
    const auto inputSizeFloor = inferInputSizeFloor(outputSize, padBegin, padEnd, kernel, stride);

    padEnd = padEnd + (inputSizeFloor - inputSize);
}

void cvtPaddingsToFloorMode(ShapeRef inShape, ShapeRef outShape, ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides,
                            ArrayRef<int64_t> dilations, MutableArrayRef<int64_t> padsBegin,
                            MutableArrayRef<int64_t> padsEnd) {
    const size_t ksdpSize = inShape.size() - INPUT_AND_KERNEL_SIZE_DIFF;
    VPUX_THROW_UNLESS(inShape.size() == outShape.size(), "Wrong input/output shape");
    VPUX_THROW_UNLESS(kernel.size() == ksdpSize && strides.size() == ksdpSize && dilations.size() == ksdpSize &&
                              padsBegin.size() == ksdpSize && padsEnd.size() == ksdpSize && ksdpSize >= 2,
                      "Wrong kernel/strides/dilations/padsBegin/padsEnd");

    for (size_t idx = 0; idx < ksdpSize; idx++) {
        auto inputDim = Dim(idx + INPUT_AND_KERNEL_SIZE_DIFF);

        auto kernelAxis = kernel[idx];

        auto strideAxis = strides[idx];

        auto dilationAxis = dilations[idx];

        auto& padBeginAxis = padsBegin[idx];

        auto& padEndAxis = padsEnd[idx];

        cvtPaddingsToFloorMode(inShape[inputDim], outShape[inputDim], kernelAxis * dilationAxis - (dilationAxis - 1),
                               strideAxis, padBeginAxis, padEndAxis);
    }
}

template <class ConcreteOp>
void ConvertPaddingsToFloorModePass::updatePoolOperation(ConcreteOp op) {
    const auto inShape = getShape(op.getInput());
    const auto outShape = getShape(op.getOutput());

    const auto kernel = parseIntArrayAttr<int64_t>(op.getKernelSize());
    const auto strides = parseIntArrayAttr<int64_t>(op.getStrides());
    SmallVector<int64_t> dilatation(kernel.size(), 1);
    if (auto maxPool8 = mlir::dyn_cast<IE::MaxPool8Op>(op.getOperation())) {
        dilatation = parseIntArrayAttr<int64_t>(maxPool8.getDilations());
    }
    auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBegin());
    auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEnd());

    cvtPaddingsToFloorMode(inShape, outShape, kernel, strides, dilatation, padsBegin, padsEnd);
    for (size_t idx = 0; idx < padsEnd.size(); idx++) {
        padsEnd[idx] = std::max<int64_t>(padsEnd[idx], 0);
    }

    const auto newPadsBeginAttr = getIntArrayAttr(op.getContext(), padsBegin);
    const auto newPadsEndAttr = getIntArrayAttr(op.getContext(), padsEnd);

    op.setPadsBeginAttr(newPadsBeginAttr);
    op.setPadsEndAttr(newPadsEndAttr);
    op.setRoundingTypeAttr(IE::RoundingTypeAttr::get(op.getContext(), IE::RoundingType::FLOOR));
}

template <>
void ConvertPaddingsToFloorModePass::updatePoolOperation(IE::AvgPool16Op op) {
    const auto inShape = getShape(op.getInput());
    const auto outShape = getShape(op.getOutput());

    const auto kernel = parseIntArrayAttr<int64_t>(op.getKernelSize());
    const auto strides = parseIntArrayAttr<int64_t>(op.getStrides());
    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
    auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBegin());
    auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEnd());

    cvtPaddingsToFloorMode(inShape, outShape, kernel, strides, dilations, padsBegin, padsEnd);
    for (size_t idx = 0; idx < padsEnd.size(); idx++) {
        padsEnd[idx] = std::max<int64_t>(padsEnd[idx], 0);
    }

    const auto newPadsBeginAttr = getIntArrayAttr(op.getContext(), padsBegin);
    const auto newPadsEndAttr = getIntArrayAttr(op.getContext(), padsEnd);

    op.setPadsBeginAttr(newPadsBeginAttr);
    op.setPadsEndAttr(newPadsEndAttr);
    op.setRoundingTypeAttr(IE::RoundingTypeAttr::get(op.getContext(), IE::RoundingType::FLOOR));
}

template <class ConcreteOp>
void ConvertPaddingsToFloorModePass::updateAvgPoolOperation(ConcreteOp op) {
    auto padsEndOriginal = parseIntArrayAttr<int64_t>(op.getPadsEnd());
    updatePoolOperation<ConcreteOp>(op);
    auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBegin());
    auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEnd());
    auto isZero = [](auto val) {
        return val == 0;
    };
    bool noPads = llvm::all_of(padsBegin, isZero) && llvm::all_of(padsEndOriginal, isZero);
    // Inside reference implementation the padded area produced by ceil is consider or not to the division,
    // base on original all 0 padded values, both for begin and end padding.
    if (noPads && (padsEnd != padsEndOriginal)) {
        auto excludePad = mlir::UnitAttr::get(op.getContext());
        op.setExcludePadsAttr(excludePad);
    }
}

template <class ConcreteOp>
void ConvertPaddingsToFloorModePass::updateConvOperation(ConcreteOp op) {
    const auto inShape = getBoundedShape(op.getInput());
    const auto filterShape = getShape(op.getFilter());
    const auto outShape = getBoundedShape(op.getOutput());

    SmallVector<int64_t> kernel;

    for (size_t idx = INPUT_AND_KERNEL_SIZE_DIFF; idx < filterShape.size(); idx++) {
        auto kernelAxis = Dim(idx);
        kernel.push_back(filterShape[kernelAxis]);
    }

    const auto strides = parseIntArrayAttr<int64_t>(op.getStrides());
    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
    auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBegin());
    auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEnd());

    cvtPaddingsToFloorMode(inShape, outShape, kernel, strides, dilations, padsBegin, padsEnd);
    for (size_t i = 0; i < padsEnd.size(); ++i) {
        padsEnd[i] = std::max<int64_t>(padsEnd[i], 0);
    }

    const auto newPadsBeginAttr = getIntArrayAttr(op.getContext(), padsBegin);
    const auto newPadsEndAttr = getIntArrayAttr(op.getContext(), padsEnd);

    op.setPadsBeginAttr(newPadsBeginAttr);
    op.setPadsEndAttr(newPadsEndAttr);
}

}  // namespace

//
// createConvertPaddingsToFloorModePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPaddingsToFloorModePass(Logger log) {
    return std::make_unique<ConvertPaddingsToFloorModePass>(log);
}
