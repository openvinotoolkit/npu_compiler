//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace vpux {

void insertReorderForInput(mlir::Operation* op, mlir::OpOperand& input, DimsOrder dstOrder,
                           mlir::PatternRewriter& rewriter, Logger log) {
    auto curOrder = DimsOrder::fromValue(input.get());
    log.trace("Insert ReorderOp: curOrder = {0}, dstOrder = {1}", curOrder, dstOrder);

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto reorderOp =
            rewriter.create<IE::ReorderOp>(takeOpLoc(op, StringLiteral("reorder_in_{0}"), input.getOperandNumber()),
                                           input.get(), dstOrder.toAffineMap(rewriter.getContext()));

    log.trace("Redirect input to the new Value");
    input.set(reorderOp.getOutput());
}

IE::ReorderOp insertReorderForOutput(mlir::Operation* op, mlir::Value output, DimsOrder dstOrder,
                                     mlir::PatternRewriter& rewriter, Logger log) {
    auto curOrder = DimsOrder::fromValue(output);
    log.trace("Insert ReorderOp: curOrder = {0}, dstOrder = {1}", curOrder, dstOrder);

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    auto reorderOp = rewriter.create<IE::ReorderOp>(takeOpLoc(op, "reorder_out"), output,
                                                    dstOrder.toAffineMap(rewriter.getContext()));

    log.trace("Redirect output users to the new Value");
    output.replaceAllUsesExcept(reorderOp.getOutput(), llvm::SmallPtrSet<mlir::Operation*, 1>{reorderOp});

    return reorderOp;
}

void changeDimsOrder(mlir::Value val, DimsOrder newOrder, Logger log) {
    const auto origType = mlir::cast<vpux::NDTypeInterface>(val.getType());
    const auto newType = origType.changeDimsOrder(newOrder);

    log.trace("Change Value type to '{0}'", newType);
    val.setType(newType);
}

mlir::FailureOr<vpux::AdjustConvShapeParams> getAdjustConvShapeParameters(IE::ConvolutionOp convOp, mlir::Value filter,
                                                                          ShapeRef outputShape, Logger _log) {
    auto inNDInterface = mlir::dyn_cast<vpux::NDTypeInterface>(convOp.getInput().getType());
    auto inDimOrder = inNDInterface.getDimsOrder();
    auto outNDInterface = mlir::dyn_cast<vpux::NDTypeInterface>(convOp.getOutput().getType());
    auto outDimOrder = outNDInterface.getDimsOrder();
    const auto strides = Shape(parseIntArrayAttr<int64_t>(convOp.getStrides()));
    if (DimsOrder::NHWC != inDimOrder || DimsOrder::NHWC != outDimOrder) {
        _log.trace("The input/output layout should be NHWC, but got {0}/{1}", inDimOrder, outDimOrder);
        return mlir::failure();
    }

    auto isQuantizedType = [](NDTypeInterface ndType) {
        const auto elementType = ndType.getElementType();
        return mlir::isa<mlir::quant::QuantizedType>(elementType);
    };

    auto filterNDInterface = mlir::dyn_cast<vpux::NDTypeInterface>(filter.getType());
    if (isQuantizedType(inNDInterface) || isQuantizedType(filterNDInterface) || isQuantizedType(outNDInterface)) {
        _log.trace("Unsupported Convolution with Quantized Type");
        return mlir::failure();
    }

    auto filterShape = vpux::getShape(filter);
    auto isConst = [](mlir::Value value) {
        auto cst = value.getDefiningOp<Const::DeclareOp>();
        if (nullptr == cst) {
            return false;
        }
        return true;
    };

    auto bias = convOp.getBias();
    if (!isConst(filter) || (bias && !isConst(bias))) {
        _log.trace("Unsupported filter and bias of Convolution is not Constant");
        return mlir::failure();
    }
    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(convOp.getOperation());
    const int64_t alignedInputChannel = iface.getInputChannelAlignment();
    int64_t alignedOutputChannel = iface.getOutputChannelAlignment();
    auto caculateExpandShapeSize = [](ShapeRef shape, int64_t alignedChannel) {
        auto expandShape = shape.toValues();
        expandShape[Dims4D::Act::C] = alignValUp(shape[Dims4D::Act::C], alignedChannel);
        return expandShape.totalSize();
    };

    if ((filterShape[Dims4D::Filter::IC] % alignedInputChannel) == 0 &&
        (filterShape[Dims4D::Filter::OC] % alignedOutputChannel) == 0) {
        _log.trace("The input/output channels are already aligned");
        return mlir::failure();
    }

    auto inputShape = inNDInterface.getShape();

    Shape maybePaddedInputShape(inputShape.raw());
    auto padNum = 0;
    auto wcInDimSize = inputShape[Dims4D::Act::C] * inputShape[Dims4D::Act::W];
    if (wcInDimSize % alignedInputChannel) {
        // If it's a 1x1 convolution, we can pad to the end of input tensor to make it get aligned
        if (filterShape[Dims4D::Filter::KX] != 1 || filterShape[Dims4D::Filter::KY] != 1) {
            _log.trace("The input channel*width ({0}) can't get align factor {1}", wcInDimSize, alignedInputChannel);
            return mlir::failure();
        }
        padNum = (alignValUp(wcInDimSize, alignedInputChannel) - wcInDimSize) / inputShape[Dims4D::Act::C];
        if (padNum <= 0 || padNum >= strides[Dims4D::Strides::X]) {
            _log.trace("Cannot get a aligned shape by padding");
            return mlir::failure();
        }
        maybePaddedInputShape[Dims4D::Act::W] = inputShape[Dims4D::Act::W] + padNum;
    }

    auto wcOutDimSize = outputShape[Dims4D::Act::C] * outputShape[Dims4D::Act::W];
    if (wcOutDimSize % alignedOutputChannel) {
        // We want output channel align to input channel because the compressed CONV's input channel alignment is 4
        // If the output channel is not multiple of alignedInputChannel, it will not work.
        if ((wcOutDimSize % alignedInputChannel) || (alignedOutputChannel % alignedInputChannel)) {
            _log.trace("The output channel*width ({0}) can't get align factor {1}", wcOutDimSize, alignedOutputChannel);
            return mlir::failure();
        }
        alignedOutputChannel = alignedInputChannel;
    }

    auto calcBorrowFactor = [](int64_t channel, int64_t alignedChannel) {
        auto leastAlignedChannel = std::lcm(channel, alignedChannel);
        return (leastAlignedChannel / channel);
    };

    auto borrowIn = calcBorrowFactor(maybePaddedInputShape[Dims4D::Act::C], alignedInputChannel);
    auto borrowOut = calcBorrowFactor(outputShape[Dims4D::Act::C], alignedOutputChannel);
    _log.trace("Input factor {0}, output factor {1}", borrowIn, borrowOut);

    //
    // Promise the input channel align first
    // To reshape the input tensor from DimW, we need promise the new shape's next W index is (stride*N).
    // Because the new convolution's stride is 1.
    //
    auto realInFactor = std::lcm(strides[Dims4D::Strides::X], borrowIn);

    if (realInFactor == 0 || maybePaddedInputShape[Dims4D::Act::W] % realInFactor != 0) {
        _log.trace("Don't have factor {0} in input DimW", realInFactor);
        return mlir::failure();
    }

    if (outputShape[Dims4D::Act::W] % borrowOut) {
        _log.trace("Don't have factor {0} in output DimW", borrowOut);
        return mlir::failure();
    }

    // To promise the new kernel's IC >= originIC * originKX
    //  And the MAX realInFactor is inputShape[Dims4D::Act::W]
    auto newInputDimW = maybePaddedInputShape[Dims4D::Act::W] / realInFactor;
    while (realInFactor < filterShape[Dims4D::Filter::KX] && newInputDimW > 1) {
        auto divisor = vpux::smallestDivisor(newInputDimW);
        realInFactor *= divisor;
        newInputDimW /= divisor;
    }

    auto padBegin = Shape(parseIntArrayAttr<int64_t>(convOp.getPadsBegin()));
    auto padEnd = Shape(parseIntArrayAttr<int64_t>(convOp.getPadsEnd()));

    int64_t leftPading = 0;
    Shape newInputShape(maybePaddedInputShape.raw());
    Shape newOutputShape(outputShape.raw());
    Shape newFilterShape(filterShape.raw());
    int64_t borrowFactor;
    if (filterShape[Dims4D::Filter::KX] == 1) {
        // If KX = 1, the DimC can borrow any dims from DimW
        // Special case to make the kernel size as small as possible
        borrowFactor = std::max(borrowIn, borrowOut);
        newInputShape[Dims4D::Act::W] /= borrowFactor * strides[Dims4D::Strides::X];
        newInputShape[Dims4D::Act::C] *= borrowFactor * strides[Dims4D::Strides::X];

        newFilterShape[Dims4D::Filter::IC] = newInputShape[Dims4D::Act::C];
        newFilterShape[Dims4D::Filter::OC] *= borrowFactor;
        newOutputShape[Dims4D::Act::W] /= borrowFactor;

        leftPading -= (padBegin[Dims4D::PadsBegin::Left] * filterShape[Dims4D::Filter::IC]);
    } else {
        borrowFactor = realInFactor / strides[Dims4D::Strides::X];
        if (borrowFactor < borrowOut) {
            // Output channel not aligned and check Input can borrow
            // If can, allocate new channels
            // If not, let input channel align
            auto outBorrowFact = std::lcm(borrowFactor, borrowOut);
            if ((newInputShape[Dims4D::Act::W] % (outBorrowFact * strides[Dims4D::Strides::X])) == 0 &&
                ((outputShape[Dims4D::Act::W] % outBorrowFact) == 0)) {
                borrowFactor = outBorrowFact;
                realInFactor = borrowFactor * strides[Dims4D::Strides::X];
            }
        }

        if (outputShape[Dims4D::Act::W] % borrowFactor) {
            _log.trace("The outputShape not aligned");
            return mlir::failure();
        }

        newInputShape[Dims4D::Act::W] /= realInFactor;
        newInputShape[Dims4D::Act::C] *= realInFactor;
        //
        // The newFilterIC >= originFilterKX * originFilterIC and newFilterIC = N * stride
        // Generally, the newKX = 2 is enough to cover full origin's calculation.
        // For example:
        //          N H W C
        //   Input: 1x4x4x3
        //  Filter: 1x3x3x3
        //  Stride:   1x2
        //  If we borrow factor 4 from W
        //    NewFilter: 2x3x2x12
        // OC = 0
        //  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
        //   c11 c12 c13 c21 c22 c23 c31 c32 c33  0    0    0
        //   0    0   0   0   0   0   0   0   0   0    0    0
        // OC = 1
        //  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
        //   0    0   0   0   0   0  c11 c12 c13  c21  c22  c23
        //   c31 c32 c33  0   0   0   0   0   0   0    0    0
        //
        // When consider the left and right padding, it need add another kernel to make it work.
        //
        if (padBegin[Dims4D::PadsBegin::Left] == 0 || padEnd[Dims4D::PadsEnd::Right] == 0) {
            newFilterShape[Dims4D::Filter::KX] = 2;
        } else {
            newFilterShape[Dims4D::Filter::KX] = 3;
        }

        if (padBegin[Dims4D::PadsBegin::Left] > 0) {
            leftPading = (realInFactor - padBegin[Dims4D::PadsBegin::Left]) * filterShape[Dims4D::Filter::IC];
        }

        newFilterShape[Dims4D::Filter::IC] = newInputShape[Dims4D::Act::C];
        newFilterShape[Dims4D::Filter::OC] *= borrowFactor;

        newOutputShape[Dims4D::Act::W] /= borrowFactor;
    }
    newOutputShape[Dims4D::Act::C] = newFilterShape[Dims4D::Filter::OC];
    auto newFilterSize = newFilterShape.totalSize();
    _log.trace("The new shape {0}, new filter shape {1}, filter size {2}", newInputShape, newFilterShape,
               newFilterSize);

    auto expandedInputSize = caculateExpandShapeSize(maybePaddedInputShape, alignedInputChannel);
    auto expandedOutputSize = caculateExpandShapeSize(outputShape, alignedOutputChannel);
    auto expandedTotalSize = expandedInputSize + expandedOutputSize;
    const auto cmxMemSize = VPU::getTotalCMXSize(convOp.getOperation());
    const auto elementSize = inNDInterface.getCompactAllocSize().count() / maybePaddedInputShape.totalSize();

    if (expandedTotalSize * elementSize < cmxMemSize.count() || newFilterSize > Byte(1_MB).count()) {
        return mlir::failure();
    }

    // As input channel already aligned, output channel unaligned, it only need slice the data.
    // So filter out the wasted calculation greater than slice
    auto kernelScaled = static_cast<float>(newFilterShape.totalSize()) / static_cast<float>(filterShape.totalSize());
    auto outputTensorScaled = static_cast<float>(expandedOutputSize) / static_cast<float>(outputShape.totalSize());
    if ((filterShape[Dims4D::Filter::IC] % alignedInputChannel) == 0 &&
        (kernelScaled / outputTensorScaled) > alignedOutputChannel) {
        _log.trace("The shape adjust cost greater than expand when input channel already aligned");
        return mlir::failure();
    }

    vpux::AdjustConvShapeParams newParamsAfterAdjust;
    newParamsAfterAdjust.filterShape = std::move(newFilterShape);
    newParamsAfterAdjust.inputShape = std::move(newInputShape);
    newParamsAfterAdjust.outputShape = std::move(newOutputShape);
    newParamsAfterAdjust.borrowFactor = borrowFactor;
    newParamsAfterAdjust.filterPading = leftPading;
    newParamsAfterAdjust.padNum = padNum;

    return newParamsAfterAdjust;
}

int64_t calculateAlignmentFactor(const vpux::NDTypeInterface sliceInType, const vpux::NDTypeInterface sliceOutType) {
    const auto channelAlignment = VPU::NCEInvariant::getAlignment(sliceInType.getElementType());

    const auto sliceInShape = sliceInType.getShape();
    const auto sliceOutShape = sliceOutType.getShape();

    int64_t factor = 1;
    while (factor * sliceInShape[Dims4D::Act::C] % channelAlignment != 0 ||
           factor * sliceOutShape[Dims4D::Act::C] % channelAlignment != 0) {
        factor++;
    }

    return factor;
}

}  // namespace vpux
