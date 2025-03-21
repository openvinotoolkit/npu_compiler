//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LogicalResult.h>

#include <iterator>
#include <numeric>

using namespace vpux;

bool vpux::IE::isBroadcastable(int64_t d0, int64_t d1) {
    return d0 == 1 || d1 == 1 || d0 == d1;
}

mlir::OpFoldResult vpux::IE::reifyDim(mlir::OpBuilder builder, mlir::Value value, mlir::RankedTensorType type,
                                      size_t idx, std::optional<mlir::Location> loc) {
    if (type.isDynamicDim(idx)) {
        return builder.createOrFold<mlir::tensor::DimOp>(loc.value_or(value.getLoc()), value, idx);
    }
    return builder.getIndexAttr(type.getDimSize(idx));
}

mlir::OpFoldResult vpux::IE::reifyDim(mlir::OpBuilder builder, mlir::Value value, size_t idx,
                                      std::optional<mlir::Location> loc) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    return IE::reifyDim(builder, value, type, idx, loc);
}

mlir::FailureOr<SmallVector<int64_t>> vpux::IE::broadcastEltwiseShape(ArrayRef<int64_t> shape1,
                                                                      ArrayRef<int64_t> shape2,
                                                                      AutoBroadcastType broadcastType,
                                                                      mlir::Location loc) {
    if (broadcastType == IE::AutoBroadcastType::NONE_OR_EXPLICIT) {
        if (shape1 != shape2) {
            return errorAt(loc, "Input shapes must be equal in case BroadcastType is NONE");
        }

        return to_small_vector(shape1);
    } else if (broadcastType == IE::AutoBroadcastType::NUMPY) {
        SmallVector<int64_t> outShape(std::max(shape1.size(), shape2.size()), 0);

        auto in1ShapeIter = shape1.rbegin();
        auto in2ShapeIter = shape2.rbegin();

        for (auto outShapeRIter = outShape.rbegin(); outShapeRIter != outShape.rend(); ++outShapeRIter) {
            if (in1ShapeIter != shape1.rend() && in2ShapeIter != shape2.rend()) {
                if (!isBroadcastable(*in1ShapeIter, *in2ShapeIter)) {
                    return errorAt(loc, "Got non broadcastable dimensions pair : '{0}' and {1}'", *in1ShapeIter,
                                   *in2ShapeIter);
                }
            }

            auto in1Shape = in1ShapeIter != shape1.rend() ? *in1ShapeIter : 0;
            auto in2Shape = in2ShapeIter != shape2.rend() ? *in2ShapeIter : 0;
            *outShapeRIter = (in1Shape == mlir::ShapedType::kDynamic || in2Shape == mlir::ShapedType::kDynamic)
                                     ? mlir::ShapedType::kDynamic
                                     : std::max(in1Shape, in2Shape);

            if (in1ShapeIter != shape1.rend()) {
                ++in1ShapeIter;
            }
            if (in2ShapeIter != shape2.rend()) {
                ++in2ShapeIter;
            }
        }

        return outShape;
    }

    return errorAt(loc, "Unsupported BroadcastType '{0}'", broadcastType);
}

mlir::FailureOr<SmallVector<int64_t>> vpux::IE::broadcastEltwiseShape(ArrayRef<ArrayRef<int64_t>> shapes,
                                                                      AutoBroadcastType broadcastType,
                                                                      mlir::Location loc) {
    if (shapes.size() < 2) {
        return errorAt(loc, "Number of input shapes must be equal or greater than 2");
    }

    if (broadcastType == vpux::IE::AutoBroadcastType::NONE_OR_EXPLICIT) {
        for (size_t i = 1; i < shapes.size(); ++i) {
            if (shapes[0] != shapes[i]) {
                return errorAt(loc, "Input shapes must be equal in case BroadcastType is NONE");
            }
        }

        return to_small_vector(shapes[0]);
    }

    size_t rank = shapes[0].size();
    size_t biggerSize = 0;
    for (size_t i = 0; i < shapes.size(); ++i) {
        if (rank < shapes[i].size()) {
            rank = shapes[i].size();
            biggerSize = i;
        }
    }

    SmallVector<int64_t> outShape(rank, 0);
    for (size_t i = 0; i < outShape.size(); ++i) {
        *(outShape.rbegin() + i) = *(shapes[biggerSize].rbegin() + i);
    }

    for (size_t i = 0; i < shapes.size(); ++i) {
        if (i != biggerSize) {
            auto in1ShapeIter = outShape.rbegin();
            auto in2ShapeIter = shapes[i].rbegin();

            for (auto outShapeRIter = outShape.rbegin(); outShapeRIter != outShape.rend(); ++outShapeRIter) {
                if (in1ShapeIter != outShape.rend() && in2ShapeIter != shapes[i].rend()) {
                    if (!isBroadcastable(*in1ShapeIter, *in2ShapeIter)) {
                        return errorAt(loc, "Got non broadcastable dimensions pair : '{0}' and {1}'", *in1ShapeIter,
                                       *in2ShapeIter);
                    }
                }

                *outShapeRIter = std::max(in1ShapeIter != outShape.rend() ? *in1ShapeIter : 0,
                                          in2ShapeIter != shapes[i].rend() ? *in2ShapeIter : 0);

                if (in1ShapeIter != outShape.rend()) {
                    ++in1ShapeIter;
                }
                if (in2ShapeIter != shapes[i].rend()) {
                    ++in2ShapeIter;
                }
            }
        }
    }

    return outShape;
}

mlir::FailureOr<SmallVector<mlir::OpFoldResult>> vpux::IE::reifyMatMulTensors(mlir::OpBuilder& builder,
                                                                              mlir::Value input1, mlir::Value input2,
                                                                              bool transposeA, bool transposeB,
                                                                              mlir::Location loc) {
    const auto type1 = mlir::cast<mlir::RankedTensorType>(input1.getType());
    const auto type2 = mlir::cast<mlir::RankedTensorType>(input2.getType());

    const auto shape1 = type1.getShape();
    const auto shape2 = type2.getShape();

    // Step 1: Apply transpositions if needed
    auto getTransposedShape = [&](ArrayRef<int64_t> shape, bool transpose) {
        if (transpose && shape.size() >= 2) {
            // Assume the default dimensions order is used. It means the H and W dimensions are the last two elements in
            // the array, so we can swap them to transpose the tensor
            SmallVector<int64_t> transposedShape(shape.begin(), shape.end());
            std::swap(transposedShape[shape.size() - 2], transposedShape[shape.size() - 1]);
            return transposedShape;
        }
        return SmallVector<int64_t>(shape.begin(), shape.end());
    };

    auto shape1Transposed = getTransposedShape(shape1, transposeA);
    auto shape2Transposed = getTransposedShape(shape2, transposeB);

    // Step 2: Unsqueeze 1D tensors
    // If rank of the first input is equal to 1, it is always unsqueezed to 2D tensor row vector
    // If rank of the second input is equal to 1, it is always unsqueezed to 2D tensor column vector
    auto unsqueeze1D = [](ArrayRef<int64_t> shape, bool isRowVector) {
        if (shape.size() == 1) {
            if (isRowVector) {
                return SmallVector<int64_t>{1, shape[0]};
            } else {
                return SmallVector<int64_t>{shape[0], 1};
            }
        }
        return SmallVector<int64_t>(shape.begin(), shape.end());
    };

    shape1Transposed = unsqueeze1D(shape1Transposed, true);
    shape2Transposed = unsqueeze1D(shape2Transposed, false);

    // Step 3: Align ranks by unsqueezing from the left
    std::pair<uint32_t, uint32_t> alignmentAxesCnt{0, 0};
    auto alignRanks = [&alignmentAxesCnt](SmallVector<int64_t>& shape1, SmallVector<int64_t>& shape2) {
        while (shape1.size() < shape2.size()) {
            ++alignmentAxesCnt.first;
            shape1.insert(shape1.begin(), 1);
        }
        while (shape2.size() < shape1.size()) {
            ++alignmentAxesCnt.second;
            shape2.insert(shape2.begin(), 1);
        }
    };

    alignRanks(shape1Transposed, shape2Transposed);

    // Step 4: Broadcast batch dimensions
    SmallVector<mlir::OpFoldResult> outDims;
    for (size_t i = 0; i < shape1Transposed.size() - 2; ++i) {
        if (shape1Transposed[i] == 1) {
            outDims.push_back(IE::reifyDim(builder, input2, type2, i, loc));
        } else if (shape2Transposed[i] == 1 || shape1Transposed[i] == shape2Transposed[i]) {
            outDims.push_back(IE::reifyDim(builder, input1, type1, i, loc));
        } else {
            return errorAt(loc, "Incompatible batch dimensions: '{0}' and '{1}'", shape1Transposed[i],
                           shape2Transposed[i]);
        }
    }

    // Step 5: Determine the output matrix dimensions, taking into account transposition and rank alignment if applied
    // result H dim is equal to input1.H
    outDims.push_back(IE::reifyDim(builder, input1, type1,
                                   shape1Transposed.size() - (transposeA ? 1 : 2) - alignmentAxesCnt.first, loc));
    // result W dim is equal to input2.W
    outDims.push_back(IE::reifyDim(builder, input2, type2,
                                   shape2Transposed.size() - (transposeB ? 2 : 1) - alignmentAxesCnt.second, loc));

    return outDims;
}

mlir::FailureOr<SmallVector<mlir::OpFoldResult>> vpux::IE::reifyEltwiseTensors(mlir::OpBuilder& builder,
                                                                               mlir::Value input1, mlir::Value input2,
                                                                               IE::AutoBroadcastType broadcastType,
                                                                               mlir::Location loc) {
    const auto type1 = mlir::cast<mlir::RankedTensorType>(input1.getType());
    const auto type2 = mlir::cast<mlir::RankedTensorType>(input2.getType());

    const auto shape1 = type1.getShape();
    const auto shape2 = type2.getShape();

    if (broadcastType == IE::AutoBroadcastType::NONE_OR_EXPLICIT) {
        if (shape1 != shape2) {
            return errorAt(loc, "Input shapes must be equal in case BroadcastType is NONE");
        }

        const auto outRank = shape1.size();
        SmallVector<mlir::OpFoldResult> outDims;
        outDims.reserve(outRank);
        auto rankRange = irange(outRank);
        std::transform(rankRange.begin(), rankRange.end(), std::back_inserter(outDims), [&](auto i) {
            return IE::reifyDim(builder, input1, type1, i, loc);
        });
        return outDims;
    } else if (broadcastType == IE::AutoBroadcastType::NUMPY) {
        const auto in1Rank = shape1.size();
        const auto in2Rank = shape2.size();

        auto in1ShapeIter = shape1.rbegin();
        auto in2ShapeIter = shape2.rbegin();

        const auto outRank = std::max(shape1.size(), shape2.size());
        SmallVector<mlir::OpFoldResult> outDims(outRank);

        for (auto i : irange(outRank)) {
            if (in1ShapeIter != shape1.rend() && in2ShapeIter != shape2.rend()) {
                if (!isBroadcastable(*in1ShapeIter, *in2ShapeIter)) {
                    return errorAt(loc, "Got non broadcastable dimensions pair : '{0}' and {1}'", *in1ShapeIter,
                                   *in2ShapeIter);
                }
            }

            if (in1ShapeIter == shape1.rend() || (in2ShapeIter != shape2.rend() && (*in1ShapeIter == 1))) {
                outDims[outRank - i - 1] = IE::reifyDim(builder, input2, type2, in2Rank - i - 1, loc);
            } else {
                VPUX_THROW_UNLESS(in1ShapeIter != shape1.rend(), "Failed to broadcast shapes: {0}, {1} at {2}", shape1,
                                  shape2, loc);
                outDims[outRank - i - 1] = IE::reifyDim(builder, input1, type1, in1Rank - i - 1, loc);
            }

            if (in1ShapeIter != shape1.rend()) {
                ++in1ShapeIter;
            }
            if (in2ShapeIter != shape2.rend()) {
                ++in2ShapeIter;
            }
        }

        return outDims;
    }

    return errorAt(loc, "Unsupported BroadcastType '{0}'", broadcastType);
}

namespace {

// 3d: [batch, channels, columns] -> 1 spatial dimension
// 4d: [batch, channels, rows, columns] -> 2 spatial dimensions
// 5d: [batch, channels, depth, rows, columns] -> 3 spatial dimensions
// Subtract 2 to exclude batch and channels.
int64_t calculateMul(const int64_t dim, const ArrayRef<int64_t> strides) {
    const int64_t spatialDim = dim - 2;
    VPUX_THROW_UNLESS(spatialDim >= 0 && spatialDim < checked_cast<int64_t>(strides.size()),
                      "Cannot get stride by index {0}", dim);
    return strides[spatialDim];
}

int64_t calculateAddend(int64_t dim, const ArrayRef<int64_t> kernelSize, const ArrayRef<int64_t> strides,
                        const ArrayRef<int64_t> padBegin, const ArrayRef<int64_t> padEnd) {
    const int64_t spatialDim = dim - 2;
    VPUX_THROW_UNLESS(spatialDim >= 0 && spatialDim < checked_cast<int64_t>(kernelSize.size()),
                      "Cannot get kernel size by index {0}", dim);
    VPUX_THROW_UNLESS(spatialDim >= 0 && spatialDim < checked_cast<int64_t>(strides.size()),
                      "Cannot get stride by index {0}", dim);
    VPUX_THROW_UNLESS(spatialDim >= 0 && spatialDim < checked_cast<int64_t>(padBegin.size()),
                      "Cannot get pad begin by index {0}", dim);
    VPUX_THROW_UNLESS(spatialDim >= 0 && spatialDim < checked_cast<int64_t>(padEnd.size()),
                      "Cannot get pad end by index {0}", dim);
    return kernelSize[spatialDim] - strides[spatialDim] - padBegin[spatialDim] - padEnd[spatialDim];
}

};  // namespace

mlir::FailureOr<SmallVector<mlir::OpFoldResult>> vpux::IE::reifyConvPoolTensors(
        mlir::OpBuilder& builder, mlir::Value input, mlir::Value output, ArrayRef<int64_t> kernelSize,
        ArrayRef<int64_t> strides, ArrayRef<int64_t> padBegin, ArrayRef<int64_t> padEnd, mlir::Location loc) {
    const auto inputShapedType = mlir::cast<mlir::ShapedType>(input.getType());
    const auto outputShapedType = mlir::cast<mlir::ShapedType>(output.getType());

    SmallVector<mlir::OpFoldResult> shapes;
    for (const auto dim : llvm::seq<int64_t>(0, outputShapedType.getRank())) {
        if (!outputShapedType.isDynamicDim(dim)) {
            // Static dim: Return IntegerAttr.
            shapes.push_back(builder.getIndexAttr(inputShapedType.getDimSize(dim)));
        } else {
            // Dynamic dim: Return Value.
            // in_x = kernel_x + stride_x * (out_x - 1) - pad_begin_x - pad_end_x
            // in_x = kernel_x + stride_x * out_x - stride_x - pad_begin_x - pad_end_x
            // multiplier = stride_x
            // addend = kernel_x - stride_x - pad_begin_x - pad_end_x
            const auto inputMul = calculateMul(dim, strides);
            const auto dimOp = builder.createOrFold<mlir::tensor::DimOp>(loc, input, dim);

            const auto applyMul = [&](mlir::Value value) {
                if (inputMul > 1) {
                    mlir::Value constOp = builder.createOrFold<mlir::arith::ConstantIndexOp>(loc, inputMul);
                    return builder.createOrFold<mlir::arith::DivSIOp>(loc, value, constOp);
                }
                return value;
            };
            const auto afterMul = applyMul(dimOp);

            const auto addend = calculateAddend(dim, kernelSize, strides, padBegin, padEnd);
            const auto applyAddend = [&](mlir::Value value) {
                if (addend != 0) {
                    mlir::Value constOp = builder.createOrFold<mlir::arith::ConstantIndexOp>(loc, addend);
                    return builder.createOrFold<mlir::arith::SubIOp>(loc, value, constOp);
                }
                return value;
            };
            const auto afterAddend = applyAddend(afterMul);
            shapes.push_back(getValueOrCreateConstantIndexOp(builder, loc, afterAddend));
        }
    }

    return shapes;
}

mlir::FailureOr<SmallVector<int64_t>> vpux::IE::constInputToData(mlir::Location loc, const mlir::Value& value) {
    if (value == nullptr) {
        return errorAt(loc, "Target shape was not provided");
    }

    auto valueConst = value.getDefiningOp<Const::DeclareOp>();
    if (valueConst == nullptr) {
        return mlir::failure();
    }

    const auto valueContent = valueConst.getContent();
    return to_small_vector(valueContent.getValues<int64_t>());
}

mlir::FailureOr<SmallVector<int64_t>> getFactors(int64_t total, size_t num) {
    if (total < 0) {
        return mlir::failure();
    }

    if (num == 1) {
        return SmallVector<int64_t>({total});
    }
    if (num > 2) {
        return mlir::failure();
    }
    for (int64_t i = static_cast<int64_t>(sqrt(total)); i >= 1; i--) {
        if (total % i == 0) {
            return SmallVector<int64_t>({total / i, i});
        }
    }
    return mlir::failure();
}

// Reorganize the shape to make dimensions align
// when the C needs alignment,
// satisfy the C dimension and divide the remaining size for other dimensions (H and W) as evenly as possible
//      e.g., [1, 3, 512, 512], new expanded shape is [1, 16, 256, 192], H >= W
// when the W needs alignment (some operations with specific layout)
// satisfy the C and W dimensions and put the remaining size on H
//      e.g., [1, 3, 512, 512], new expanded shape is [1, 16, 3072, 16]
// N dimension is never changed
//
// If the total size is not divisible by all the required alignment, return failure
//      e.g., [1, 3, 22, 22], 3*22*22 is not divisible by 16, return failure
mlir::FailureOr<Shape> vpux::IE::getShapeCastExpandedShape(mlir::Operation* operation, ShapeRef expandedShape,
                                                           ShapeRef unExpandedShape, Logger log) {
    if (operation == nullptr) {
        return mlir::failure();
    }

    if (unExpandedShape.empty()) {
        return mlir::failure();
    }
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType());
    const auto sizeToAlign = std::max(VPU::NCEInvariant::getAlignment(inputType.getElementType()),
                                      VPU::NCEInvariant::getAlignment(outputType.getElementType()));
    const auto totalSize = unExpandedShape.totalSize();

    auto newExpandedShape = Shape(expandedShape.size(), 1);
    llvm::DenseSet<int64_t> dimsToAlign;
    if (unExpandedShape[Dims4D::Act::C] % sizeToAlign == 0) {
        // if the original channel dimension was aligned, keep it
        newExpandedShape[Dims4D::Act::C] = sizeToAlign;
        dimsToAlign.insert(Dims4D::Act::C.ind());
    }
    auto inOrder = inputType.getDimsOrder();
    auto outOrder = outputType.getDimsOrder();
    if (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NCHW &&
        (unExpandedShape[Dims4D::Act::W] % sizeToAlign == 0)) {
        // if the original width dimension needs to align and is already aligned, keep it
        newExpandedShape[Dims4D::Act::W] = sizeToAlign;
        dimsToAlign.insert(Dims4D::Act::W.ind());
    }
    for (auto expandMap : enumerate(expandedShape)) {
        if (expandMap.value() != unExpandedShape[Dim(expandMap.index())]) {
            // other dimensions to expand
            newExpandedShape[Dim(expandMap.index())] = sizeToAlign;
            dimsToAlign.insert(expandMap.index());
        }
    }

    auto totalSizeToAlign = checked_cast<int64_t>(std::pow(sizeToAlign, dimsToAlign.size()));
    if (totalSize % totalSizeToAlign != 0) {
        log.trace("Unable to adjust the input shape for op {0} at {1}", operation->getName(), operation->getLoc());
        return mlir::failure();
    }
    const auto remainingSize = totalSize / totalSizeToAlign;
    auto factors = getFactors(remainingSize, unExpandedShape.size() - 1 - dimsToAlign.size());
    if (mlir::failed(factors)) {
        log.trace("Input shape is not divisible to align for op {0} at {1}", operation->getName(), operation->getLoc());
        return mlir::failure();
    }

    size_t factorIndex = 0;
    for (auto index : irange<size_t>(1, newExpandedShape.size())) {
        if (dimsToAlign.contains(index)) {
            continue;
        }

        newExpandedShape[Dim(index)] = factors.value()[factorIndex];
        factorIndex++;
    }
    return newExpandedShape;
}

mlir::FailureOr<Shape> vpux::IE::getShapeCastExpandedShapeInDimC(mlir::Operation* operation, ShapeRef originShape,
                                                                 Logger log) {
    if (originShape.empty()) {
        return mlir::failure();
    }
    const auto inputType = operation->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto sizeToAlign = VPU::NCEInvariant::getAlignment(inputType.getElementType());
    const auto totalSize = originShape.totalSize();

    if (totalSize % sizeToAlign) {
        log.trace("Input shape is not divisible to {0} for op {1} at {2}", sizeToAlign, operation->getName(),
                  operation->getLoc());
        return mlir::failure();
    }

    const auto remainingSize = totalSize / sizeToAlign;
    auto factors = getFactors(remainingSize, originShape.size() - 2);  // Except DimC and DimN
    if (mlir::failed(factors)) {
        log.trace("Input shape is not divisible to align for op {0} at {1}", operation->getName(), operation->getLoc());
        return mlir::failure();
    }

    auto hwShape = factors.value();
    Shape newExpandedShape = {1, sizeToAlign, hwShape[0], hwShape[1]};
    return newExpandedShape;
}

mlir::FailureOr<Shape> vpux::IE::getShapeCastExpandedShapeKeepDimC(mlir::Operation* operation, ShapeRef originShape,
                                                                   Logger log) {
    const auto inputType = operation->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto sizeToAlign = VPU::NCEInvariant::getAlignment(inputType.getElementType());
    auto channelSize = originShape[Dims4D::Act::C];

    // Least Common Multiple
    auto newChannelSize = std::lcm(channelSize, sizeToAlign);

    auto totalSize = inputType.getShape().totalSize();
    auto wcSize = originShape[Dims4D::Act::W] * channelSize;
    // If it's enough to adjust aligned channel just borrowing from weight,
    // we don't need to change the shape size of height
    if (wcSize % newChannelSize == 0) {
        return Shape(
                {originShape[Dims4D::Act::N], newChannelSize, originShape[Dims4D::Act::H], wcSize / newChannelSize});
    } else if (totalSize % newChannelSize == 0) {
        auto factors = getFactors(totalSize / newChannelSize, 2);
        if (mlir::failed(factors)) {
            log.trace("Input shape is not divisible to align for op {0} at {1}", operation->getName(),
                      operation->getLoc());
            return mlir::failure();
        }
        auto hwShape = factors.value();
        return Shape({originShape[Dims4D::Act::N], newChannelSize, hwShape[0], hwShape[1]});
    }

    return mlir::failure();
}

//  Handle the total size can't divisible by the required alignment case.
//  This function will minimize expansion and divide the remaining size for other dimensions (H and W) as evenly as
//  possible
//  e.g. input shape is [1, 1, 289, 289], new expanded shape is [1, 289, 17, 17]
//  input shape is [1, 1, 1384, 27], new expanded shape is [1, 519, 12, 6]
mlir::FailureOr<Shape> vpux::IE::getShapeCastExpandedShapeCanNotAlign(mlir::Operation* operation, ShapeRef inputShape,
                                                                      Logger log) {
    if (operation == nullptr) {
        return mlir::failure();
    }

    if (inputShape.empty()) {
        return mlir::failure();
    }
    const auto inputType = operation->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto sizeToAlign = VPU::NCEInvariant::getAlignment(inputType.getElementType());
    const auto totalSize = inputShape.totalSize();

    if (totalSize % sizeToAlign == 0) {
        log.trace("The shape of op {0} at {1} can get C align", operation->getName(), operation->getLoc());
        return mlir::failure();
    }

    auto factors = vpux::getPrimeFactors(totalSize);

    if (factors.empty()) {
        log.trace("Can't shapeCast for input shape {0}", inputShape);
        return mlir::failure();
    }

    auto newExpandedShape = Shape(inputShape.size(), 1);

    if (factors.size() == 1) {
        newExpandedShape[Dims4D::Act::C] = factors[0];
    } else if (factors.size() == 2) {
        newExpandedShape[Dims4D::Act::C] = factors[1];
        newExpandedShape[Dims4D::Act::H] = factors[0];
    } else {
        // Try to split some data on HW as much as possible to keep hardware utilization.
        // eg. input shape is [1, 1, 1384, 27] the factors will be [2, 2, 2, 3, 3, 3, 173]
        //  | W | H |       C        |
        // [| 2,| 2,| 2, 3, 3, 3, 173|]
        //    \  /    /
        //     \/    /
        //     /\   /
        //    |  \ /
        //  | W | H |      C      |
        // [| 2,| 4,| 3, 3, 3, 173|]
        //    \  /    /
        //     \/    /
        //     /\   /
        //    |  \ /
        //  | W | H |     C    |
        // [| 4,| 6,| 3, 3, 173|]
        //    \  /    /
        //     \/    /
        //     /\   /
        //    |  \ /
        //  | W | H |   C   |
        // [| 6,|12,| 3, 173|]
        // The new expanded shape will be [1, 173x3, 12, 6]
        auto shapeBegin = factors.begin();
        auto shapeBeginLimit = factors.end() - 3;
        while ((shapeBegin != shapeBeginLimit) && ((*shapeBegin) * (*(shapeBegin + 2)) < 16)) {
            *(shapeBegin + 2) = (*shapeBegin) * (*(shapeBegin + 2));
            shapeBegin++;
        }

        auto preCalculateChannel =
                std::accumulate(shapeBegin + 2, factors.end(), (int64_t)1, std::multiplies<int64_t>());

        while ((shapeBegin != shapeBeginLimit) && (preCalculateChannel > VPU::NCEInvariant::VPU_DIMENSION_LIMIT)) {
            preCalculateChannel = preCalculateChannel / (*(shapeBegin + 2));
            *(shapeBegin + 2) = (*shapeBegin) * (*(shapeBegin + 2));
            shapeBegin++;
        }

        newExpandedShape[Dims4D::Act::W] = *shapeBegin;
        newExpandedShape[Dims4D::Act::H] = *(shapeBegin + 1);
        newExpandedShape[Dims4D::Act::C] = preCalculateChannel;
    }

    if (newExpandedShape == inputShape) {
        // Not need reshape.
        return mlir::failure();
    }

    return newExpandedShape;
}
