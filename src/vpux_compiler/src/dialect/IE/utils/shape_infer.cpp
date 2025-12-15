//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/range.hpp"

#include <iterator>
#include <numeric>

using namespace vpux;

vpux::DimsOrder vpux::IE::inferOrder(vpux::NDTypeInterface lhsType, vpux::NDTypeInterface rhsType) {
    return lhsType.getRank() > rhsType.getRank() ? lhsType.getDimsOrder() : rhsType.getDimsOrder();
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
        SmallVector<ArrayRef<int64_t>> shapes{shape1, shape2};
        return broadcastEltwiseShape(shapes, broadcastType, loc);
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

                auto in1Shape = in1ShapeIter != outShape.rend() ? *in1ShapeIter : 0;
                auto in2Shape = in2ShapeIter != shapes[i].rend() ? *in2ShapeIter : 0;
                *outShapeRIter = (in1Shape == mlir::ShapedType::kDynamic || in2Shape == mlir::ShapedType::kDynamic)
                                         ? mlir::ShapedType::kDynamic
                                         : std::max(in1Shape, in2Shape);

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

mlir::FailureOr<SmallVector<int64_t>> vpux::IE::constInputToData(mlir::Location loc, mlir::Value value) {
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
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType());
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
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType());
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
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType());
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

// If it's enough to adjust aligned channel just borrowing from width,
// we don't need to change the shape size of height
mlir::FailureOr<Shape> vpux::IE::getShapeCastExpandedShapeWithMinimalDimChange(mlir::Operation* operation,
                                                                               ShapeRef originShape, Logger log) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType());
    const auto sizeToAlign = VPU::NCEInvariant::getAlignment(inputType.getElementType());
    auto channelSize = originShape[Dims4D::Act::C];
    auto wcSize = originShape[Dims4D::Act::W] * channelSize;
    if (wcSize % sizeToAlign != 0) {
        log.trace("Input shape is not divisible to align on dimW and dim C for op {0} at {1}", operation->getName(),
                  operation->getLoc());
        return mlir::failure();
    }

    return Shape({originShape[Dims4D::Act::N], sizeToAlign, originShape[Dims4D::Act::H], wcSize / sizeToAlign});
}

SmallVector<int64_t> vpux::IE::dispatchBounds(const mlir::Value operand) {
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(operand.getType())) {
        return to_small_vector(boundedType.getBounds());
    }

    return to_small_vector(getShape(operand));
}

SmallVector<int64_t> vpux::IE::inferOutputBounds(const mlir::Value lhs, const mlir::Value rhs,
                                                 const ShapeRef outputShape,
                                                 const IE::AutoBroadcastType autoBroadcast) {
    if (outputShape.isStatic()) {
        return {};
    }
    const auto lhsBounds = dispatchBounds(lhs);
    const auto rhsBounds = dispatchBounds(rhs);
    auto outBoundsRes = IE::broadcastEltwiseShape(lhsBounds, rhsBounds, autoBroadcast, lhs.getLoc());
    if (mlir::failed(outBoundsRes)) {
        return {};
    }
    return outBoundsRes.value();
}
