//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

bool isBroadcastable(int64_t d0, int64_t d1) {
    return d0 == 1 || d1 == 1 || d0 == d1;
}

}  // namespace

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

            *outShapeRIter = std::max(in1ShapeIter != shape1.rend() ? *in1ShapeIter : 0,
                                      in2ShapeIter != shape2.rend() ? *in2ShapeIter : 0);

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
    } else {
        size_t rank = shapes[0].size();
        for (size_t i = 1; i < shapes.size(); ++i) {
            rank = std::max(rank, shapes[i].size());
        }

        SmallVector<int64_t> outShape(rank, 0);
        for (size_t i = 0; i < outShape.size(); ++i) {
            *(outShape.rbegin() + i) = *(shapes[0].rbegin() + i);
        }

        for (size_t i = 1; i < shapes.size(); ++i) {
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

        return outShape;
    }

    return errorAt(loc, "Unsupported BroadcastType '{0}'", broadcastType);
}

mlir::FailureOr<SmallVector<int64_t>> vpux::IE::constInputToData(mlir::Location loc, const mlir::Value& value) {
    if (value == nullptr) {
        return errorAt(loc, "Target shape was not provided");
    }

    auto valueConst = value.getDefiningOp<Const::DeclareOp>();
    if (valueConst == nullptr) {
        return errorAt(loc, "Only constant input is supported");
    }

    const auto valueContent = valueConst.content();
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
    for (int64_t i = sqrt(total); i >= 1; i--) {
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
    const auto inputType = operation->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto sizeToAlign = VPU::NCEInvariant::getAlignment(inputType.getElementType());
    const auto totalSize = unExpandedShape.totalSize();

    auto newExpandedShape = Shape(expandedShape.size(), 1);
    llvm::DenseSet<int64_t> dimsToAlign;
    if (unExpandedShape[Dims4D::Act::C] % sizeToAlign == 0) {
        // if the original channel dimension was aligned, keep it
        newExpandedShape[Dims4D::Act::C] = sizeToAlign;
        dimsToAlign.insert(Dims4D::Act::C.ind());
    }
    auto inOrder = inputType.getDimsOrder();
    auto outOrder = operation->getResult(0).getType().cast<vpux::NDTypeInterface>().getDimsOrder();
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
    newExpandedShape[Dims4D::Act::N] = unExpandedShape[Dims4D::Act::N];
    for (auto index : irange<size_t>(1, newExpandedShape.size())) {
        if (dimsToAlign.contains(index)) {
            continue;
        }

        newExpandedShape[Dim(index)] = factors.getValue()[factorIndex];
        factorIndex++;
    }
    return newExpandedShape;
}

bool vpux::IE::isShapeCompatibleWithODUPermute(const ShapeRef shape, const int64_t alignment) {
    if (shape.size() != 4) {
        return false;
    }
    if (shape[Dims4D::Act::N] != 1) {
        return false;
    }
    const auto tensorSizeZ = shape[Dims4D::Act::W];
    return tensorSizeZ % alignment == 0;
}
