//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

void vpux::IE::DepthToSpaceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     int64_t block_size, IE::DepthToSpaceMode mode) {
    build(builder, state, input, block_size, mode, nullptr);
}

void vpux::IE::DepthToSpaceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::IntegerAttr block_size, IE::DepthToSpaceModeAttr mode) {
    build(builder, state, input, block_size, mode, nullptr);
}

void vpux::IE::DepthToSpaceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                     mlir::Value input, mlir::IntegerAttr block_size, IE::DepthToSpaceModeAttr mode) {
    build(builder, state, outType, input, block_size, mode, nullptr);
}

mlir::LogicalResult vpux::IE::DepthToSpaceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DepthToSpaceOpAdaptor depthToSpace(operands, attrs, prop);
    if (mlir::failed(depthToSpace.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(depthToSpace.getInput());
    const auto inType = mlir::cast<mlir::ShapedType>(depthToSpace.getInput().getType()).getElementType();
    const auto block_size = depthToSpace.getBlockSize();
    auto paddedChannels = depthToSpace.getPaddedChannels();

    if (!(inType.isF16() || inType.isF32() || inType.isUnsignedInteger(8) ||
          mlir::isa<mlir::quant::QuantizedType>(inType))) {
        return errorAt(loc, "DepthToSpace only support FP16, FP32, U8, quant data type");
    }

    if (inShape.size() < 3) {
        return errorAt(loc, "Invalid input tensor shape, dimension must be greater than 2.");
    }

    if (block_size <= 0) {
        return errorAt(loc, "Invalid block size {0}, should be greater than zero", block_size);
    }

    if (inShape[Dims4D::Act::C] % (block_size * block_size) != 0) {
        return errorAt(loc, "Invalid block size {0}, which is not divisible by input shape {1}", block_size,
                       inShape[Dims4D::Act::C]);
    }

    if (inShape[Dims4D::Act::C] == mlir::ShapedType::kDynamic) {
        return errorAt(loc, "Input channels dimension is dynamic, cannot infer output shape");
    }
    if (inShape[Dims4D::Act::N] == mlir::ShapedType::kDynamic) {
        return errorAt(loc, "Input batch size dimension is dynamic, cannot infer output shape");
    }

    int64_t paddedIC = 0;
    int64_t paddedOC = 0;

    auto blockSizeSquare = block_size * block_size;
    if (paddedChannels.has_value()) {
        paddedIC = paddedChannels.value().getInput() ? paddedChannels.value().getInput().getInt() : 0;
        paddedOC = paddedChannels.value().getOutput() ? paddedChannels.value().getOutput().getInt() : 0;

        auto unpaddedChannels = inShape[Dims4D::Act::C] - paddedIC;
        if (unpaddedChannels % blockSizeSquare != 0) {
            return errorAt(loc, "Invalid block size {0}, which is not divisible by input shape {1}", block_size,
                           unpaddedChannels);
        }

        if (paddedOC != 0 &&
            (inShape[Dims4D::Act::C] / blockSizeSquare != unpaddedChannels / blockSizeSquare + paddedOC)) {
            return errorAt(loc, "Invalid padded output channels {0}", paddedOC);
        }
    }

    int64_t W_out = inShape[Dims4D::Act::W] == mlir::ShapedType::kDynamic
                            ? mlir::ShapedType::kDynamic
                            : checked_cast<int64_t>(inShape[Dims4D::Act::W] * block_size);
    int64_t H_out = inShape[Dims4D::Act::H] == mlir::ShapedType::kDynamic
                            ? mlir::ShapedType::kDynamic
                            : checked_cast<int64_t>(inShape[Dims4D::Act::H] * block_size);
    int64_t C_out = checked_cast<int64_t>((inShape[Dims4D::Act::C] - paddedIC) / blockSizeSquare + paddedOC);
    int64_t N_out = checked_cast<int64_t>(inShape[Dims4D::Act::N]);

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(depthToSpace.getInput().getType());

    auto [outDesc, outShape] = callOnShapeOf(inputType, [&](const auto& shape) {
        SmallVector<int64_t> outShape{N_out, C_out, H_out, W_out};
        if constexpr (std::is_same_v<std::decay_t<decltype(shape)>, BoundedShape>) {
            SmallVector<int64_t> outBounds{inShape[Dims4D::Act::N],
                                           (inShape[Dims4D::Act::C] - paddedIC) / blockSizeSquare + paddedOC,
                                           static_cast<int64_t>(shape[Dims4D::Act::H].dimValue() * block_size),
                                           static_cast<int64_t>(shape[Dims4D::Act::W].dimValue() * block_size)};
            auto desc =
                    vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace(), BoundsRef(outBounds));
            return std::make_pair(std::move(desc), std::move(outShape));
        } else {
            auto desc = vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace());
            return std::make_pair(std::move(desc), std::move(outShape));
        }
    });

    inferredReturnShapes.emplace_back(outShape, inType, outDesc);
    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::DepthToSpaceOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 1, "Wrong number of operands : {0}", operands.size());
    // when block_size == 1, fold to input itself
    if (getBlockSize() == 1) {
        return getInput();
    }

    return nullptr;
}

mlir::LogicalResult IE::DepthToSpaceOp::verify() {
    if (getBlockSize() <= 0) {
        return errorAt(*this, "Block size should be a positive integer, while it is {0}", getBlockSize());
    }
    return mlir::success();
}
