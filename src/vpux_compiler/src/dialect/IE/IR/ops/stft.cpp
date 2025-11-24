//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::STFTOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    IE::STFTOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto signalType = mlir::cast<mlir::ShapedType>(op.getSignal().getType());
    const auto signalShape = signalType.getShape();
    const auto elemType = signalType.getElementType();

    int64_t frameSize = 0;
    int64_t frameStep = 0;

    auto extractParam = [&](std::optional<int64_t> attr, mlir::Value operand,
                            const std::string& name) -> mlir::LogicalResult {
        int64_t& target = (name == "frame_size") ? frameSize : frameStep;

        if (attr.has_value()) {
            target = attr.value();
        } else if (operand) {
            if (auto constOp = mlir::dyn_cast_or_null<Const::DeclareOp>(operand.getDefiningOp())) {
                const auto content = constOp.getContent();
                if (!content.isSplat()) {
                    return errorAt(loc, "STFT {0} must be a splat", name);
                }
                target = content.getSplatValue<int64_t>();
            } else {
                return errorAt(loc, "STFT {0} must be a constant", name);
            }
        } else {
            return errorAt(loc, "STFT {0} must be specified", name);
        }
        return mlir::success();
    };

    if (mlir::failed(extractParam(op.getFrameSizeAttr(), op.getFrameSize(), "frame_size")) ||
        mlir::failed(extractParam(op.getFrameStepAttr(), op.getFrameStep(), "frame_step"))) {
        return mlir::failure();
    }

    auto outputShape = to_small_vector(signalShape);
    const auto signalLength = signalShape.back();
    const auto numFrames = (signalLength - frameSize) / frameStep + 1;
    const auto fftSize = frameSize / 2 + 1;

    outputShape.pop_back();
    if (op.getTransposeFrames()) {
        outputShape.push_back(fftSize);
        outputShape.push_back(numFrames);
    } else {
        outputShape.push_back(numFrames);
        outputShape.push_back(fftSize);
    }
    outputShape.push_back(2);

    inferredReturnShapes.emplace_back(outputShape, elemType);
    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::IE::STFTOp::verify() {
    const auto signalType = mlir::cast<mlir::ShapedType>(getSignal().getType());
    const auto signalShape = signalType.getShape();

    if (signalShape.size() < 1 || signalShape.size() > 2) {
        return errorAt(getLoc(), "STFT signal must be 1D or 2D tensor, got {0}D", signalShape.size());
    }

    if (getWindow()) {
        const auto windowType = mlir::cast<mlir::ShapedType>(getWindow().getType());
        const auto windowShape = windowType.getShape();

        if (windowShape.size() != 1) {
            return errorAt(getLoc(), "STFT window must be 1D tensor, got {0}D", windowShape.size());
        }
    }

    bool hasFrameSize = getFrameSizeAttr() || getFrameSize();
    bool hasFrameStep = getFrameStepAttr() || getFrameStep();

    if (!hasFrameSize) {
        return errorAt(getLoc(), "STFT frame_size must be specified");
    }

    if (!hasFrameStep) {
        return errorAt(getLoc(), "STFT frame_step must be specified");
    }

    return mlir::success();
}
