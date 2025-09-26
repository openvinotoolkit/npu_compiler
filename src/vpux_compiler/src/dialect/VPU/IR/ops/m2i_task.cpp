//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::M2ITaskOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    M2ITaskOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto inShape = inType.getShape().raw();

    SmallVector<int64_t> outShape(4);
    auto N = inShape[Dims4D::Act::N.ind()];
    auto Ci = inShape[Dims4D::Act::C.ind()];
    auto Hi = inShape[Dims4D::Act::H.ind()];
    auto Wi = inShape[Dims4D::Act::W.ind()];

    int64_t actualInputH{};
    const auto iFmt = op.getInFmt();
    if ((iFmt == M2iColorFmt::PL_YUV420_8) || (iFmt == M2iColorFmt::SP_NV12_8)) {
        // Input buffers for M2I are always declared with DimsOrder::NCHW so that the compiler doesn't introduce permute
        // layers that would ruin special formats like NV12, YUV420. But additional logic is added to correctly handle
        // the formats
        Hi = inShape[1];
        Wi = inShape[2];
        Ci = inShape[3];

        // Y,UV (1 or 2 plane configs) are expected to have C = 1
        if (Ci != 1) {
            return errorAt(loc, "Incorrect number of input channels: expecting 1, got '{0}'", Ci);
        }

        // input Height is big enough to include Chroma, so lower for RGB output
        actualInputH = Hi * 2 / 3;
    } else if (iFmt == M2iColorFmt::IL_RGB888) {
        actualInputH = inShape[1];
        Wi = inShape[2];
        Ci = inShape[3];
    } else if (iFmt == M2iColorFmt::PL_RGB24 || iFmt == M2iColorFmt::PL_FP16_RGB) {
        actualInputH = Hi;
    } else {
        VPUX_THROW("M2iTask currently unsupported input format '{0}'", iFmt);
    }

    int64_t Ho{}, Wo{};
    if (op.getSizes().has_value()) {
        // Note: limited to 'shape_calculation_mode = sizes'
        const auto outSize = parseIntArrayAttr<int64_t>(op.getSizes().value());
        Ho = outSize[0];
        Wo = outSize[1];
    } else {
        Ho = actualInputH;
        Wo = Wi;
    }

    const auto oFmt = op.getOutFmt();
    if ((oFmt == M2iColorFmt::PL_YUV420_8) || (oFmt == M2iColorFmt::SP_NV12_8)) {
        outShape[0] = N;
        outShape[1] = Ho * 3 / 2;
        outShape[2] = Wo;
        outShape[3] = 1;
    } else if (oFmt == M2iColorFmt::PL_RGB24 || oFmt == M2iColorFmt::PL_FP16_RGB) {
        outShape[0] = N;
        outShape[1] = 3;
        outShape[2] = Ho;
        outShape[3] = Wo;
    } else if (oFmt == M2iColorFmt::IL_RGB888) {
        outShape[0] = N;
        outShape[1] = Ho;
        outShape[2] = Wo;
        outShape[3] = 3;
    } else {
        VPUX_THROW("M2iTask currently unsupported output format '{0}'", oFmt);
    }

    mlir::Type outElemType = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
    if (oFmt == M2iColorFmt::PL_FP16_RGB) {
        outElemType = mlir::Float16Type::get(ctx);
    }

    const auto outType = inType.changeElemType(outElemType).changeShape(ShapeRef(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
