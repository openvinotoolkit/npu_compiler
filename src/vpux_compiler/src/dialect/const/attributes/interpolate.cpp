//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"

#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

mlir::LogicalResult vpux::Const::InterpolateAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                         mlir::ArrayAttr axes, mlir::ArrayAttr sizes,
                                                         mlir::StringAttr mode, mlir::StringAttr coordMode,
                                                         mlir::StringAttr nearestMode, mlir::BoolAttr /*antialias*/,
                                                         mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd,
                                                         mlir::FloatAttr cubeCoeff) {
    // Verify that axes and sizes have the same length
    if (axes.size() != sizes.size()) {
        return emitError() << "axes and sizes must have the same length";
    }

    // Verify that mode is one of the allowed values
    static const std::set<std::string> allowedModes = {"CUBIC"};
    if (allowedModes.find(mode.getValue().str()) == allowedModes.end()) {
        return emitError() << "only 'CUBIC' mode is supported now";
    }

    // Verify that coordMode is one of the allowed values
    static const std::set<std::string> allowedCoordModes = {"HALF_PIXEL"};
    if (allowedCoordModes.find(coordMode.getValue().str()) == allowedCoordModes.end()) {
        return emitError() << "only 'HALF_PIXEL' coordMode is supported now";
    }

    // Verify that nearestMode is one of the allowed values
    static const std::set<std::string> allowedNearestModes = {"FLOOR", "ROUND_PREFER_FLOOR"};
    if (allowedNearestModes.find(nearestMode.getValue().str()) == allowedNearestModes.end()) {
        return emitError() << "nearestMode must be one of 'FLOOR' or 'ROUND_PREFER_FLOOR'";
    }

    // Verify that padsBegin and padsEnd have the same length
    if (padsBegin.size() != padsEnd.size()) {
        return emitError() << "padsBegin and padsEnd must have the same length";
    }

    // Verify that cubeCoeff is within a reasonable range (example range check)
    if (cubeCoeff.getValueAsDouble() < -1.0 || cubeCoeff.getValueAsDouble() > 1.0) {
        return emitError() << "cubeCoeff must be between -1.0 and 1.0";
    }

    return mlir::success();
}

vpux::NDTypeInterface vpux::Const::InterpolateAttr::inferOutputType(vpux::NDTypeInterface inputType) const {
    // the size is already computed with pads, so we can directly use it
    auto outShape = to_small_vector(inputType.getShape());
    const auto axes = parseIntArrayAttr<int64_t>(getAxes());
    const auto sizes = parseIntArrayAttr<int64_t>(getSizes());
    for (const auto& [axis, size] : zip(axes, sizes)) {
        outShape[axis] = size;
    }
    return inputType.changeTypeComponents(TypeComponents().setShape(ShapeRef(outShape)));
}

bool vpux::Const::InterpolateAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

template <typename ElemType>
void interpCubicImpl(ArrayRef<ElemType> inputValues, Const::Content& output, NDTypeInterface inputType,
                     mlir::ArrayAttr axes, mlir::ArrayAttr sizes, mlir::StringAttr coordMode,
                     mlir::StringAttr /*nearestMode*/, mlir::BoolAttr /*antialias*/, mlir::ArrayAttr padsBegin,
                     mlir::ArrayAttr padsEnd, mlir::FloatAttr cubeCoeff) {
    VPUX_THROW_UNLESS(inputValues.size() > 1, "Splat case is handled outside of this function");
    const int64_t inputRank = inputType.getRank();

    // Validate axes
    const auto axesVec = parseIntArrayAttr<int64_t>(axes);
    for (auto axis : axesVec) {
        VPUX_THROW_UNLESS(axis >= 2 && axis < inputRank, "got unexpected content dimension {0} (only 2/3 is support)",
                          axis);
    }

    // Validate padding if specified
    if (padsBegin && padsEnd) {
        for (int64_t i = 0; i < inputRank; ++i) {
            int64_t padBegin = mlir::cast<mlir::IntegerAttr>(padsBegin.getValue()[i]).getInt();
            int64_t padEnd = mlir::cast<mlir::IntegerAttr>(padsEnd.getValue()[i]).getInt();
            VPUX_THROW_UNLESS(padBegin == 0 && padEnd == 0, "pad is not supported for interp const fold");
        }
    }

    // Coordinate transformation based on coordMode
    auto coordinateTransformFunc = [coordMode](float x_resized, int64_t length_resized, int64_t length_original) {
        const float original_scale = static_cast<float>(length_original) / static_cast<float>(length_resized);
        if (coordMode.getValue() == "HALF_PIXEL") {
            return (x_resized + 0.5f) * original_scale - 0.5f;
        } else {
            return x_resized * original_scale;
        }
    };

    // Cubic interpolation coefficients with configurable 'a' parameter
    const float a = cubeCoeff ? cubeCoeff.getValueAsDouble() : -0.75f;
    auto getCubicCoeff = [a](SmallVector<float>& coeffs, float s) {
        float abs_s = std::abs(s);
        coeffs[0] = ((a * (abs_s + 1) - 5 * a) * (abs_s + 1) + 8 * a) * (abs_s + 1) - 4 * a;
        coeffs[1] = ((a + 2) * abs_s - (a + 3)) * abs_s * abs_s + 1;
        coeffs[2] = ((a + 2) * (1 - abs_s) - (a + 3)) * (1 - abs_s) * (1 - abs_s) + 1;
        coeffs[3] = ((a * (2 - abs_s) - 5 * a) * (2 - abs_s) + 8 * a) * (2 - abs_s) - 4 * a;
    };

    // Calculate the stride for each axis, assume the input to be NCHW
    const auto inShape = ShapeRef(inputType.getShape());
    auto outShapeVec = to_small_vector(inputType.getShape());
    const auto sizesVec = parseIntArrayAttr<int64_t>(sizes);
    for (const auto& [axis, size] : zip(axesVec, sizesVec)) {
        outShapeVec[axis] = size;
    }

    auto outBuf = output.getTempBuf<ElemType>();
    // Perform interpolation for H and W, aligned with SHAVE kernel interpCubicCHW
    SmallVector<float> cubic_coeffs_W(4);
    SmallVector<float> cubic_coeffs_H(4);
    SmallVector<int> coords_W(4);
    SmallVector<int> coords_H(4);
    int base_coords_W;
    int base_coords_H;
    int IH = inShape[Dims4D::Act::H], IW = inShape[Dims4D::Act::W];
    int OH = outShapeVec[2], OW = outShapeVec[3], OC = outShapeVec[1];

    for (int h = 0; h < OH; h++) {
        float fh = coordinateTransformFunc(h, OH, IH);
        int ih = std::floor(fh);
        float h_lambda = fh - ih;
        getCubicCoeff(cubic_coeffs_H, h_lambda);
        base_coords_H = ih;

        coords_H[0] = std::clamp(base_coords_H + 0 - 1, 0, IH - 1);
        coords_H[1] = std::clamp(base_coords_H + 1 - 1, 0, IH - 1);
        coords_H[2] = std::clamp(base_coords_H + 2 - 1, 0, IH - 1);
        coords_H[3] = std::clamp(base_coords_H + 3 - 1, 0, IH - 1);
        for (int w = 0; w < OW; w++) {
            float fw = coordinateTransformFunc(w, OW, IW);
            int iw = std::floor(fw);
            float w_lambda = fw - iw;
            getCubicCoeff(cubic_coeffs_W, w_lambda);
            base_coords_W = iw;
            coords_W[0] = std::clamp(base_coords_W + 0 - 1, 0, IW - 1);
            coords_W[1] = std::clamp(base_coords_W + 1 - 1, 0, IW - 1);
            coords_W[2] = std::clamp(base_coords_W + 2 - 1, 0, IW - 1);
            coords_W[3] = std::clamp(base_coords_W + 3 - 1, 0, IW - 1);

            SmallVector<float> s0 = {cubic_coeffs_H[0], cubic_coeffs_H[0], cubic_coeffs_H[0], cubic_coeffs_H[0],
                                     cubic_coeffs_H[1], cubic_coeffs_H[1], cubic_coeffs_H[1], cubic_coeffs_H[1],
                                     cubic_coeffs_H[2], cubic_coeffs_H[2], cubic_coeffs_H[2], cubic_coeffs_H[2],
                                     cubic_coeffs_H[3], cubic_coeffs_H[3], cubic_coeffs_H[3], cubic_coeffs_H[3]};
            SmallVector<float> s1 = {cubic_coeffs_W[0], cubic_coeffs_W[1], cubic_coeffs_W[2], cubic_coeffs_W[3],
                                     cubic_coeffs_W[0], cubic_coeffs_W[1], cubic_coeffs_W[2], cubic_coeffs_W[3],
                                     cubic_coeffs_W[0], cubic_coeffs_W[1], cubic_coeffs_W[2], cubic_coeffs_W[3],
                                     cubic_coeffs_W[0], cubic_coeffs_W[1], cubic_coeffs_W[2], cubic_coeffs_W[3]};
            SmallVector<float> prod_0;
            for (size_t i = 0; i < 16; ++i) {
                prod_0.push_back(s0[i] * s1[i]);
            }
            for (int c = 0; c < OC; c++) {
                SmallVector<float> in = {inputValues[c * IH * IW + coords_H[0] * IW + coords_W[0]],
                                         inputValues[c * IH * IW + coords_H[0] * IW + coords_W[1]],
                                         inputValues[c * IH * IW + coords_H[0] * IW + coords_W[2]],
                                         inputValues[c * IH * IW + coords_H[0] * IW + coords_W[3]],
                                         inputValues[c * IH * IW + coords_H[1] * IW + coords_W[0]],
                                         inputValues[c * IH * IW + coords_H[1] * IW + coords_W[1]],
                                         inputValues[c * IH * IW + coords_H[1] * IW + coords_W[2]],
                                         inputValues[c * IH * IW + coords_H[1] * IW + coords_W[3]],
                                         inputValues[c * IH * IW + coords_H[2] * IW + coords_W[0]],
                                         inputValues[c * IH * IW + coords_H[2] * IW + coords_W[1]],
                                         inputValues[c * IH * IW + coords_H[2] * IW + coords_W[2]],
                                         inputValues[c * IH * IW + coords_H[2] * IW + coords_W[3]],
                                         inputValues[c * IH * IW + coords_H[3] * IW + coords_W[0]],
                                         inputValues[c * IH * IW + coords_H[3] * IW + coords_W[1]],
                                         inputValues[c * IH * IW + coords_H[3] * IW + coords_W[2]],
                                         inputValues[c * IH * IW + coords_H[3] * IW + coords_W[3]]};
                float result = 0.0f;
                for (size_t i = 0; i < 16; ++i) {
                    result += prod_0[i] * in[i];
                }
                int output_offset = c * OH * OW + h * OW + w;
                outBuf[output_offset] = Const::details::CvtHelper<ElemType>::cvt(result);
            }
        }
    }
}

Const::Content vpux::Const::InterpolateAttr::transform(vpux::Const::Content& input) const {
    const auto inType = input.getType();
    const auto outType = inferOutputType(inType);
    const auto elemType = input.getStorageElemType();

    auto output = Const::Content::allocTempBuffer(outType, elemType, inferOutputSplat(input.isSplat(), inType));
    if (input.isSplat()) {
        const auto inBuf = input.getRawStorageBuf();
        auto outBuf = output.getRawTempBuf();
        std::copy_n(inBuf.data(), inBuf.size(), outBuf.data());
    } else {
        if (mlir::isa<mlir::Float16Type>(elemType)) {
            const auto inBuf = input.getStorageBuf<vpux::type::float16>();
            interpCubicImpl<vpux::type::float16>(inBuf, output, inType, getAxes(), getSizes(), getCoordMode(),
                                                 getNearestMode(), getAntialias(), getPadsBegin(), getPadsEnd(),
                                                 getCubeCoeff());
        } else if (mlir::isa<mlir::Float32Type>(elemType)) {
            const auto inBuf = input.getStorageBuf<float>();
            interpCubicImpl<float>(inBuf, output, inType, getAxes(), getSizes(), getCoordMode(), getNearestMode(),
                                   getAntialias(), getPadsBegin(), getPadsEnd(), getCubeCoeff());
        }
    }

    return output;
}

mlir::Attribute vpux::Const::InterpolateAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    // We are trying to parse:
    // '#const.Interpolate<$axes, $sizes, $mode, $coordMode, $nearestMode, $antialias,
    //                     $padsBegin, $padsEnd, $cubeCoeff>'.
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr axes;
    if (mlir::failed(parser.parseAttribute(axes))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::ArrayAttr sizes;
    if (mlir::failed(parser.parseAttribute(sizes))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::StringAttr mode;
    if (mlir::failed(parser.parseAttribute(mode))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::StringAttr coordMode;
    if (mlir::failed(parser.parseAttribute(coordMode))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::StringAttr nearestMode;
    if (mlir::failed(parser.parseAttribute(nearestMode))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::BoolAttr antialias;
    if (mlir::failed(parser.parseAttribute(antialias))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::ArrayAttr padsBegin;
    if (mlir::failed(parser.parseAttribute(padsBegin))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::ArrayAttr padsEnd;
    if (mlir::failed(parser.parseAttribute(padsEnd))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::FloatAttr cubeCoeff;
    if (mlir::failed(parser.parseAttribute(cubeCoeff))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::InterpolateAttr::get(axes, sizes, mode, coordMode, nearestMode, antialias, padsBegin, padsEnd,
                                       cubeCoeff);
}

void vpux::Const::InterpolateAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getAxes());
    printer << ", ";
    printer.printAttribute(getSizes());
    printer << ", ";
    printer.printAttribute(getMode());
    printer << ", ";
    printer.printAttribute(getCoordMode());
    printer << ", ";
    printer.printAttribute(getNearestMode());
    printer << ", ";
    printer.printAttribute(getAntialias());
    printer << ", ";
    printer.printAttribute(getPadsBegin());
    printer << ", ";
    printer.printAttribute(getPadsEnd());
    printer << ", ";
    printer.printAttribute(getCubeCoeff());
    printer << ">";
}

llvm::hash_code vpux::Const::InterpolateAttr::getStableHashValue() const {
    const auto axes = parseIntArrayAttr<int64_t>(getAxes());
    const auto sizes = parseIntArrayAttr<int64_t>(getSizes());
    const auto mode = getMode().getValue();
    const auto coordMode = getCoordMode().getValue();
    const auto nearestMode = getNearestMode().getValue();
    const auto antialias = getAntialias().getValue();
    const auto padsBegin = parseIntArrayAttr<int64_t>(getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(getPadsEnd());
    const auto cubeCoeff = getCubeCoeff().getValue();

    return llvm::hash_combine(getMnemonic(), llvm::hash_value(ArrayRef<int64_t>(axes)),
                              llvm::hash_value(ArrayRef<int64_t>(sizes)), mode, coordMode, nearestMode, antialias,
                              llvm::hash_value(ArrayRef<int64_t>(padsBegin)),
                              llvm::hash_value(ArrayRef<int64_t>(padsEnd)), cubeCoeff);
}
