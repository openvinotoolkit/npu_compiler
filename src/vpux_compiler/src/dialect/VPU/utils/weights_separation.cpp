//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/core/force_link_macros.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Value.h>
#include <algorithm>

using namespace vpux;

// TODO: E#162744 remove this
DEFINE_FORCE_LINK(WsUtils)

namespace vpux::VPU {

MemPermuteConversionAttributes extractMemPermuteConversionAttributes(NDTypeInterface input,
                                                                     Const::MemPermuteAttr memPermuteAttr) {
    const auto inOrder = input.getDimsOrder();
    const auto inShape = input.getShape();

    const auto identityLayout = DimsOrder::fromNumDims(input.getRank()).toAffineMap(input.getContext());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    const auto memPermute = memPermuteAttr.getMemPerm().getAffineMap();
    const auto dstOrder = memPermuteAttr.getDstOrder().getAffineMap();
    const auto outShape = memPermuteAttr.inferOutputType(input).getShape();

    return MemPermuteConversionAttributes{identityLayout, inMemShape, memPermute, dstOrder, outShape};
}
namespace {

SmallVector<uint32_t> computeOrder(DimsOrder inOrder, DimsOrder outOrder) {
    auto inPerm = inOrder.toPermutation();
    auto outPerm = outOrder.toPermutation();
    SmallVector<uint32_t> memPerm(inPerm.size());
    for (auto p : outPerm | indexed) {
        memPerm[p.index()] = static_cast<uint32_t>(inOrder.dimPos(p.value()));
    }
    return memPerm;
}

/// Encloses the details of handling pure-view-like transformations within this
/// pass.
namespace ViewLikeUtils {
/// Returns whether the given transformation is considered view-like.
bool isViewLike(NDTypeInterface inputType, Const::TransformAttrInterface t) {
    const auto ctx = inputType.getContext();
    return llvm::TypeSwitch<Const::TransformAttrInterface, bool>(t)
            .Case<Const::ReshapeAttr, Const::SubViewAttr, Const::LayoutCastAttr, Const::AffineReshapeAttr>(
                    [](Const::TransformAttrInterface) {
                        return true;
                    })
            .Case([&](Const::MemPermuteAttr memPermute) {
                const auto inOrder = inputType.getDimsOrder();
                const auto inMemShape = inOrder.toMemoryOrder(inputType.getShape());
                const auto memPerm = memPermute.getMemPerm().getValue();
                return isTrivialPermute(inMemShape, memPerm);
            })
            .Case([&](Const::TransposeAttr transpose) {
                const auto inOrder = inputType.getDimsOrder();
                const auto inMemShape = inOrder.toMemoryOrder(inputType.getShape());
                const auto inPerm = inOrder.toAffineMap(ctx);
                const auto memPerm = inPerm.compose(transpose.getOrder().getValue());
                return isTrivialPermute(inMemShape, memPerm);
            })
            .Case([&](Const::ReorderAttr reorder) {
                const auto inOrder = inputType.getDimsOrder();
                const auto inMemShape = inOrder.toMemoryOrder(inputType.getShape());
                const auto outType = reorder.inferOutputType(inputType);
                const auto outOrder = outType.getDimsOrder();
                const auto memPerm = mlir::AffineMap::getPermutationMap(ArrayRef(computeOrder(inOrder, outOrder)), ctx);
                return isTrivialPermute(inMemShape, memPerm);
            })
            .Default([](Const::TransformAttrInterface) {
                return false;
            });
}

/// Returns whether the given constant contains only view-like transformations.
/// Note that constant with no transformations is also considered as having
/// view-like-only transformations.
bool hasOnlyViewLikeTransformations(const Const::ContentAttr& contentAttr) {
    auto previousType = mlir::cast<vpux::NDTypeInterface>(contentAttr.getBaseContent().getType());
    return llvm::all_of(contentAttr.getTransformations(), [&](auto trans) -> bool {
        const bool result = ViewLikeUtils::isViewLike(previousType, trans);
        previousType = trans.inferOutputType(previousType);
        return result;
    });
}

}  // namespace ViewLikeUtils

namespace conversions {  // forward declarations
bool isSupportedTransformation(vpux::NDTypeInterface inType, Const::TransformAttrInterface t);
}
}  // namespace

bool isTrivialForWeightsSeparation(Const::DeclareOp constOp) {
    // E#151098: this should be handled the same way as view-like-only
    // transformations.
    const auto contentAttr = constOp.getContentAttr();
    if (contentAttr.getTransformations().empty()) {
        return true;
    }

    return contentAttr.isSplat() || ViewLikeUtils::hasOnlyViewLikeTransformations(contentAttr);
}

bool isSuitableForWeightsSeparation(Const::DeclareOp constOp) {
    // ignore all non-OV constants
    if (!Const::isOpenVINOConstant(constOp)) {
        return false;
    }

    if (isTrivialForWeightsSeparation(constOp)) {
        return false;
    }

    auto hasOnlySupportedTransformations = [](Const::DeclareOp constOp) {
        auto contentAttr = constOp.getContentAttr();
        auto baseType = contentAttr.getBaseContent().getType();
        auto transformations = contentAttr.getTransformations();

        for (auto t : transformations.drop_back(1)) {
            if (!conversions::isSupportedTransformation(baseType, t)) {
                return false;
            }

            baseType = t.inferOutputType(baseType);
        }

        return conversions::isSupportedTransformation(baseType, transformations.back());
    };

    return hasOnlySupportedTransformations(constOp);
}

namespace {
/// Utilities related to converting const transformations to IR operations.
namespace conversions {
/// Returns a QuantizeCast, optionally wrapped into Convert ops to ensure
/// QuantizeCast's validity. This function assumes that either input or output
/// type is a quantized type.
mlir::Value createTypeCorrectedQuantizedCast(mlir::OpBuilder& builder, mlir::Value input, mlir::Location loc,
                                             mlir::Type inType, mlir::Type outType) {
    const auto qTypeIn = mlir::dyn_cast<mlir::quant::QuantizedType>(inType);
    const auto qTypeOut = mlir::dyn_cast<mlir::quant::QuantizedType>(outType);

    // IE::QuantizeCastOp does not support directly casting from one quantized type to another one.
    // This is why we perform qTypeIn -> intType -> qTypeOut.
    if (qTypeIn != nullptr && qTypeOut != nullptr) {
        auto cvtOp = builder.create<IE::QuantizeCastOp>(appendLoc(loc, "quant_cast_prepare_input"), input,
                                                        qTypeIn.getStorageType());
        return builder.create<IE::QuantizeCastOp>(loc, cvtOp, outType);
    }

    // from normal type to quantized type:
    if (qTypeOut != nullptr && mlir::isa<mlir::FloatType>(inType) && inType != qTypeOut.getStorageType()) {
        auto cvtOp = builder.create<IE::ConvertOp>(appendLoc(loc, "quant_cast_prepare_input"), input,
                                                   qTypeOut.getStorageType());
        return builder.create<IE::QuantizeCastOp>(loc, cvtOp, outType);
    }
    // from quantized type to normal type:
    if (qTypeIn != nullptr && mlir::isa<mlir::FloatType>(outType) && qTypeIn.getStorageType() != outType) {
        auto castOp = builder.create<IE::QuantizeCastOp>(loc, input, qTypeIn.getStorageType());
        return builder.create<IE::ConvertOp>(appendLoc(loc, "quant_cast_prepare_output"), castOp, outType);
    }

    return builder.create<IE::QuantizeCastOp>(loc, input, outType);
}

// This method attempts to create a matching IE::AvgPoolOp for per-axis
// quantized ConvertElemType. This is expected to yield more efficient IR.
mlir::Value createAvgPoolForInterQuantizedConvert(mlir::OpBuilder& builder, mlir::Value value, mlir::Location loc,
                                                  mlir::quant::QuantizedType inQType,
                                                  mlir::quant::QuantizedType outQType) {
    auto inPerAxisQType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inQType);
    auto outPerAxisQType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(outQType);

    const bool isPerTensor = inPerAxisQType == nullptr && outPerAxisQType == nullptr;
    const bool isSupportedPerAxis = [&]() {
        if (isPerTensor) {
            return false;
        }
        const auto inScales = extractScalesAndZeroPoints(inPerAxisQType).first;
        const auto outScales = extractScalesAndZeroPoints(outPerAxisQType).first;
        if (inScales.size() != outScales.size()) {
            return false;
        }

        auto allScalesAreEqual = llvm::all_of(llvm::zip(inScales, outScales), [](auto scales) {
            return std::get<0>(scales) == std::get<1>(scales);
        });
        return allScalesAreEqual;
    }();

    // Note: AvgPool only supports 3D, 4D, or 5D shapes. Anything larger is not
    // supported, anything smaller requires a reshape.
    const auto avgPoolShape = getShape(value);

    const bool isValidOverall = (isPerTensor || isSupportedPerAxis) && avgPoolShape.size() <= 5;
    if (!isValidOverall) {
        return nullptr;
    }

    const Shape properAvgPoolShape = [&]() {
        // Note: 4D AvgPool seems to be needed by NCE (e.g. see
        // --convert-nce-ops-to-4d), thus use 4D unconditionally - also to
        // eagerly extend 3D AvgPool.
        constexpr size_t preferredShapeSize = 4;
        if (bool noReshapeNeeded = avgPoolShape.size() >= preferredShapeSize; noReshapeNeeded) {
            return Shape(avgPoolShape);
        }

        const auto extraDims = preferredShapeSize - avgPoolShape.size();
        Shape newShape(preferredShapeSize, 1);
        std::copy(avgPoolShape.begin(), avgPoolShape.end(), newShape.begin() + extraDims);
        return newShape;
    }();

    const auto reshapeIfNecessary = [&](mlir::Location reshapeLoc, mlir::Value input, ShapeRef newShape) {
        if (properAvgPoolShape == avgPoolShape) {
            return input;
        }

        const auto newShapeAttr = getIntArrayAttr(builder.getContext(), newShape);
        return builder.createOrFold<IE::ReshapeOp>(reshapeLoc, input, nullptr, false, newShapeAttr);
    };

    const auto prepareZpsForAvgPool =
            [&](int64_t inZeroPoint,
                int64_t outZeroPoint) -> std::pair<mlir::Value, mlir::quant::UniformQuantizedType> {
        auto normInQuantCast = builder.create<IE::QuantizeCastOp>(loc, value, normalizeQuantStorageType(inQType));

        auto perTensorInQType = mlir::quant::UniformQuantizedType::get(
                inQType.getFlags(), inQType.getStorageType(), inQType.getExpressedType(),
                /*scale=*/1.0, /*zeroPoint=*/inZeroPoint, inQType.getStorageTypeMin(), inQType.getStorageTypeMax());

        auto resultValue = builder.create<IE::QuantizeCastOp>(loc, normInQuantCast, perTensorInQType).getResult();
        auto resultOutElemType = mlir::quant::UniformQuantizedType::get(
                outQType.getFlags(), outQType.getStorageType(), outQType.getExpressedType(),
                /*scale=*/1.0, /*zeroPoint=*/outZeroPoint, outQType.getStorageTypeMin(), outQType.getStorageTypeMax());

        return std::pair<mlir::Value, mlir::quant::UniformQuantizedType>(resultValue, resultOutElemType);
    };

    const auto hasNegativeZp = [&]() -> bool {
        auto zpIn = extractSingleZeroPoint(inQType);
        auto zpOut = extractSingleZeroPoint(outQType);
        VPUX_THROW_UNLESS(zpIn.has_value() && zpOut.has_value(), "Unsupported conversion: {0} -> {1}", inQType,
                          outQType);
        return zpIn.value() < 0 || zpOut.value() < 0;
    }();

    auto avgOutElemType = outQType;

    if (isSupportedPerAxis || hasNegativeZp) {
        // Note: do what ConvertElemType does to restore the offset.
        const auto offset = Const::details::getValueRangeOffset(inQType, outQType);
        // Negative zero points are not supported - we do the following trick:
        // Instead of converting !quant.uniform<u8:f16:s1:0> -> !quant.uniform<i8:f16:s1:-128>, we cast to
        // !quant.uniform<u8:f16:s1:128> and convert to !quant.uniform<i8:f16:s1:0>.
        const auto isNegative = offset < 0;
        const int64_t inZeroPoint = isNegative ? -offset : 0;
        const int64_t outZeroPoint = isNegative ? 0 : offset;
        // convert per-axis case to per-tensor:
        std::tie(value, avgOutElemType) = prepareZpsForAvgPool(inZeroPoint, outZeroPoint);
    }

    // Note: do reshape after per-axis to per-tensor conversion to eliminate the
    // need to patch quant axis.
    value = reshapeIfNecessary(appendLoc(value.getLoc(), "reshape_to_4d"), value, properAvgPoolShape);

    // Note: the length of strides, kernel, and pads correlates with tensor
    // sizes as `X = tensor_size - 2`.
    constexpr size_t avgPoolParamOffset = 2;
    assert(getShape(value) == properAvgPoolShape &&
           "Reshape logic guarantees that the value shape matches proper shape");
    const SmallVector<int64_t> poolStrides(properAvgPoolShape.size() - avgPoolParamOffset, 1);
    const SmallVector<int64_t> poolKernels(properAvgPoolShape.size() - avgPoolParamOffset, 1);
    const SmallVector<int64_t> pads(properAvgPoolShape.size() - avgPoolParamOffset, 0);
    auto ctx = builder.getContext();

    // implement inter-quantized-type convert via average pooling (on per-tensor
    // quantization types): such convert is essentially `IE.Add(%x, zero-point)`
    // and Add could be done via AvgPool.
    auto avgType = mlir::cast<NDTypeInterface>(value.getType()).changeElemType(avgOutElemType);
    auto avgPool = builder.create<IE::AvgPoolOp>(
            loc, avgType, value, getIntArrayAttr(ctx, poolKernels), getIntArrayAttr(ctx, poolStrides),
            getIntArrayAttr(ctx, pads), getIntArrayAttr(ctx, pads),
            vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR),
            mlir::UnitAttr::get(builder.getContext()), nullptr, nullptr, nullptr, nullptr, nullptr);
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(avgPool.getInput().getType().getElementType()) ||
        mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(avgPool.getOutput().getType().getElementType())) {
        return nullptr;
    }
    value = reshapeIfNecessary(appendLoc(value.getLoc(), "reshape_from_4d"), avgPool.getOutput(), avgPoolShape);

    if (isSupportedPerAxis || hasNegativeZp) {
        // convert per-tensor case to per-axis (restore the original type):
        auto normOutQuantCast =
                builder.create<IE::QuantizeCastOp>(loc, value, normalizeQuantStorageType(avgOutElemType));
        return builder.create<IE::QuantizeCastOp>(loc, normOutQuantCast, outQType).getResult();
    }

    return value;
}

/// Returns an IE operation for the given constant transformation.
mlir::Value createMatchingIeOperation(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                      Const::TransformAttrInterface t) {
    return llvm::TypeSwitch<Const::TransformAttrInterface, mlir::Value>(t)
            .Case<Const::AddAttr>([&](Const::AddAttr add) {
                const auto biasValue = checked_cast<float>(add.getBias().getValueAsDouble());

                const auto biasLoc = appendLoc(loc, "_bias");
                SmallVector<int64_t> shapeRank = {1};
                auto biasType = mlir::RankedTensorType::get(shapeRank, mlir::Float32Type::get(builder.getContext()));
                auto transform = [&](Const::ContentSetup& setup) -> Const::ContentSetup {
                    return setup.castElemType(mlir::cast<NDTypeInterface>(input.getType()).getElementType());
                };
                auto bias = Const::createConst<float>(builder, biasLoc, biasType, {biasValue}, transform);

                return builder.create<IE::AddOp>(loc, input, bias, IE::AutoBroadcastType::NUMPY,
                                                 /*postOp=*/nullptr, /*clamp=*/nullptr,
                                                 /*outputPadding=*/nullptr,
                                                 /*inputPadding=*/nullptr);
            })
            .Case<Const::BroadcastAttr>([&](Const::BroadcastAttr broadcast) {
                const auto axis = broadcast.getAxis().getInt();
                const auto dimValue = broadcast.getValue().getInt();
                auto shape = SmallVector<int64_t>(mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape().raw());
                shape[axis] = dimValue;

                const auto targetShapeLoc = appendLoc(loc, "_shape");
                SmallVector<int64_t> shapeRank = {static_cast<int64_t>(shape.size())};
                auto targetShapeType = mlir::RankedTensorType::get(shapeRank, getInt64Type(builder.getContext()));
                auto targetShape = Const::createConst<int64_t>(builder, targetShapeLoc, targetShapeType, shape);

                return builder.create<IE::BroadcastOp>(loc, input, targetShape, /*axesMapping=*/nullptr,
                                                       /*mode=*/nullptr);
            })
            .Case<Const::ChangeShapeAndElemTypeAttr>([&](Const::ChangeShapeAndElemTypeAttr changeShapeAndElemType) {
                const auto inElemType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(
                        mlir::cast<NDTypeInterface>(input.getType()).getElementType());
                const auto outElemType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(
                        changeShapeAndElemType.getElemType());

                // see IE::AffineReshape::fold()
                const bool specialCaseOfAffineReshapeFolding = inElemType != nullptr && outElemType != nullptr &&
                                                               isQuantizedDimensionPermutation(inElemType, outElemType);
                VPUX_THROW_UNLESS(specialCaseOfAffineReshapeFolding, "Unsupported affine-reshape operation");

                const auto outputShape = parseIntArrayAttr<int64_t>(changeShapeAndElemType.getShape());
                const auto reassociationMap = IE::getReassociationMap(
                        mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape().raw(), outputShape);
                const auto dimMapping =
                        getIntArrayOfArray(changeShapeAndElemType.getContext(), reassociationMap.value());
                return builder.create<IE::AffineReshapeOp>(loc, input, dimMapping, changeShapeAndElemType.getShape());
            })
            .Case<Const::CastElemTypeAttr>([&](Const::CastElemTypeAttr cast) -> mlir::Value {
                const auto inElemType = mlir::cast<NDTypeInterface>(input.getType()).getElementType();
                const auto outElemType = cast.getElemType();

                const bool inputQuantized = mlir::isa<mlir::quant::QuantizedType>(inElemType);
                const bool outputQuantized = mlir::isa<mlir::quant::QuantizedType>(outElemType);

                if (inputQuantized || outputQuantized) {
                    return createTypeCorrectedQuantizedCast(builder, input, loc, inElemType, outElemType);
                }
                return builder.create<IE::ConvertOp>(loc, input, outElemType);
            })
            .Case<Const::ConvertElemTypeAttr>([&](Const::ConvertElemTypeAttr convert) -> mlir::Value {
                const auto inElemType = mlir::dyn_cast_or_null<mlir::quant::QuantizedType>(
                        mlir::cast<NDTypeInterface>(input.getType()).getElementType());
                const auto outElemType = mlir::dyn_cast_or_null<mlir::quant::QuantizedType>(convert.getElemType());

                // quantized-to-quantized conversion is special
                if (inElemType != nullptr && outElemType != nullptr) {
                    return createAvgPoolForInterQuantizedConvert(builder, input, loc, inElemType, outElemType);
                }
                return builder.create<IE::ConvertOp>(loc, input, convert.getElemType());
            })
            .Case<Const::DequantizeAttr>([&](Const::DequantizeAttr /*dequantize*/) {
                const auto qElemType = mlir::cast<mlir::quant::QuantizedType>(
                        mlir::cast<vpux::NDTypeInterface>(input.getType()).getElementType());
                return builder.create<IE::DequantizeOp>(loc, input, qElemType.getExpressedType());
            })
            .Case<Const::QuantizeAttr>([&](Const::QuantizeAttr quantizeAttr) {
                return builder.create<IE::QuantizeOp>(loc, input, quantizeAttr.getTargetType());
            })
            .Case<Const::LayoutCastAttr>([&](Const::LayoutCastAttr layoutCast) {
                return builder.create<IE::LayoutCastOp>(loc, input, layoutCast.getDstOrder());
            })
            .Case<Const::MemPermuteAttr>([&](Const::MemPermuteAttr memPermute) -> mlir::Value {
                const auto inType = mlir::cast<NDTypeInterface>(input.getType());
                const auto [identityLayout, inMemShape, memPermuteMap, dstOrder, outShape] =
                        extractMemPermuteConversionAttributes(inType, memPermute);

                const auto memShapeAttr = getIntArrayAttr(builder.getContext(), inMemShape.raw());
                auto memShapeCastOp =
                        builder.create<IE::ShapeCastOp>(appendLoc(loc, "inMemShape"), input, memShapeAttr);

                auto castToIdentityLayoutOp = builder.create<IE::LayoutCastOp>(
                        appendLoc(memShapeCastOp.getLoc(), "identityLayout"), memShapeCastOp, identityLayout);

                auto transpose = builder.create<IE::TransposeOp>(
                        appendLoc(castToIdentityLayoutOp.getLoc(), "transpose"), castToIdentityLayoutOp,
                        /*order=*/nullptr, mlir::AffineMapAttr::get(memPermuteMap));

                const auto outShapeAttr = getIntArrayAttr(builder.getContext(), outShape.raw());
                auto newOutShapeCastOp = builder.create<IE::ShapeCastOp>(
                        appendLoc(transpose.getLoc(), "tranposedShape"), transpose, outShapeAttr);

                return builder.create<IE::LayoutCastOp>(appendLoc(newOutShapeCastOp.getLoc(), "recreateLayout"),
                                                        newOutShapeCastOp, dstOrder);
            })
            .Case<Const::PadWithZeroAttr>([&](Const::PadWithZeroAttr padWithZero) {
                auto extractPadValue = [](mlir::Type type) {
                    double padValue = 0.0;
                    if (auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(type)) {
                        const auto zeroPoints = perAxisQType.getZeroPoints();
                        const auto isSameZeroPoint = std::adjacent_find(zeroPoints.begin(), zeroPoints.end(),
                                                                        std::not_equal_to<>()) == zeroPoints.end();

                        VPUX_THROW_WHEN(!isSameZeroPoint, "Different ZPs for PadOp are not supported");
                        padValue = checked_cast<double>(zeroPoints.front());
                    } else if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(type)) {
                        padValue = checked_cast<double>(qType.getZeroPoint());
                    }
                    return padValue;
                };

                const auto inType = mlir::cast<NDTypeInterface>(input.getType());
                const auto outElemType = padWithZero.inferOutputType(inType).getElementType();
                return builder.create<IE::PadOp>(loc, input, /*padsBegin=*/nullptr, /*padsEnd=*/nullptr,
                                                 /*padValue=*/nullptr, padWithZero.getPadBefore(),
                                                 padWithZero.getPadAfter(),
                                                 getFPAttr(builder.getContext(), extractPadValue(outElemType)),
                                                 IE::PadMode::CONSTANT, nullptr, nullptr);
            })
            .Case<Const::ReorderAttr>([&](Const::ReorderAttr reorder) {
                return builder.create<IE::ReorderOp>(loc, input, reorder.getOrder());
            })
            .Case<Const::RescaleAttr>([&](Const::RescaleAttr rescale) {
                auto foldedAttr = rescale.getScale().fold();
                const auto scaleValue = foldedAttr.getSplatValue<float>();

                const auto scaleLoc = appendLoc(loc, "_scale");
                SmallVector<int64_t> shapeRank = {1};
                auto scaleType = mlir::RankedTensorType::get({shapeRank}, mlir::Float32Type::get(builder.getContext()));
                auto transform = [&](Const::ContentSetup& setup) -> Const::ContentSetup {
                    return setup.castElemType(mlir::cast<NDTypeInterface>(input.getType()).getElementType());
                };
                auto scale = Const::createConst<float>(builder, scaleLoc, scaleType, {scaleValue}, transform);

                return builder.create<IE::MultiplyOp>(loc, input, scale, IE::AutoBroadcastType::NUMPY,
                                                      /*postOp=*/nullptr, /*clamp=*/nullptr,
                                                      /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);
            })
            .Case<Const::ReshapeAttr>([&](Const::ReshapeAttr reshape) -> mlir::Value {
                if (mlir::cast<NDTypeInterface>(input.getType()).getDimsOrder().isIdentity()) {
                    return builder.create<IE::ReshapeOp>(loc, input, nullptr, false, reshape.getShape());
                }

                return builder.create<IE::ShapeCastOp>(loc, input, reshape.getShape());
            })
            .Case<Const::AffineReshapeAttr>([&](Const::AffineReshapeAttr affineReshape) -> mlir::Value {
                return builder.create<IE::AffineReshapeOp>(loc, input, affineReshape.getDimMapping(),
                                                           affineReshape.getShapeValue());
            })
            .Case<Const::ScalarMultInverseAttr>([&](Const::ScalarMultInverseAttr /*scalarMultInverse*/) -> mlir::Value {
                const auto inverseLoc = appendLoc(loc, "_inverse");
                SmallVector<int64_t> shapeRank = {1};
                const auto inputElemType = mlir::cast<vpux::NDTypeInterface>(input.getType()).getElementType();
                auto inverseType = mlir::RankedTensorType::get({shapeRank}, inputElemType);

                const auto data = [&]() -> mlir::DenseElementsAttr {
                    if (mlir::isa<mlir::Float16Type>(inputElemType)) {
                        return mlir::DenseElementsAttr::get(inverseType, ArrayRef({type::float16(1.0)}));
                    } else if (mlir::isa<mlir::Float32Type>(inputElemType)) {
                        return mlir::DenseElementsAttr::get(inverseType, ArrayRef({1.0f}));
                    } else if (mlir::isa<mlir::Float64Type>(inputElemType)) {
                        return mlir::DenseElementsAttr::get(inverseType, ArrayRef({1.0}));
                    }
                    return nullptr;
                }();
                if (data == nullptr) {
                    return nullptr;
                }

                auto contentAttr = Const::ContentAttr::get(data);
                auto inverse = builder.create<Const::DeclareOp>(inverseLoc, inverseType, std::move(contentAttr));
                return builder.create<IE::DivideOp>(loc, inverse, input, IE::AutoBroadcastType::NUMPY);
            })
            .Case<Const::SubViewAttr>([&](Const::SubViewAttr subview) {
                return builder.create<IE::SliceOp>(loc, input, subview.getOffset(), subview.getShape());
            })
            .Case<Const::TransposeAttr>([&](Const::TransposeAttr transpose) {
                return builder.create<IE::TransposeOp>(loc, input, /*order=*/nullptr, transpose.getOrder());
            })
            .Default([](Const::TransformAttrInterface) {
                return nullptr;
            });
}

/// Returns an IE operation for the given constant transformation.
bool isSupportedTransformation(vpux::NDTypeInterface inType, Const::TransformAttrInterface t) {
    if (mlir::isa<Const::DequantizeAttr>(t)) {
        if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(inType.getElementType())) {
            auto axis = perAxisType.getQuantizedDimension();
            auto rank = inType.getRank();
            auto isRank4Compatible = rank == 4 && axis == Dims4D::Act::C.ind();
            auto isRank5Compatible = rank == 5 && axis == Dims5D::Act::C.ind();

            return isRank4Compatible || isRank5Compatible;
        }

        return true;
    }

    // Note: this *has* to be aligned with createMatchingIeOperation and
    // createMatchingVpuOperation (mostly with the former).
    return mlir::isa<Const::AddAttr, Const::BroadcastAttr, Const::ChangeShapeAndElemTypeAttr, Const::CastElemTypeAttr,
                     Const::ConvertElemTypeAttr, Const::QuantizeAttr, Const::LayoutCastAttr, Const::MemPermuteAttr,
                     Const::PadWithZeroAttr, Const::ReorderAttr, Const::RescaleAttr, Const::ReshapeAttr,
                     Const::ScalarMultInverseAttr, Const::SubViewAttr, Const::TransposeAttr, Const::AffineReshapeAttr>(
            t);
}

/// Returns a VPU operation for the given constant transformation.
mlir::Value createMatchingVpuOperation(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                       Const::TransformAttrInterface t) {
    auto ctx = input.getContext();
    auto inputType = mlir::cast<NDTypeInterface>(input.getType());

    VPUX_THROW_WHEN(!ViewLikeUtils::isViewLike(inputType, t), "Transformation {0} has to be view-like", loc);

    return llvm::TypeSwitch<Const::TransformAttrInterface, mlir::Value>(t)
            .Case([&](Const::ReshapeAttr reshape) -> mlir::Value {
                if (inputType.getDimsOrder().isIdentity()) {
                    // Type inference rules between the attribute and the operation doesn't match
                    // Example: 2x4 #CN -> 2x2x2 isn't possible
                    return builder.create<VPU::ReshapeOp>(loc, input, nullptr, false, reshape.getShape());
                }

                return builder.create<VPU::ShapeCastOp>(loc, input, reshape.getShape());
            })
            .Case([&](Const::SubViewAttr subView) -> mlir::Value {
                return builder.create<VPU::SliceOp>(loc, input, subView.getOffset(), subView.getShape());
            })
            .Case([&](Const::LayoutCastAttr layoutCast) -> mlir::Value {
                return builder.create<VPU::LayoutCastOp>(loc, input, layoutCast.getDstOrder());
            })
            .Case([&](Const::MemPermuteAttr permute) -> mlir::Value {
                return builder.create<VPU::PermuteCastOp>(loc, input, permute.getDstOrder(), permute.getMemPerm());
            })
            .Case([&](Const::TransposeAttr transpose) -> mlir::Value {
                const auto outType = transpose.inferOutputType(inputType);
                const auto outShape = getIntArrayAttr(builder, outType.getShape());

                if (transpose.getOrder().getValue().isIdentity()) {
                    return builder.create<VPU::ReshapeOp>(loc, input, nullptr, false, outShape);
                }

                return builder.create<VPU::ShapeCastOp>(loc, input, outShape);
            })
            .Case([&](Const::ReorderAttr reorder) -> mlir::Value {
                const auto inOrder = inputType.getDimsOrder();
                auto outType = reorder.inferOutputType(inputType);
                const auto outOrder = outType.getDimsOrder();
                const auto memPerm = mlir::AffineMap::getPermutationMap(ArrayRef(computeOrder(inOrder, outOrder)), ctx);
                return builder.create<VPU::PermuteCastOp>(loc, input, reorder.getOrder(),
                                                          mlir::AffineMapAttr::get(memPerm));
            })
            .Case([&](Const::AffineReshapeAttr affineReshape) -> mlir::Value {
                return builder.create<VPU::AffineReshapeOp>(loc, input, affineReshape.getDimMapping(),
                                                            affineReshape.getShapeValue());
            })
            .Default([](Const::TransformAttrInterface) -> mlir::Value {
                return nullptr;
            });
}

/// A simple dispatch around IE and VPU operation creation functions.
mlir::Value createMatchingOperation(WeightsSeparationSchedule scheduleKind, mlir::OpBuilder& builder,
                                    mlir::Location loc, mlir::Value input, Const::TransformAttrInterface t) {
    VPUX_THROW_UNLESS(isSupportedTransformation(mlir::cast<vpux::NDTypeInterface>(input.getType()), t),
                      "Found non-supported transformation {0}", t);
    switch (scheduleKind) {
    case WeightsSeparationSchedule::Init:
        return createMatchingIeOperation(builder, loc, input, t);
    case WeightsSeparationSchedule::Main:
        return createMatchingVpuOperation(builder, loc, input, t);
    }

    // Note: in reality the below should never trigger since switch with no
    // default must raise at least a compiler warning (hopefully, an error in
    // our setup).
    VPUX_THROW("Internal error: unexpected kind of schedule provided");
}
}  // namespace conversions

std::vector<VPU::CallChainData> collectCallChains(mlir::func::FuncOp funcOp) {
    std::vector<VPU::CallChainData> functions;
    funcOp.walk([&](mlir::func::CallOp callOp) {
        functions.push_back({callOp, getCalledFunction(callOp)});
    });
    return functions;
}

std::vector<VPU::CallChainData> findChildren(const VPU::CallChainTree::Node& node) {
    auto funcOp = node.data().second;
    auto chains = collectCallChains(funcOp);
    // Note: sort call-chains lexicographically (using function names) to ensure
    // outlining-independent processing. while this disregards the call
    // sequence, this allows to avoid differences in schedule generation when
    // independent calls get reordered in IR:
    // ```cpp
    //  %call1 = call @foo1(...)
    //  %call2 = call @foo2(...)
    //  // vs:
    //  %call2 = call @foo2(...)
    //  %call1 = call @foo1(...)
    //
    //  // independent usage of calls:
    //  %op1 = VPU.Convolution(%call1)
    //  %ops2 = VPU.Convolution(%call2)
    // ```
    std::sort(chains.begin(), chains.end(), [](const VPU::CallChainData& x, const VPU::CallChainData& y) {
        auto xFunc = x.second;
        auto yFunc = y.second;
        // lexicographical comparison
        return xFunc.getSymName() < yFunc.getSymName();
    });

    return chains;
}

/// Helper utility that recognizes that duplicate transformation chains.
struct InitBufferSizeCache {
    mlir::DenseSet<llvm::hash_code> cache;  // caches transformation hashes

    /// Returns the buffer size of the init-produced result exactly once for
    /// every unique transformation chain.
    vpux::Byte getResultBufferSizeForInit(const TransformationsSplit& x) {
        const auto proj = x.take(WeightsSeparationSchedule::Init);
        const auto hashCode = Const::ContentAttr::getTransformationHash(proj.transformations);
        const bool firstOccurrence = cache.insert(hashCode).second;
        // Note: return 0 if "already seen" - to not account for the same result
        // multiple times.
        return firstOccurrence ? detail::getResultBufferSizeForInit(x) : vpux::Byte(0);
    }
};

template <typename Iterator>
vpux::Byte getResultBufferSizeForInit(Iterator first, Iterator last) {
    vpux::Byte res(0);
    // Note: when calculating the total buffer size of init results, one must
    // ensure that "same" transformations are not accounted for more than once.
    // Consider:
    // ```cpp
    //  %ov1_0 = const.Declare tensor<1x2xf16> = dense_resource<ov1> : tensor<2x2xf16>,
    //      [#const.Add<1.0>, #const.SubView<[0, 0], [1, 2]>]
    //  %ov1_1 = const.Declare tensor<1x2xf16> = dense_resource<ov1> : tensor<2x2xf16>,
    //      [#const.Add<1.0>, #const.SubView<[1, 0], [1, 2]>]
    // ```
    // init will have *1* result (tensor<2x2xf16>) for *2* subviews (because
    // subviews are part of main)
    InitBufferSizeCache cache;
    for (; first != last; ++first) {
        res += cache.getResultBufferSizeForInit(*first);
    }
    return res;
}

template <typename Iterator>
struct GroupedTransformationSplits {
    Iterator first;
    Iterator last;
    vpux::Byte totalBufferSize;  // Note: cached to reduce algorithmic complexity

    GroupedTransformationSplits(Iterator f, Iterator l, mlir::ElementsAttr baseContent)
            : first(f),
              last(l),
              totalBufferSize(getExpectedBufferSize(baseContent.getType()) + getResultBufferSizeForInit(f, l)) {
    }

    Iterator begin() const {
        return first;
    }
    Iterator end() const {
        return last;
    }
};

using SplitPosition = ArrayRef<TransformationsSplit>::const_iterator;
/// Groups transformation splits by dense_resource<>. Requires sorted
/// transformations splits.
SmallVector<GroupedTransformationSplits<SplitPosition>> groupTransformationSplitsByName(
        ArrayRef<TransformationsSplit> splits) {
    assert(std::is_sorted(splits.begin(), splits.end()) && "Requires transformation splits to be globally sorted");

    SmallVector<GroupedTransformationSplits<SplitPosition>> groupedByName;

    assert(!splits.empty());
    auto prev = splits.begin();
    auto prevResource = prev->getContentAttr().getBaseContent();

    // Note: since splits are sorted, we're guaranteed to have
    // same-resource-name constants to be together in sequence
    for (auto first = std::next(prev); first != splits.end(); ++first) {
        const auto& split = *first;
        const auto currResource = split.getContentAttr().getBaseContent();
        if (bool newConstant = (prevResource != currResource); newConstant) {
            groupedByName.emplace_back(prev, first, prevResource);
            prevResource = currResource;
            prev = first;
        }
    }
    groupedByName.emplace_back(prev, splits.end(), prevResource);

    return groupedByName;
}

/// A heuristic that places "small" constants closer together. Thus, in theory
/// reducing the amount of init schedules vs "no shuffle" for a simple
/// adjacent_find-like merging algorithm.
///
/// Consider:
/// ```
///     %big0 = dense_resource<ov_0> : tensor<large> // 250 MiB
///     %small0 = dense_resource<ov_1> : tensor<small> // 49 MiB
///     %big1 = dense_resource<ov_2> : tensor<large> // 330 MiB
///     %small1 = dense_resource<ov_3> : tensor<small> // 10 MiB
/// ```
/// given memory threshold = 200 MiB, the result is:
/// * init #0 takes %big0: sizeof(%big0) = 250 > 200 - exceeding already
/// * init #1 takes *only* %small0: sizeof(%small0) < 200
///   * but: sizeof(%small0 + %big1) = 49 + 330 > 200 - exceeds!
/// * init #2 takes %big1
/// * init #3 takes *only* %small1
///
/// yet when sorted, %small0 and %small1 are together in the sequence which
/// allows to reduce the number of inits to 3
void shuffleGroupedTransformationSplitsForBetterInitSchedule(
        SmallVector<GroupedTransformationSplits<SplitPosition>>& groupedSplits) {
    const auto lessByMemory = [&](const auto& x, const auto& y) {
        return x.totalBufferSize < y.totalBufferSize;
    };

    // Note: this sort should be stable to avoid same-size elements to be
    // reordered during sorting. this is important since this directly affects
    // the "order" of init schedules across different compilation calls
    // (potential to have different blobs across runs otherwise?)
    std::stable_sort(groupedSplits.begin(), groupedSplits.end(), lessByMemory);
}

/// Linearly traverses the specified groups of transformations splits, merging
/// neighbouring groups if the threshold allows.
std::vector<std::vector<TransformationsSplit>> getConstantsForInitSchedules(
        const SmallVector<GroupedTransformationSplits<SplitPosition>>& groupedSplits, vpux::Byte threshold) {
    // Note: instead of doing a greedy merging here, it may make sense to
    // implement something a bit more advanced such as FFD (see
    // https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing).

    assert(!groupedSplits.empty());
    std::vector<std::vector<TransformationsSplit>> slices;

    auto currGroupPosition = groupedSplits.begin();
    slices.emplace_back(std::make_move_iterator(currGroupPosition->begin()),
                        std::make_move_iterator(currGroupPosition->end()));

    auto accumulatedMemoryUsage = currGroupPosition->totalBufferSize;
    for (++currGroupPosition; currGroupPosition != groupedSplits.end(); ++currGroupPosition) {
        const auto currMemoryUsage = currGroupPosition->totalBufferSize;
        const bool canAddConstantToCurrentInit = (accumulatedMemoryUsage + currMemoryUsage) <= threshold;
        if (canAddConstantToCurrentInit) {
            accumulatedMemoryUsage += currMemoryUsage;
            slices.back().insert(slices.back().end(), std::make_move_iterator(currGroupPosition->begin()),
                                 std::make_move_iterator(currGroupPosition->end()));
            continue;
        }
        // create new init
        slices.emplace_back(std::make_move_iterator(currGroupPosition->begin()),
                            std::make_move_iterator(currGroupPosition->end()));
        accumulatedMemoryUsage = currMemoryUsage;
    }

    return slices;
}

}  // namespace

CallChainTree getOutliningRepresentation(mlir::func::FuncOp startFunc) {
    VPU::CallChainTree tree({VPU::CallChainTree::Node(VPU::CallChainData{nullptr, startFunc}, {})}, findChildren);
    return tree;
}

TransformationsSplit::TransformationsSplit(Const::DeclareOp declareOp)
        : _loc(declareOp.getLoc()), _contentAttr(declareOp.getContentAttr()) {
    const auto contentAttr = declareOp.getContentAttr();
    auto transformations = contentAttr.getTransformations();
    auto baseType = contentAttr.getBaseContent().getType();

    auto splitElements = [&]() {
        auto previousType = mlir::cast<vpux::NDTypeInterface>(baseType);
        auto splitIndex = 0;
        for (auto [idx, trans] : transformations | indexed) {
            if (!ViewLikeUtils::isViewLike(previousType, trans)) {
                splitIndex = idx + 1;
            }
            previousType = trans.inferOutputType(previousType);
        }
        return transformations.size() - splitIndex;
    }();

    _inInitTransformations = transformations.drop_back(splitElements);
    _postInitTransformations = transformations.take_back(splitElements);

    auto finalType = Const::inferFinalType(baseType, _inInitTransformations);

    // quantized types are not supported for network IO
    if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(finalType.getElementType()); qType != nullptr) {
        auto normalizedType = normalizeQuantStorageType(qType);
        _ioTypeInfo = {qType, normalizedType};
    }
}

NDTypeInterface TransformationsSplit::getBaseType() const {
    return mlir::cast<NDTypeInterface>(_contentAttr.getBaseContent().getType());
}

NDTypeInterface TransformationsSplit::getBoundaryType() const {
    auto type = Const::inferFinalType(getBaseType(), _inInitTransformations);
    if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type.getElementType()); qType != nullptr) {
        return type.changeElemType(normalizeQuantStorageType(qType));
    }
    return type;
}

TransformationsSplit::Projection TransformationsSplit::take(WeightsSeparationSchedule schedule) const {
    auto [argType, precedingTransformations,
          transformations] = [&]() -> std::tuple<mlir::Type, ArrayRef<Const::TransformAttrInterface>,
                                                 ArrayRef<Const::TransformAttrInterface>> {
        if (schedule == WeightsSeparationSchedule::Init) {
            return {getBaseType(), ArrayRef<Const::TransformAttrInterface>{}, _inInitTransformations};
        }
        return {getBoundaryType(), _inInitTransformations, _postInitTransformations};
    }();
    return Projection{_contentAttr, argType, precedingTransformations, transformations, _ioTypeInfo};
}

Const::ContentAttr TransformationsSplit::getContentAttr() const {
    return _contentAttr;
}

mlir::Location TransformationsSplit::getLoc() const {
    return _loc;
}

namespace detail {
vpux::Byte getResultBufferSizeForInit(const TransformationsSplit& x) {
    // Note: main's input is init's result, thus:
    // sizeof(init result) == sizeof(main arg)
    auto proj = x.take(WeightsSeparationSchedule::Main);
    return getExpectedBufferSize(proj.argType);
}
}  // namespace detail

namespace {
// comparison template that serves both TransformationsSplit and
// Const::DeclareOp
template <typename T>
bool stableCompare(const T& x, const T& y) {
    const auto& xContent = x.getContentAttr();
    const auto& yContent = y.getContentAttr();

    const auto xName = getResourceName(xContent.getBaseContent());
    const auto yName = getResourceName(yContent.getBaseContent());
    assert((!xName.empty() && !yName.empty()) && "Only dense_resource<> constants should be collected");
    // sort by resource name, then by transformations
    if (xName < yName) {
        return true;
    }
    if (xName == yName) {
        auto xHash = xContent.getTransformationHash();
        auto yHash = yContent.getTransformationHash();
        // Note: since we expect these hashes to be stable across
        // compilations, we could also rely on them to sort the constants.
        return xHash < yHash;
    }
    return false;  // xName > yName
}
}  // namespace

bool operator<(const TransformationsSplit& x, const TransformationsSplit& y) {
    return stableCompare(x, y);
}

std::vector<Const::DeclareOp> collectMoveWorthyConstants(const Logger& log, mlir::func::FuncOp mainFunc) {
    std::vector<Const::DeclareOp> ops;
    mainFunc.walk([&](Const::DeclareOp constOp) {
        if (!isSuitableForWeightsSeparation(constOp)) {
            log.trace("Constant is NOT used in init schedule: {0}", constOp);
            return;
        }
        ops.emplace_back(constOp);
    });

    // sort the found constants. this ensures that the schedule stays the same
    // even when constant operation order changes.
    llvm::sort(ops, stableCompare<Const::DeclareOp>);

    if (log.isActive(LogLevel::Trace)) {
        log.trace("Found the following constants in {0}:", mainFunc.getSymName());
        for (const auto& [index, declareOp] : ops | indexed) {
            log.trace("  {0}: {1}", index, declareOp);
        }
    }

    return ops;
}

namespace {
/// Collects all constant operations that are worth moving to the Init schedule
/// across the full IR (represented as a call-chain tree).
std::vector<Const::DeclareOp> collectMoveWorthyConstants(const Logger& log, const CallChainTree& tree) {
    std::vector<Const::DeclareOp> ops;

    FuncOpVisitor hasSeenThisFunction;
    utils::CallbackVisitor<CallChainData> splitCollector(
            [&](const CallChainTree::Node& node) {
                auto currOp = node.data().second;
                if (hasSeenThisFunction(currOp)) {
                    return false;
                }

                auto locals = VPU::collectMoveWorthyConstants(log, currOp);
                ops.insert(ops.end(), std::make_move_iterator(locals.begin()), std::make_move_iterator(locals.end()));
                return true;
            },
            nullptr);
    tree.apply(splitCollector);

    return ops;
}
}  // namespace

std::vector<std::vector<TransformationsSplit>> sliceAccordingToMemoryLimit(const Logger& log,
                                                                           ArrayRef<TransformationsSplit> splits,
                                                                           vpux::Byte memoryLimit) {
    // Note: by default, splits are sorted "locally" to the function that uses
    // the associated constants. that is:
    // * main_part1() uses dense_resource<ov_1> && dense_resource<ov_2>
    // * main_part2() uses dense_resource<ov_1>
    // * the splits are: [dense_resource<ov_1>, dense_resource<ov_2>,
    //   dense_resource<ov_1>]
    //
    // What this algorithm requires are "globally" sorted splits:
    assert(llvm::is_sorted(splits) && "Requires transformation splits to be globally sorted");
    if (splits.empty()) {
        return {};
    }

    if (log.isActive(LogLevel::Trace)) {
        log.trace("Slicing the following constants:");
        for (const auto& split : splits) {
            log.nest().trace("{0}", split.getContentAttr());
        }
    }

    // step 1: group all transformation splits by resource name. every group is
    // an "atomic element" that maps to an isolated init
    auto groupedByName = groupTransformationSplitsByName(splits);

    // step 2: mutually arrange groups in an "optimal" way
    shuffleGroupedTransformationSplitsForBetterInitSchedule(groupedByName);

    if (log.isActive(LogLevel::Trace)) {
        log.trace("Constants grouped by names:");
        for (const auto& [index, range] : groupedByName | indexed) {
            for (const auto& split : range) {
                log.nest().trace("group #{0}: {1}", index, split.getContentAttr());
            }
        }
    }

    // step 3: construct constants for init schedule(s) based on the memory
    // limit threshold
    auto constantsForInits = getConstantsForInitSchedules(groupedByName, memoryLimit);

    if (log.isActive(LogLevel::Trace)) {
        log.trace("Constants merged together:");
        for (const auto& [index, slice] : constantsForInits | indexed) {
            for (const auto& split : slice) {
                log.nest().trace("init #{0}: {1}", index, split.getContentAttr());
            }
        }
    }

    return constantsForInits;
}

mlir::StringRef getResourceId(mlir::DenseResourceElementsAttr attr) {
    auto fullKey = getResourceName(attr);

    return fullKey.drop_front(mlir::StringRef(Const::IMPORTED_WEIGHT_PREFIX).size());
}

std::string ConstArg::getUniqueName() const {
    const auto id = getResourceId(content);

    assert(!id.empty() && "Weights separation only works with dense_resource<>");
    const size_t hash = Const::ContentAttr::getTransformationHash(transformations);
    return formatv("{0}{1}_hash_{2}", INIT_OUTPUT_PREFIX, id, hash).str();
}

// We want to cache the results of mapping a list of transformations to operations to avoid the call of a
// UniquifyOps pass. Experiments showed that the load became significant in some cases.
class ConstOpConverter::OperationCache {
public:
    using KeyT = std::tuple<mlir::Value, ArrayRef<Const::TransformAttrInterface>>;

    OperationCache(const Logger& log): _log(log) {
    }

    mlir::Value findCachedResult(const KeyT& key) {
        auto it = _cachedResults.find(key);
        if (it == _cachedResults.end()) {
            return {};
        }
        return it->second;
    }

    void cacheResult(const KeyT& key, mlir::Value value) {
        _log.trace("Mapping input {0} with transformations {1} to {2}", std::get<0>(key), std::get<1>(key), value);
        _cachedResults[key] = value;
    }

private:
    llvm::DenseMap<KeyT, mlir::Value> _cachedResults;
    Logger _log;
};

ConstOpConverter::ConstOpConverter(mlir::func::FuncOp func, const Logger& log)
        : _func(func),
          _builderLogger(log.nest()),
          _opBuilder(mlir::OpBuilder::atBlockBegin(&_func.getFunctionBody().front(), &_builderLogger)),
          _log(log),
          _operationCache(std::make_unique<ConstOpConverter::OperationCache>(_log)) {
}
ConstOpConverter::~ConstOpConverter() = default;

std::tuple<ArrayRef<Const::TransformAttrInterface>, mlir::Value> ConstOpConverter::createMatchingOperation(
        mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
        ArrayRef<Const::TransformAttrInterface> transformations, WeightsSeparationSchedule scheduleKind) {
    if (transformations.empty()) {
        return {ArrayRef<Const::TransformAttrInterface>{}, input};
    }

    // 1-to-n mappings
    if (auto cachedValue = _operationCache->findCachedResult({input, transformations.take_front()});
        cachedValue != nullptr) {
        return {transformations.drop_front(), cachedValue};
    }

    const auto outputValue =
            conversions::createMatchingOperation(scheduleKind, builder, loc, input, transformations.front());
    if (outputValue == nullptr) {
        return {transformations, mlir::Value{}};
    }

    _operationCache->cacheResult({input, transformations.take_front()}, outputValue);

    return {transformations.drop_front(), outputValue};
}

mlir::Value ConstOpConverter::convertToIrForm(mlir::Location loc, const VPU::TransformationsSplit::Projection& split,
                                              mlir::BlockArgument argValue, const IoBoundaryAdapter& ioAdaptor,
                                              WeightsSeparationSchedule scheduleKind) {
    auto [contentAttr, argType, precedingTransformations, transformations, typeInfo] = split;

    mlir::Value value = argValue;

    // cache input
    if (auto cachedValue = _operationCache->findCachedResult({value, {}}); cachedValue != nullptr) {
        value = cachedValue;
    } else {
        // when not cached, also apply any required input value modifications
        auto newValue = ioAdaptor.wrapInput(_opBuilder, appendLoc(loc, "quant"), value, typeInfo);
        _operationCache->cacheResult({value, {}}, newValue);
        value = newValue;
    }

    _log.trace("Creating matching operations for '{0}'", contentAttr);
    _log.trace("  These transformations are converted into IR '{0}'", transformations);

    // convert transformation chain to operations
    size_t count = 0;
    while (!transformations.empty()) {
        const auto currLoc = appendLoc(loc, "trans{0}", count++);
        auto [remainingTransformations, resultValue] =
                createMatchingOperation(_opBuilder, currLoc, value, transformations, scheduleKind);
        VPUX_THROW_WHEN(resultValue == nullptr,
                        "The following transformations cannot be mapped to equivalent IE operations: {0}",
                        transformations);

        value = resultValue;
        transformations = remainingTransformations;
    }

    // cache output
    if (auto cachedValue = _operationCache->findCachedResult({value, {}}); cachedValue != nullptr) {
        value = cachedValue;
    } else {
        // when not cached, also apply any required output value modifications
        auto newValue = ioAdaptor.wrapOutput(_opBuilder, appendLoc(loc, "dequant"), value, typeInfo);
        _operationCache->cacheResult({value, {}}, newValue);
        value = newValue;
    }

    return value;
}

WeightsSeparationInfo::WeightsSeparationInfo(mlir::ModuleOp moduleOp) {
    auto log = Logger::global().nest("weights-separation-info", 0);
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp mainFuncOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, mainFuncOp);

    auto tree = VPU::getOutliningRepresentation(mainFuncOp);
    auto constants = collectMoveWorthyConstants(log, tree);
    _splits.reserve(constants.size());
    std::move(constants.begin(), constants.end(), std::back_inserter(_splits));
}

const std::vector<TransformationsSplit>& WeightsSeparationInfo::getCollectedSplits() const {
    return _splits;
}

bool WeightsSeparationInfo::isInvalidated(const mlir::AnalysisManager::PreservedAnalyses&) {
    return !_preserved;
}

void WeightsSeparationInfo::invalidate() {
    _preserved = false;
}

namespace {
mlir::RankedTensorType obfuscateType(ArrayRef<mlir::Type> originals) {
    const auto totalSize = std::accumulate(originals.begin(), originals.end(), vpux::Byte(0),
                                           [](vpux::Byte i, NDTypeInterface original) {
                                               return i + vpux::getExpectedBufferSize(original);
                                           });

    SmallVector<int64_t> shape({totalSize.count()});
    return mlir::RankedTensorType::get(shape, getInt8Type(originals.front().getContext()));
}

mlir::RankedTensorType obfuscateType(mlir::Type original) {
    return obfuscateType(ArrayRef{original});
}
}  // namespace

void obfuscateInputs(const Logger& log, mlir::Location loc, mlir::func::FuncOp funcOp, ArrayRef<size_t> indices,
                     CreateSliceOpFunc createSlice) {
    VPUX_THROW_WHEN(indices.empty(), "At least 1 input index is expected");

    auto& bodyBlock = funcOp.getFunctionBody().front();
    OpBuilderLogger builderLog(log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(&bodyBlock, &builderLog);

    if (bool singleInput = (indices.size() == 1); singleInput) {
        // Note: to simplify the external code, the procedure still has to
        // maintain relative order of arguments given multiple init schedules,
        // thus, one has to "push" block argument to the end.
        const size_t i = indices.front();
        auto oldArg = bodyBlock.getArgument(i);
        auto newArg = bodyBlock.addArgument(oldArg.getType(), oldArg.getLoc());
        oldArg.replaceAllUsesWith(newArg);
        bodyBlock.eraseArgument(i);

        funcOp.setFunctionType(
                mlir::FunctionType::get(builder.getContext(), bodyBlock.getArgumentTypes(), funcOp.getResultTypes()));
        return;
    }

    // obfuscate tensors to tensor<(TOTAL_SHAPE * ELEM_SIZE)xi8> - as specified
    // by indices
    auto inputs = to_std_vector(indices | transformed([&](size_t index) {
                                    return bodyBlock.getArgument(index);
                                }));
    auto inputTypes = to_std_vector(inputs | transformed([](mlir::BlockArgument x) {
                                        return x.getType();
                                    }));
    const auto blobType = obfuscateType(inputTypes);
    auto blobArg = bodyBlock.addArgument(blobType, loc);

    // slice obfuscated blob into deobfuscated tensors
    SmallVector<int64_t> offsets({0});  // updated in the loop
    for (auto [index, inputValue] : inputs | indexed) {
        const auto obfuscatedType = obfuscateType(inputValue.getType());

        const auto shape = mlir::cast<NDTypeInterface>(obfuscatedType).getShape();
        const auto inputValueLoc = inputValue.getLoc();
        auto sliceOp = createSlice(builder, appendLoc(inputValueLoc, "slice"), blobArg, offsets, shape.raw());
        auto castOp = builder.create<Core::ReinterpretCastOp>(appendLoc(sliceOp->getLoc(), "deobfuscated"),
                                                              inputValue.getType(), sliceOp->getResult(0));
        inputValue.replaceAllUsesWith(castOp.getResult());

        offsets[0] += shape[Dim(0)];

        log.debug("value #{0} is {1}, obfuscated as {2}", index, inputValue.getType(), obfuscatedType);
        // Note: location is printed separately because it differs between init
        // and main (this is by design).
        log.debug("value #{0} has location: {1}", index, inputValueLoc);
    }

    // Note: avoid index recalculation by deleting from the end first
    SmallVector<size_t> sorted(indices);
    llvm::sort(sorted, std::not_fn(std::less<size_t>{}));
    for (size_t index : sorted) {
        bodyBlock.eraseArgument(index);
    }

    funcOp.setFunctionType(
            mlir::FunctionType::get(builder.getContext(), bodyBlock.getArgumentTypes(), funcOp.getResultTypes()));
}

void obfuscateOutputs(const Logger& log, mlir::Location loc, mlir::func::FuncOp funcOp, ArrayRef<size_t> indices,
                      CreateConcatOpFunc createConcat) {
    VPUX_THROW_WHEN(indices.empty(), "At least 1 output index is expected");
    if (bool singleOutput = (indices.size() == 1); singleOutput) {
        return;
    }

    OpBuilderLogger builderLog(log.nest());
    mlir::OpBuilder builder(funcOp, &builderLog);

    // locate the return op
    const auto returns = funcOp.getOps<mlir::func::ReturnOp>();
    const size_t returnOpsCount = std::distance(returns.begin(), returns.end());
    VPUX_THROW_WHEN(returnOpsCount != 1, "Expected exactly 1 return op in function '{0}', found {1}",
                    funcOp.getSymName(), returnOpsCount);

    mlir::func::ReturnOp returnOp = *returns.begin();
    builder.setInsertionPoint(returnOp);

    // obfuscate tensors to tensor<(TOTAL_SHAPE * ELEM_SIZE)xi8> - as specified
    // by indices
    auto allReturns = to_std_vector(returnOp->getOperands());
    for (size_t index : indices) {
        auto returnValue = allReturns[index];
        const auto returnValueLoc = returnValue.getLoc();
        const auto type = obfuscateType(returnValue.getType());
        auto castOp =
                builder.create<Core::ReinterpretCastOp>(appendLoc(returnValueLoc, "obfuscated"), type, returnValue);
        allReturns[index] = castOp.getResult();

        log.debug("value #{0} is {1}, obfuscated as {2}", index, returnValue.getType(), type);
        // Note: location is printed separately because it differs between init
        // and main (this is by design).
        log.debug("value #{0} has location: {1}", index, returnValueLoc);
    }

    // concatenate obfuscated tensors
    const auto isObfuscated = [](mlir::Value x) {
        return x.getDefiningOp<Core::ReinterpretCastOp>() != nullptr;
    };

    std::vector<mlir::Value> concatReturns;
    concatReturns.reserve(indices.size());
    std::copy_if(allReturns.begin(), allReturns.end(), std::back_inserter(concatReturns), isObfuscated);
    auto concatOp = createConcat(builder, loc, concatReturns, 0);

    // rewrite the return statement to return a concatenated "blob", where new
    // returns = unmodified returns + single blob from concat
    std::vector<mlir::Value> newReturns;
    newReturns.reserve(allReturns.size() - concatReturns.size() + 1);
    std::copy_if(allReturns.begin(), allReturns.end(), std::back_inserter(newReturns), std::not_fn(isObfuscated));
    newReturns.push_back(concatOp->getResult(0));
    auto newReturnOp = builder.create<mlir::func::ReturnOp>(returnOp.getLoc(), newReturns);

    returnOp->replaceAllUsesWith(newReturnOp);
    returnOp->erase();

    funcOp.setFunctionType(mlir::FunctionType::get(builder.getContext(), funcOp.getArgumentTypes(),
                                                   newReturnOp.getOperands().getTypes()));
}

}  // namespace vpux::VPU

namespace llvm {
vpux::VPU::ConstArg DenseMapInfo<vpux::VPU::ConstArg>::getEmptyKey() {
    return vpux::VPU::ConstArg{llvm::DenseMapInfo<mlir::DenseResourceElementsAttr>::getEmptyKey(),
                               llvm::DenseMapInfo<ArrayRef<vpux::Const::TransformAttrInterface>>::getEmptyKey()};
}

vpux::VPU::ConstArg DenseMapInfo<vpux::VPU::ConstArg>::getTombstoneKey() {
    return vpux::VPU::ConstArg{llvm::DenseMapInfo<mlir::DenseResourceElementsAttr>::getTombstoneKey(),
                               llvm::DenseMapInfo<ArrayRef<vpux::Const::TransformAttrInterface>>::getTombstoneKey()};
}
unsigned DenseMapInfo<vpux::VPU::ConstArg>::getHashValue(const vpux::VPU::ConstArg& x) {
    // Note: a combination of base content and transformations on it
    // must uniquely identify any argument:
    // * for init, transformations are empty, thus deduplication happens
    //   at base content level
    // * for main, transformations are the ones that happen in init,
    //   thus deduplication happens at transformation chain level
    const auto name = getResourceName(x.content);
    auto hash = llvm::hash_combine(name, vpux::Const::ContentAttr::getTransformationHash(x.transformations));
    return hash;
}
bool DenseMapInfo<vpux::VPU::ConstArg>::isEqual(const vpux::VPU::ConstArg& x, const vpux::VPU::ConstArg& y) {
    return x.content == y.content && x.transformations == y.transformations;
}
}  // namespace llvm
