//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>

using namespace vpux;

namespace vpux::VPU {
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
            .Case<Const::ReshapeAttr, Const::SubViewAttr, Const::LayoutCastAttr>([](Const::TransformAttrInterface) {
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

bool shouldProcessThisConstant(Const::DeclareOp constOp) {
    // preserve splats in @main - they (should be) cheap to work with.
    const auto contentAttr = constOp.getContentAttr();

    // E#151098: this should be handled the same way as view-like-only
    // transformations.
    if (contentAttr.getTransformations().empty()) {
        return false;
    }

    // ignore all non-OV constants
    if (!Const::isOpenVINOConstant(constOp)) {
        return false;
    }

    // splat values should be quick enough to process in main()
    if (contentAttr.isSplat()) {
        return false;
    }

    if (ViewLikeUtils::hasOnlyViewLikeTransformations(contentAttr)) {
        return false;
    }

    return true;
}

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

    auto avgOutElemType = outQType;

    if (isSupportedPerAxis) {
        // Note: do what ConvertElemType does to restore the offset.
        const auto offset = Const::details::getValueRangeOffset(inQType, outQType);
        // Negative zero points are not supported - we do the following trick:
        // Instead of converting !quant.uniform<u8:f16:s1:0> -> !quant.uniform<i8:f16:s1:-128>, we cast to
        // !quant.uniform<u8:f16:s1:128> and convert to !quant.uniform<i8:f16:s1:0>.
        const auto isNegative = offset < 0;
        const int64_t inZeroPoint = isNegative ? -offset : 0;
        const int64_t outZeroPoint = isNegative ? 0 : offset;

        // convert per-axis case to per-tensor:
        auto normInQuantCast = builder.create<IE::QuantizeCastOp>(loc, value, normalizeQuantStorageType(inQType));

        auto perTensorInQType = mlir::quant::UniformQuantizedType::get(
                inPerAxisQType.getFlags(), inPerAxisQType.getStorageType(), inPerAxisQType.getExpressedType(),
                /*scale=*/1.0, /*zeroPoint=*/inZeroPoint, inPerAxisQType.getStorageTypeMin(),
                inPerAxisQType.getStorageTypeMax());

        value = builder.create<IE::QuantizeCastOp>(loc, normInQuantCast, perTensorInQType).getResult();
        avgOutElemType = mlir::quant::UniformQuantizedType::get(
                outPerAxisQType.getFlags(), outPerAxisQType.getStorageType(), outPerAxisQType.getExpressedType(),
                /*scale=*/1.0, /*zeroPoint=*/outZeroPoint, outPerAxisQType.getStorageTypeMin(),
                outPerAxisQType.getStorageTypeMax());
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

    if (isSupportedPerAxis) {
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
                                                 /*outputChannels=*/nullptr,
                                                 /*inputChannels=*/nullptr);
            })
            .Case<Const::BroadcastAttr>([&](Const::BroadcastAttr broadcast) {
                const auto axis = broadcast.getAxis().getInt();
                const auto dimValue = broadcast.getValue().getInt();
                auto shape = SmallVector<int64_t>(input.getType().cast<NDTypeInterface>().getShape().raw());
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
                const auto reassociationMap =
                        IE::getReassociationMap(input.getType().cast<NDTypeInterface>().getShape().raw(), outputShape);
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
                const auto qElemType =
                        input.getType().cast<NDTypeInterface>().getElementType().cast<mlir::quant::QuantizedType>();
                return builder.create<IE::DequantizeOp>(loc, input, qElemType.getExpressedType());
            })
            .Case<Const::QuantizeAttr>([&](Const::QuantizeAttr quantizeAttr) {
                return builder.create<IE::QuantizeOp>(loc, input, quantizeAttr.getTargetType());
            })
            .Case<Const::LayoutCastAttr>([&](Const::LayoutCastAttr layoutCast) {
                return builder.create<IE::LayoutCastOp>(loc, input, layoutCast.getDstOrder());
            })
            .Case<Const::MemPermuteAttr>([&](Const::MemPermuteAttr memPermute) {
                return builder.create<IE::MemPermuteOp>(loc, input, memPermute.getDstOrder(), memPermute.getMemPerm());
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
                return builder.create<IE::PadOp>(
                        loc, input, /*padsBegin=*/nullptr, /*padsEnd=*/nullptr,
                        /*padValue=*/nullptr, padWithZero.getPadBefore(), padWithZero.getPadAfter(),
                        getFPAttr(builder.getContext(), extractPadValue(outElemType)), IE::PadMode::CONSTANT, nullptr);
            })
            .Case<Const::ReorderAttr>([&](Const::ReorderAttr reorder) {
                return builder.create<IE::ReorderOp>(loc, input, reorder.getOrder());
            })
            .Case<Const::RescaleAttr>([&](Const::RescaleAttr rescale) {
                const auto scaleValue = checked_cast<float>(rescale.getScale().getValueAsDouble());

                const auto scaleLoc = appendLoc(loc, "_scale");
                SmallVector<int64_t> shapeRank = {1};
                auto scaleType = mlir::RankedTensorType::get({shapeRank}, mlir::Float32Type::get(builder.getContext()));
                auto transform = [&](Const::ContentSetup& setup) -> Const::ContentSetup {
                    return setup.castElemType(mlir::cast<NDTypeInterface>(input.getType()).getElementType());
                };
                auto scale = Const::createConst<float>(builder, scaleLoc, scaleType, {scaleValue}, transform);

                return builder.create<IE::MultiplyOp>(loc, input, scale, IE::AutoBroadcastType::NUMPY,
                                                      /*postOp=*/nullptr, /*clamp=*/nullptr,
                                                      /*outputChannels=*/nullptr, /*inputChannels=*/nullptr);
            })
            .Case<Const::ReshapeAttr>([&](Const::ReshapeAttr reshape) -> mlir::Value {
                if (mlir::cast<NDTypeInterface>(input.getType()).getDimsOrder().isIdentity()) {
                    return builder.create<IE::ReshapeOp>(loc, input, nullptr, false, reshape.getShape());
                }

                return builder.create<IE::ShapeCastOp>(loc, input, reshape.getShape());
            })
            .Case<Const::ScalarMultInverseAttr>([&](Const::ScalarMultInverseAttr /*scalarMultInverse*/) -> mlir::Value {
                const auto inverseLoc = appendLoc(loc, "_inverse");
                SmallVector<int64_t> shapeRank = {1};
                const auto inputElemType = input.getType().cast<NDTypeInterface>().getElementType();
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
            .Case<Const::SparsifyAttr>([&](Const::SparsifyAttr sparsify) -> mlir::Value {
                // it's fine if sparsity is disabled
                if (!sparsify.getCompressOutputType().getValue()) {
                    return input;
                }

                return nullptr;
            })
            .Default([](Const::TransformAttrInterface) {
                return nullptr;
            });
}

/// Returns a VPU operation for the given constant transformation.
mlir::Value createMatchingVpuOperation(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                       Const::TransformAttrInterface t) {
    return llvm::TypeSwitch<Const::TransformAttrInterface, mlir::Value>(t)
            .Case([&](Const::SubViewAttr subView) -> mlir::Value {
                return builder.create<VPU::SliceOp>(loc, input, subView.getOffset(), subView.getShape());
            })
            .Default([](Const::TransformAttrInterface) -> mlir::Value {
                return nullptr;
            });
}

/// A simple dispatch around IE and VPU operation creation functions.
mlir::Value createMatchingOperation(WeightsSeparationSchedule scheduleKind, mlir::OpBuilder& builder,
                                    mlir::Location loc, mlir::Value input, Const::TransformAttrInterface t) {
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

}  // namespace

TransformationsSplit::TransformationsSplit(Const::DeclareOp declareOp): _declareOp(declareOp) {
    const auto contentAttr = declareOp.getContentAttr();
    auto transformations = contentAttr.getTransformations();

    if (mlir::isa<Const::SubViewAttr>(transformations.back())) {
        // move subviews to main since this reduces constant forwarding
        // between init and main (e.g. improves I/O interactions).
        _inInitTransformations = transformations.drop_back();
        _postInitTransformations = transformations.take_back();
    } else {
        _inInitTransformations = transformations;
        _postInitTransformations = {};
    }

    auto baseType = contentAttr.getBaseContent().getType();
    auto finalType = Const::inferFinalType(baseType, _inInitTransformations);

    // quantized types are not supported for network IO
    if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(finalType.getElementType()); qType != nullptr) {
        auto normalizedType = normalizeQuantStorageType(qType);
        _ioTypeInfo = {qType, normalizedType};
    }
}

NDTypeInterface TransformationsSplit::getBaseType() const {
    return mlir::cast<NDTypeInterface>(_declareOp.getContentAttr().getBaseContent().getType());
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
    return Projection{_declareOp, argType, precedingTransformations, transformations, _ioTypeInfo};
}

SmallVector<TransformationsSplit> collectMoveWorthyTransformationSplits(mlir::func::FuncOp mainFunc) {
    SmallVector<TransformationsSplit> splits;
    mainFunc.walk([&](Const::DeclareOp constOp) {
        if (!shouldProcessThisConstant(constOp)) {
            return;
        }
        splits.emplace_back(constOp);
    });

    // sort the found constants in a stable way. this ensures that the schedule
    // stays the same even when constant operation order changes.
    std::sort(splits.begin(), splits.end(), [](const TransformationsSplit& x, const TransformationsSplit& y) {
        const auto& xContent = x.declareOp().getContentAttr();
        const auto& yContent = y.declareOp().getContentAttr();

        const auto xName = getResourceName(xContent.getBaseContent());
        const auto yName = getResourceName(yContent.getBaseContent());
        assert((!xName.empty() && !yName.empty()) && "Non dense_resource<> constants must not be collected");
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
    });

    return splits;
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

mlir::Value ConstOpConverter::convertToIrForm(mlir::Location baseLoc,
                                              const VPU::TransformationsSplit::Projection& split,
                                              mlir::BlockArgument argValue, const IoBoundaryAdapter& ioAdaptor,
                                              WeightsSeparationSchedule scheduleKind) {
    auto [declareOp, argType, precedingTransformations, transformations, typeInfo] = split;

    auto loc = appendLoc(baseLoc, "arg{0}", argValue.getArgNumber());

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

    _log.trace("Creating matching operations for '{0}'", declareOp);
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

    // Note: do not store the operation anywhere as it is going to be deleted.
    _convertedConsts.emplace_back(value, declareOp.getContentAttr());

    return value;
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
