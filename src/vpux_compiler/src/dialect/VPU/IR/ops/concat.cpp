//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <map>
#include <unordered_set>

using namespace vpux;

//
// build
//

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                IE::ConcatAttr per_axis) {
    build(builder, state, inputs, per_axis, nullptr, nullptr, nullptr);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                IE::ConcatAttr per_axis, mlir::ArrayAttr static_offsets) {
    build(builder, state, inputs, per_axis, static_offsets, nullptr, nullptr);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                mlir::IntegerAttr axis, mlir::IntegerAttr offset, mlir::IntegerAttr stride) {
    build(builder, state, inputs, IE::ConcatAttr::get(builder.getContext(), axis, offset, stride));
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                int64_t axis, int64_t offset, int64_t stride) {
    build(builder, state, inputs, getIntAttr(builder, axis), offset != 0 ? getIntAttr(builder, offset) : nullptr,
          stride != 1 ? getIntAttr(builder, stride) : nullptr);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                Dim axis, int64_t offset, int64_t stride) {
    build(builder, state, inputs, axis.ind(), offset, stride);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                mlir::ValueRange inputs, mlir::ArrayAttr static_offsets) {
    build(builder, state, outType, inputs, nullptr, static_offsets, nullptr, nullptr);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                mlir::ValueRange inputs, IE::ConcatAttr per_axis, mlir::ArrayAttr static_offsets) {
    build(builder, state, outType, inputs, per_axis, static_offsets, nullptr, nullptr);
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                mlir::ValueRange inputs, ArrayRef<Shape> static_offsets) {
    const auto attrArr = to_small_vector(static_offsets | transformed([&](ShapeRef arr) -> mlir::Attribute {
                                             return getIntArrayAttr(builder, arr);
                                         }));

    build(builder, state, outType, inputs, builder.getArrayAttr(attrArr));
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                mlir::ValueRange inputs, ArrayRef<ShapeRef> static_offsets) {
    const auto attrArr = to_small_vector(static_offsets | transformed([&](ShapeRef arr) -> mlir::Attribute {
                                             return getIntArrayAttr(builder, arr);
                                         }));

    build(builder, state, outType, inputs, builder.getArrayAttr(attrArr));
}

void vpux::VPU::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                                mlir::ValueRange inputs, mlir::ArrayAttr static_offsets, mlir::ArrayAttr strides) {
    build(builder, state, outType, inputs, nullptr, static_offsets, strides, nullptr);
}

//
// InferTypeOpInterface
//

namespace {

using GetShapeFunc = std::function<Shape(const mlir::Value)>;

Shape getDynamicShape(const mlir::Value val) {
    return mlir::cast<vpux::NDTypeInterface>(val.getType()).getShape().toValues();
}

Shape getUpperBounds(const mlir::Value val) {
    auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(val.getType());
    VPUX_THROW_UNLESS(boundedType != nullptr, "Expected to get BoundedTensorType at {0}", val.getLoc());

    return Shape(boundedType.getBounds().raw());
}

Dim normalizeAxis(VPU::ConcatOpAdaptor concat) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(concat.getInputs().front().getType());
    const auto inRank = inType.getRank();

    auto axisInd = concat.getPerAxis().value().getAxis().getValue().getSExtValue();

    // Negative value means counting dimension from the end
    if (axisInd < 0) {
        axisInd += inRank;
    }

    VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Concat axis '{0}', out of range '{1}'", axisInd,
                      inRank);

    return Dim(axisInd);
}

mlir::FailureOr<Shape> inferOutShapeWithAxis(VPU::ConcatOpAdaptor concat, const GetShapeFunc& getShapeFunctor,
                                             mlir::Location loc) {
    const auto axis = normalizeAxis(concat);

    Shape outShape = getShapeFunctor(concat.getInputs().front());

    for (const auto val : concat.getInputs().drop_front()) {
        const auto curShape = getShapeFunctor(val);

        if (curShape.size() != outShape.size()) {
            return errorAt(loc, "Concat inputs have mismatched ranks: '{0}' vs '{1}'", curShape.size(),
                           outShape.size());
        }

        outShape[axis] += curShape[axis];
    }

    const auto perAxis = concat.getPerAxis().value();
    const auto offset = perAxis.getOffset() ? perAxis.getOffset().getValue().getSExtValue() : 0;
    const auto stride = perAxis.getStride() ? perAxis.getStride().getValue().getSExtValue() : 1;

    int64_t maxLatestIdx = -1;
    for (const auto idx : irange(concat.getInputs().size())) {
        const auto curShape = getShape(concat.getInputs()[idx]);
        const int64_t sizeByAxis = curShape[axis];
        const int64_t latestElemIdx = offset * idx + (sizeByAxis > 0 ? stride * (sizeByAxis - 1) : 0);
        maxLatestIdx = std::max(maxLatestIdx, latestElemIdx);
    }

    if (maxLatestIdx >= outShape[axis]) {
        return errorAt(loc, "Concat with offset '{0}' and stride '{1}' doesn't fit to output dimension '{2}'", offset,
                       stride, outShape[axis]);
    }

    return outShape;
}

mlir::FailureOr<Shape> inferOutShapeWithOffsets(VPU::ConcatOpAdaptor concat, const GetShapeFunc& getShapeFunctor,
                                                mlir::Location loc, mlir::MLIRContext* ctx) {
    if (!concat.getStaticOffsets().has_value()) {
        return errorAt(loc, "Missing static_offsets attribute");
    }

    const auto staticOffsets = concat.getStaticOffsets().value();
    if (staticOffsets.size() != concat.getInputs().size()) {
        return errorAt(loc, "Concat 'static_offsets' count '{0}' doesn't match inputs count '{1}'",
                       staticOffsets.size(), concat.getInputs().size());
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(concat.getInputs().front().getType());
    const auto allOffsets = staticOffsets.getAsRange<mlir::ArrayAttr>();

    const auto dummyStridesAttr = getIntArrayAttr(ctx, SmallVector<int64_t>(inType.getRank(), 1));
    SmallVector<mlir::ArrayAttr> allStrides(concat.getInputs().size(), dummyStridesAttr);

    if (concat.getStrides().has_value()) {
        allStrides = parseCustomAttrArray<mlir::ArrayAttr>(concat.getStridesAttr());
        if (allStrides.size() != concat.getInputs().size()) {
            return errorAt(loc, "Concat 'strides' count '{0}' doesn't match inputs count '{1}'", allStrides.size(),
                           concat.getInputs().size());
        }
    }

    Shape outShape(checked_cast<size_t>(inType.getRank()), 0);

    for (const auto p : zip(concat.getInputs(), allOffsets, allStrides)) {
        const auto curVal = std::get<0>(p);
        const auto curShape = getShapeFunctor(curVal);

        if (curShape.size() != outShape.size()) {
            return errorAt(loc, "Concat inputs have mismatched ranks: '{0}' vs '{1}'", curShape.size(),
                           outShape.size());
        }

        const auto curOffsets = Shape(parseIntArrayAttr<int64_t>(std::get<1>(p)));

        if (curOffsets.size() != curShape.size()) {
            return errorAt(loc, "Concat 'static_offsets' rank doesn't match its input");
        }

        const auto curStrides = Shape(parseIntArrayAttr<int64_t>(std::get<2>(p)));

        if (concat.getStrides().has_value() && curStrides.size() != curShape.size()) {
            return errorAt(loc, "Concat 'strides' rank doesn't match its input");
        }

        for (const auto ind : irange(outShape.size())) {
            const auto d = Dim(ind);
            if (curShape[d] == mlir::ShapedType::kDynamic) {
                outShape[d] = curShape[d];
            } else {
                const auto curOutShape = curStrides[d] * (curShape[d] - 1) + 1 + curOffsets[d];
                outShape[d] = std::max(outShape[d], curOutShape);
            }
        }
    }

    // TODO: validate that inputs+static_offsets fully covers the output without intersections

    return outShape;
}

mlir::FailureOr<mlir::Type> inferOutElemTypeWithAxis(VPU::ConcatOpAdaptor concat, mlir::Location loc) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(concat.getInputs().front().getType());
    const auto inElemType = inType.getElementType();

    const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(inElemType);
    SmallVector<mlir::quant::UniformQuantizedPerAxisType> inPerAxisQTypes;

    if (perAxisQType != nullptr) {
        const auto axis = normalizeAxis(concat);

        if (perAxisQType.getQuantizedDimension() == axis.ind()) {
            inPerAxisQTypes.push_back(perAxisQType);
        }
    }

    for (const auto val : concat.getInputs().drop_front()) {
        const auto curType = mlir::cast<vpux::NDTypeInterface>(val.getType());
        const auto curElemType = curType.getElementType();

        if (inPerAxisQTypes.empty()) {
            if (curElemType != inElemType) {
                return errorAt(loc, "Misaligned element types : '{0}' vs '{1}'", curElemType, inElemType);
            }
        } else {
            const auto curPerAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(curElemType);

            if (curPerAxisQType == nullptr) {
                return errorAt(loc,
                               "Misaligned element types : not all of them are per-axis quantized : '{0}' vs '{1}'",
                               curElemType, inElemType);
            }

            if (curPerAxisQType.getQuantizedDimension() != perAxisQType.getQuantizedDimension()) {
                return errorAt(
                        loc,
                        "Misaligned element types : per-axis quantization is done on different axis : '{0}' vs '{1}'",
                        curPerAxisQType.getQuantizedDimension(), perAxisQType.getQuantizedDimension());
            }

            if (!canBeMerged(curPerAxisQType, perAxisQType)) {
                return errorAt(loc, "Misaligned element types : per-axis quantization parameters can't be merged");
            }

            inPerAxisQTypes.push_back(curPerAxisQType);
        }
    }

    return inPerAxisQTypes.empty() ? inElemType : concatScalesAndZP(inPerAxisQTypes);
}

std::unordered_set<Dim> getConcatAxesFromOffsets(VPU::ConcatOpAdaptor concat, ShapeRef outShape) {
    std::unordered_set<Dim> res;

    for (const auto inVal : concat.getInputs()) {
        const auto curShape = getShape(inVal);

        for (const auto ind : irange(outShape.size())) {
            const auto d = Dim(ind);

            if (curShape[d] != outShape[d]) {
                res.insert(d);
            }
        }
    }

    return res;
}

mlir::FailureOr<mlir::Type> inferOutElemTypeWithOffsets(VPU::ConcatOpAdaptor concat, ShapeRef outShape,
                                                        mlir::Location loc) {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(concat.getInputs().front().getType());
    const auto inElemType = inType.getElementType();

    const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(inElemType);

    const auto isConcatOverPerAxisQuantization = [&]() {
        if (perAxisQType == nullptr) {
            return false;
        }

        const auto qDim = Dim(perAxisQType.getQuantizedDimension());
        const auto concatAxes = getConcatAxesFromOffsets(concat, outShape);

        return concatAxes.count(qDim) != 0;
    }();

    if (!isConcatOverPerAxisQuantization) {
        for (const auto val : concat.getInputs().drop_front()) {
            const auto curType = mlir::cast<vpux::NDTypeInterface>(val.getType());
            const auto curElemType = curType.getElementType();

            if (curElemType != inElemType) {
                return errorAt(loc, "Misaligned element types : '{0}' vs '{1}'", curElemType, inElemType);
            }
        }

        return inElemType;
    }

    const auto qDim = perAxisQType.getQuantizedDimension();
    const auto allOffsets = concat.getStaticOffsets().value().getAsRange<mlir::ArrayAttr>();

    std::map<int64_t, mlir::quant::UniformQuantizedPerAxisType> perSliceQuantTypes;

    for (const auto p : zip(concat.getInputs(), allOffsets)) {
        const auto curVal = std::get<0>(p);

        const auto curType = mlir::cast<vpux::NDTypeInterface>(curVal.getType());
        const auto curElemType = curType.getElementType();
        const auto curPerAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(curElemType);

        if (curPerAxisQType == nullptr) {
            return errorAt(loc, "Misaligned element types : not all of them are per-axis quantized : '{0}' vs '{1}'",
                           curElemType, inElemType);
        }

        if (curPerAxisQType.getQuantizedDimension() != qDim) {
            return errorAt(
                    loc, "Misaligned element types : per-axis quantization is done on different axis : '{0}' vs '{1}'",
                    curPerAxisQType.getQuantizedDimension(), qDim);
        }

        const auto curOffsets = parseIntArrayAttr<int64_t>(std::get<1>(p));
        const auto sliceOffset = curOffsets[checked_cast<size_t>(qDim)];

        const auto it = perSliceQuantTypes.find(sliceOffset);
        if (it == perSliceQuantTypes.end()) {
            perSliceQuantTypes.insert({sliceOffset, curPerAxisQType});
        } else {
            if (curPerAxisQType != it->second) {
                return errorAt(loc, "Per-axis quantization is not aligned over non quantized axis : '{0}' vs '{1}'",
                               curPerAxisQType, it->second);
            }
        }
    }

    const auto inPerAxisQTypes = to_small_vector(perSliceQuantTypes | map_values);
    return concatScalesAndZP(inPerAxisQTypes);
}

}  // namespace

mlir::LogicalResult vpux::VPU::ConcatOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ConcatOpAdaptor concat(operands, attrs, prop);
    if (mlir::failed(concat.verify(loc))) {
        return mlir::failure();
    }

    // Infer output shape
    const auto wrappedOutShape = concat.getPerAxis() ? inferOutShapeWithAxis(concat, getDynamicShape, loc)
                                                     : inferOutShapeWithOffsets(concat, getDynamicShape, loc, ctx);
    if (mlir::failed(wrappedOutShape)) {
        return mlir::failure();
    }

    const auto outShape = wrappedOutShape.value();

    // Infer output element type

    const auto outElemType = concat.getPerAxis() ? inferOutElemTypeWithAxis(concat, loc)
                                                 : inferOutElemTypeWithOffsets(concat, outShape, loc);
    if (mlir::failed(outElemType)) {
        return mlir::failure();
    }

    const auto inType = concat.getInputs().front().getType();
    const auto distributedIn = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(inType);
    VPU::DistributionInfoAttr possibleDistribution =
            distributedIn != nullptr && distributedIn.containsDistributedTypes()
                    ? mlir::cast<vpux::VPU::DistributedTensorType>(distributedIn.getDistributedTypes().front())
                              .getDistribution()
                    : nullptr;

    if (possibleDistribution != nullptr && VPU::isDistributedAttrWithExplicitShapesAndOffsets(possibleDistribution)) {
        const auto outputDistribution =
                VPU::getConcatExplicitDistributedAttrForNewShape(possibleDistribution, outShape, ctx);
        const auto outputType = distributedIn.changeShapeForExplicitDistribution(outShape, outputDistribution);
        inferredTypes.emplace_back(outputType);
    } else {
        if (outShape.isStatic()) {
            const auto typeComponents = TypeComponents().setShape(outShape).setElementType(outElemType.value());

            const auto outputType = mlir::cast<vpux::NDTypeInterface>(inType).changeTypeComponents(typeComponents);
            inferredTypes.emplace_back(outputType);
        } else {
            // Infer output bounds
            const auto outBounds = concat.getPerAxis() ? inferOutShapeWithAxis(concat, getUpperBounds, loc)
                                                       : inferOutShapeWithOffsets(concat, getUpperBounds, loc, ctx);
            if (mlir::failed(outBounds)) {
                return mlir::failure();
            }

            const auto typeComponents = TypeComponents()
                                                .setShape(outShape)
                                                .setElementType(outElemType.value())
                                                .setBounds(Bounds(outBounds.value().raw()));
            const auto outputType = mlir::cast<vpux::NDTypeInterface>(inType).changeTypeComponents(typeComponents);

            inferredTypes.emplace_back(outputType);
        }
    }

    return mlir::success();
}

//
// ConvertPerAxisToOffsets
//

namespace {

class ConvertPerAxisToOffsets final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    using mlir::OpRewritePattern<VPU::ConcatOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;
};

const mlir::ArrayAttr inferOffsetsAttrWithAxis(VPU::ConcatOp origOp, const int64_t& axis) {
    auto rank = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getRank();

    SmallVector<SmallVector<int64_t>> finalOffsets;
    finalOffsets.push_back(SmallVector<int64_t>(rank, 0));
    int64_t correctAxis;
    if (axis < 0) {
        correctAxis = axis + rank;
    } else {
        correctAxis = axis;
    }
    for (auto input : origOp.getInputs() | indexed) {
        auto inputShape = getShape(input.value());
        auto offsets = SmallVector<int64_t>(rank, 0);
        offsets[correctAxis] = inputShape[Dim(correctAxis)] + finalOffsets.back()[correctAxis];
        finalOffsets.push_back(offsets);
    }
    finalOffsets.pop_back();

    return getIntArrayOfArray(origOp.getContext(), finalOffsets);
}

mlir::LogicalResult ConvertPerAxisToOffsets::matchAndRewrite(VPU::ConcatOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (origOp.getStaticOffsetsAttr()) {
        return mlir::failure();
    }

    if (origOp.getPerAxisAttr().getStride() || origOp.getPerAxisAttr().getOffset()) {
        return mlir::failure();
    }

    const auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto axis = origOp.getPerAxisAttr().getAxis().getValue().getSExtValue();
    const auto finalOffsetsAttr = inferOffsetsAttrWithAxis(origOp, axis);

    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(origOp, outType, origOp.getInputs(), finalOffsetsAttr);
    return mlir::success();
}

}  // namespace

//
// FuseConcat
//

namespace {

class FuseConcat final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp op, mlir::PatternRewriter& rewriter) const final;
};

SmallVector<mlir::Value> getAllInputOp(VPU::ConcatOp origOp, const std::unordered_set<Dim>& axis) {
    SmallVector<mlir::Value> inputOps;
    for (auto preOps : origOp.getInputs()) {
        auto producerConcatOp = preOps.getDefiningOp<VPU::ConcatOp>();

        if (producerConcatOp != nullptr && producerConcatOp.getStaticOffsetsAttr()) {
            const auto subAxis = getConcatAxesFromOffsets(producerConcatOp, getShape(producerConcatOp.getOutput()));
            if (subAxis == axis) {
                for (auto inputTensor : producerConcatOp.getInputs()) {
                    inputOps.emplace_back(inputTensor);
                }
                continue;
            }
        }

        inputOps.emplace_back(preOps);
    }
    return inputOps;
}

mlir::LogicalResult FuseConcat::matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    // Dynamic concats will be lowered to SHAVE, which can only support two buffers being concatenated
    if (mlir::isa<Core::DynamicDimsMaskTensorType>(origOp.getOutput().getType())) {
        return mlir::failure();
    }

    if (origOp.getPerAxisAttr()) {
        return mlir::failure();
    }

    const auto axis = getConcatAxesFromOffsets(origOp, getShape(origOp.getOutput()));
    if (axis.size() != 1) {
        return mlir::failure();
    }

    const auto fuseInputs = getAllInputOp(origOp, axis);
    if (fuseInputs.size() <= origOp.getInputs().size()) {
        return mlir::failure();
    }

    const auto axisValue = *axis.begin();
    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(origOp, fuseInputs, axisValue.ind());

    return mlir::success();
}

//
// FuseConcatsWithDifferentAxes
//

class FuseConcatsWithDifferentAxes final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    FuseConcatsWithDifferentAxes(mlir::MLIRContext* ctx): mlir::OpRewritePattern<VPU::ConcatOp>(ctx) {
        setDebugName("FuseConcatsWithDifferentAxes");
    }

    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    static bool hasConcatProducer(const mlir::Value input);
    SmallVector<mlir::Value> flattenInputs(const mlir::ValueRange concatViewInputs) const;
    SmallVector<SmallVector<int64_t>> recalculateConcatOffsets(const mlir::Value concatInput,
                                                               ArrayRef<int64_t> origOpOffsets) const;
    SmallVector<SmallVector<int64_t>> recalculateOffsets(VPU::ConcatOp origOp) const;
};

bool FuseConcatsWithDifferentAxes::hasConcatProducer(const mlir::Value input) {
    auto concatOp = mlir::dyn_cast_or_null<VPU::ConcatOp>(input.getDefiningOp());
    // The producer must have static offsets defined.
    // Otherwise it's hard to recalculate the offsets properly.
    if (concatOp == nullptr || concatOp.getStaticOffsetsAttr() == nullptr || !concatOp.getStaticOffsets().has_value()) {
        return false;
    }

    // Fusion of VPU.Concat operations with multiple consumers results in scheduling errors. It is safe to fuse
    // VPU.Concat producer when all its consumers are VPU.Concat/VPU.Slice operations. See [E#75133].
    // Note that the below scenarios may cause performance regression, so skip fusing concat if so.
    // 1. slice is only on outer most dim, in which the fusion may cause inefficient dma ops.
    // 2. slice has concat op users, in which the fusion will break other concat related optimizations.
    auto isSliceNotOnOuterMostDim = [](mlir::Operation* op) {
        if (!mlir::isa<VPU::SliceOp>(op)) {
            return false;
        }
        auto usedByConcat = llvm::any_of(op->getUsers(), [](const mlir::Operation* user) {
            return mlir::isa<VPU::ConcatOp>(user);
        });
        if (usedByConcat) {
            return false;
        }
        auto inShape = getShape(op->getOperand(0));
        auto outShape = getShape(op->getResult(0));
        auto dimOrder = DimsOrder::fromValue(op->getOperand(0));
        auto highestDim = vpux::getHighestNonTrivialDim(inShape, dimOrder);
        if (!highestDim.has_value()) {
            return false;
        }
        auto sliceDim = IE::getDiffInOutSizeDims(inShape, outShape);
        return llvm::any_of(sliceDim, [&](auto dim) {
            return dim != highestDim.value();
        });
    };
    const auto isConcatOrSlice = [&](mlir::Operation* consumer) -> bool {
        return mlir::isa<VPU::ConcatOp>(consumer) || isSliceNotOnOuterMostDim(consumer);
    };
    const auto consumers = concatOp->getUsers();
    return std::all_of(consumers.begin(), consumers.end(), isConcatOrSlice);
}

// Propagate inputs from producer concat to consumer concat:
// %concat = VPU.Concat(%val0, %val1)
// VPU.Concat(%val2, %concat)
// Results in VPU.Concat(%val2, %val0, %val1)
SmallVector<mlir::Value> FuseConcatsWithDifferentAxes::flattenInputs(const mlir::ValueRange concatViewInputs) const {
    SmallVector<mlir::Value> newInputs;
    for (const auto& input : concatViewInputs) {
        if (hasConcatProducer(input)) {
            auto producerConcatOp = mlir::cast<VPU::ConcatOp>(input.getDefiningOp());
            const auto producerConcatInputs = producerConcatOp.getInputs();
            newInputs.append(producerConcatInputs.begin(), producerConcatInputs.end());
        } else {
            newInputs.push_back(input);
        }
    }
    return newInputs;
}

SmallVector<SmallVector<int64_t>> FuseConcatsWithDifferentAxes::recalculateConcatOffsets(
        const mlir::Value concatInput, ArrayRef<int64_t> origOpOffsets) const {
    SmallVector<SmallVector<int64_t>> recalculatedOffsets;
    auto producerConcatOp = concatInput.getDefiningOp<VPU::ConcatOp>();
    const auto oldOffsetsArr = producerConcatOp.getStaticOffsets().value().getAsRange<mlir::ArrayAttr>();
    for (const auto& oldOffsets : oldOffsetsArr) {
        auto offsets = parseIntArrayAttr<int64_t>(oldOffsets);
        VPUX_THROW_WHEN(offsets.size() != origOpOffsets.size(), "Rank of offsets mismatch: {0} vs {1}", offsets,
                        origOpOffsets);
        for (const auto& axis : irange(offsets.size())) {
            offsets[axis] += origOpOffsets[axis];
        }
        recalculatedOffsets.push_back(offsets);
    }
    return recalculatedOffsets;
}

// Consider VPU.Concat with concat axis C:
// %concat = VPU.Concat(%val0, %val1) { static_offsets = [[0, 0, 0, 0], [0, 0, 125, 0]] }
// VPU.Concat(%val2, %concat) { static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]] }
// In order to fuse the first VPU.Concat into the last, %val0 and %val1 must have offsets 64 by axis C:
// VPU.Concat(%val2, %val0, %val1) { static_offsets = [[0, 0, 0, 0], [[0, 64, 0, 0], [0, 64, 125, 0]] }
SmallVector<SmallVector<int64_t>> FuseConcatsWithDifferentAxes::recalculateOffsets(VPU::ConcatOp origOp) const {
    const auto concatViewInputs = origOp.getInputs();
    const auto origOpOffsets = origOp.getStaticOffsets().value().getAsRange<mlir::ArrayAttr>();
    SmallVector<SmallVector<int64_t>> newOffsets;
    for (const auto inputWithOffset : zip(concatViewInputs, origOpOffsets)) {
        const auto& input = std::get<0>(inputWithOffset);
        const auto origOpOffsets = parseIntArrayAttr<int64_t>(std::get<1>(inputWithOffset));
        if (hasConcatProducer(input)) {
            const auto offsets = recalculateConcatOffsets(input, ArrayRef(origOpOffsets));
            newOffsets.append(offsets.begin(), offsets.end());
        } else {
            newOffsets.push_back(origOpOffsets);
        }
    }

    return newOffsets;
}

mlir::LogicalResult FuseConcatsWithDifferentAxes::matchAndRewrite(VPU::ConcatOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    // Dynamic concats will be lowered to SHAVE, which can only support two buffers being concatenated
    if (mlir::isa<Core::DynamicDimsMaskTensorType>(origOp.getOutput().getType())) {
        return mlir::failure();
    }

    if (origOp.getStaticOffsetsAttr() == nullptr || !origOp.getStaticOffsets().has_value()) {
        return mlir::failure();
    }

    // Do not fuse if concat is strided
    if (origOp.getStridesAttr() != nullptr) {
        return mlir::failure();
    }

    const auto concatViewInputs = origOp.getInputs();
    if (std::none_of(concatViewInputs.begin(), concatViewInputs.end(), hasConcatProducer)) {
        return mlir::failure();
    }
    SmallVector<VPU::ConcatOp> concatProducers;
    for (const auto& input : concatViewInputs) {
        if (hasConcatProducer(input)) {
            concatProducers.push_back(input.getDefiningOp<VPU::ConcatOp>());
        }
    }

    /*Do not fuse concat with subgraph below, since the fuse will prevent the concat to become cmx concat. The condition
      check should be logically equal to the check in Pass OptimizeSharedInputCopyForConcat.
                  NCE/SW Op
                     |
                   Concat
                /          \
             Concat      Concat
               |            |
             Slice        Slice
               |            |
           NCE/SW Op    NCE/SW Op

    */
    auto checkConcatPattern = [&](mlir::Operation* op) {
        auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(op);
        if (concatOp == nullptr) {
            return false;
        }
        if (!concatOp.getStaticOffsets().has_value()) {
            return false;
        }
        auto concatAxes = getConcatAxesFromOffsets(concatOp, getShape(concatOp.getResult()));
        return llvm::all_of(op->getUsers(), [&](const auto& user) {
            auto sliceOp = mlir::dyn_cast<VPU::SliceOp>(user);
            if (sliceOp == nullptr || !sliceOp->hasOneUse()) {
                return false;
            }
            auto nextUser = *sliceOp->getUsers().begin();
            if (!mlir::isa<VPU::NCEOpInterface, VPU::SWOpInterface>(nextUser)) {
                return false;
            };
            // If the user of slice is NCE or SW op with multi cluster strategy, need to check if the tiling
            // axis is in concat axes. If so, the cmx concat optimization can not be used, so the concat should
            // keep fused in current rewriter.
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nextUser);
            if (clusteredOp == nullptr || !clusteredOp.getMultiClusterStrategy().has_value()) {
                return true;
            }

            const auto strategy = clusteredOp.getMultiClusterStrategy().value();
            const auto numCluster = clusteredOp.getOptimalNumClusters(getShape(clusteredOp->getResult(0)), strategy);
            auto inType = sliceOp.getResult().getType();
            SmallVector<int64_t> tilingScheme;
            if (mlir::isa<VPU::SWOpInterface>(nextUser)) {
                tilingScheme =
                        VPU::getSWInputTensorNumTiles(clusteredOp, numCluster, strategy, sliceOp.getResult(), inType);
            } else {
                auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(nextUser);
                auto isFilter = !mlir::isa<VPU::NCEEltwiseOp>(nceOp.getOperation()) &&
                                nceOp.getWeightsOperand() != nullptr &&
                                nceOp.getWeightsOperand().getDefiningOp() == sliceOp;
                tilingScheme = isFilter ? VPU::getWeightsTensorNumTiles(clusteredOp, inType, numCluster, strategy)
                                        : VPU::getActivationTensorNumTiles(clusteredOp, numCluster, strategy);
            }
            auto tilingOnDifferentDim = llvm::all_of(concatAxes, [&](const auto& dim) {
                return tilingScheme[dim.ind()] == 1;
            });
            return tilingOnDifferentDim;
        });
    };

    auto isSlicedForNceOrSwOp = llvm::any_of(concatProducers, [&](auto producer) {
        auto hasNceOrSwOpInput = llvm::any_of(producer.getInputs(), [](auto input) {
            return mlir::isa_and_nonnull<VPU::NCEOpInterface, VPU::SWOpInterface>(input.getDefiningOp());
        });
        return hasNceOrSwOpInput && VPU::hasMultiBranches(producer) &&
               llvm::all_of(producer->getUsers(), checkConcatPattern);
    });

    if (isSlicedForNceOrSwOp) {
        return mlir::failure();
    }

    // Fuse the inputs.
    const auto newInputs = flattenInputs(concatViewInputs);

    // Recalculate the offsets.
    const auto newOffsets = recalculateOffsets(origOp);

    auto opConcat = rewriter.create<VPU::ConcatOp>(origOp->getLoc(), origOp->getResult(0).getType(), newInputs,
                                                   getIntArrayOfArray(rewriter.getContext(), newOffsets));
    rewriter.replaceOp(origOp, opConcat->getResult(0));

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::ConcatOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ConvertPerAxisToOffsets>(ctx);
    results.add<FuseConcat>(ctx);
    results.add<FuseConcatsWithDifferentAxes>(ctx);
}

//
// fold
//

mlir::OpFoldResult VPU::ConcatOp::fold(FoldAdaptor) {
    if (getInputs().size() == 1) {
        return getInputs().front();
    }

    return nullptr;
}

namespace {
bool isSOHWithHeightConcatAxis(ShapeRef inputShape, ShapeRef outputShape, VPU::DistributionInfoAttr distributedAttr) {
    if (inputShape[Dims4D::Act::H] == outputShape[Dims4D::Act::H]) {
        // inputs are not concatenated over H
        return false;
    }

    return VPU::isSegmentedOverH(distributedAttr) || VPU::isOverlappedOverH(distributedAttr);
}
}  // namespace

//
// verify
//

mlir::LogicalResult vpux::VPU::ConcatOp::verify() {
    const auto loc = getLoc();

    if (getInputs().empty()) {
        return errorAt(loc, "Missing inputs for '{0}'", VPU::ConcatOp::getOperationName());
    }

    if (!getPerAxis().has_value() && !getStaticOffsets().has_value()) {
        return errorAt(loc, "Missing either 'per_axis' or 'static_offsets' attribute");
    }
    if (getPerAxis().has_value() && getStaticOffsets().has_value()) {
        return errorAt(loc, "Only one attribute ('per_axis' or 'static_offsets') should be provided");
    }

    auto input1DataType = getInputs().front().getType();
    auto input1SparseType = mlir::dyn_cast_or_null<VPU::SparseTensorType>(input1DataType);
    if (input1SparseType != nullptr) {
        input1DataType = input1SparseType.getData();
    }

    if (const auto inTypeRanked = mlir::dyn_cast<mlir::RankedTensorType>(input1DataType)) {
        // Check consistent tensor attributes

        const auto inDesc = vpux::getTensorAttr(inTypeRanked);

        for (const auto val : getInputs().drop_front()) {
            if (!mlir::isa<mlir::RankedTensorType, vpux::VPU::SparseTensorType>(val.getType())) {
                return errorAt(loc, "Misaligned tensor type for '{0}' inputs", getOperationName());
            }

            const auto curType = (input1SparseType != nullptr)
                                         ? mlir::cast<mlir::RankedTensorType>(
                                                   mlir::cast<vpux::VPU::SparseTensorType>(val.getType()).getData())
                                         : mlir::cast<mlir::RankedTensorType>(val.getType());
            const auto curDesc = vpux::getTensorAttr(curType);

            if (mlir::cast<vpux::NDTypeInterface>(val.getType()).getShape().isStatic()) {
                if (curDesc != inDesc) {
                    return errorAt(loc, "Misaligned TensorType attributes for '{0}' inputs", getOperationName());
                }
            } else {
                // Bounds may be different.
                // Other than that the descriptors must match.
                if (curDesc.getOrder() != inDesc.getOrder() || curDesc.getMemSpace() != inDesc.getMemSpace()) {
                    return errorAt(loc, "Misaligned TensorType attributes for '{0}' inputs", getOperationName());
                }
            }
        }
    } else if (const auto inTypeDistributed = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(input1DataType)) {
        const auto inOrder = inTypeDistributed.getOrder();
        const auto inMemSpace = inTypeDistributed.getMemSpace();
        const auto inDistribution = inTypeDistributed.getDistribution();
        const auto inShape = inTypeDistributed.getShape();
        const auto outShape = getShape(getOutput());

        // Check consistent distributed tensor attributes

        if (isSOHWithHeightConcatAxis(inShape, outShape, inDistribution)) {
            return errorAt(loc,
                           "Input is concatenated over H, but clustering mode is SEGMENTED or OVERLAPPED on H dim "
                           "for op {0}",
                           getOperationName());
        }

        for (const auto val : getInputs().drop_front()) {
            if (!mlir::isa<vpux::VPU::DistributedTensorType, vpux::VPU::SparseTensorType>(val.getType())) {
                return errorAt(loc, "Misaligned tensor type for '{0}' inputs", getOperationName());
            }

            const auto curType = (input1SparseType != nullptr)
                                         ? mlir::cast<vpux::VPU::DistributedTensorType>(
                                                   mlir::cast<vpux::VPU::SparseTensorType>(val.getType()).getData())
                                         : mlir::cast<vpux::VPU::DistributedTensorType>(val.getType());

            if (curType.getOrder() != inOrder || curType.getMemSpace() != inMemSpace) {
                return errorAt(loc, "Misaligned DistributedTensorType attributes for '{0}' inputs", getOperationName());
            }

            if (isSOHWithHeightConcatAxis(curType.getShape(), outShape, curType.getDistribution())) {
                return errorAt(loc,
                               "Input is concatenated over H, but clustering mode is SEGMENTED or OVERLAPPED on H dim "
                               "for op {0}",
                               getOperationName());
            }

            if (curType.getDistribution() != inDistribution) {
                if ((!VPU::isDistributedAttrWithExplicitShapesAndOffsets(curType.getDistribution()) ||
                     !VPU::isDistributedAttrWithExplicitShapesAndOffsets(inDistribution))) {
                    return errorAt(loc, "Misaligned DistributionInfoAttr for '{0}' inputs", getOperationName());
                }

                const auto curTypeShape = curType.getShape();

                const auto checkShapesOffsets =
                        [&](const SmallVector<SmallVector<int64_t>>& lhs,
                            const SmallVector<SmallVector<int64_t>>& rhs) -> mlir::LogicalResult {
                    for (const auto& pair : zip(lhs, rhs)) {
                        const auto shapesOffsetsLhs = std::get<0>(pair);
                        const auto shapesOffsetsRhs = std::get<1>(pair);

                        if (shapesOffsetsLhs.size() != shapesOffsetsRhs.size()) {
                            return errorAt(loc, "Per cluster shapes/offsets don't have the same rank for '{0}' inputs",
                                           getOperationName());
                        }

                        for (size_t dim = 0; dim < shapesOffsetsLhs.size(); dim++) {
                            // if dim is not a concatenation axis, check that shapes/offsets are the same for
                            // the two inputs
                            if (inShape[Dim(dim)] == curTypeShape[Dim(dim)] &&
                                inShape[Dim(dim)] == outShape[Dim(dim)]) {
                                if (shapesOffsetsLhs[dim] != shapesOffsetsRhs[dim]) {
                                    return errorAt(loc, "Misaligned per cluster shapes/offsets for '{0}' inputs",
                                                   getOperationName());
                                }
                            }
                        }
                    }

                    return mlir::success();
                };

                const auto input0PerClusterOffsets =
                        vpux::parseIntArrayOfArrayAttr<int64_t>(curType.getDistribution().getMemoryOffsets());
                const auto input1PerClusterOffsets =
                        vpux::parseIntArrayOfArrayAttr<int64_t>(inDistribution.getMemoryOffsets());

                const auto verifyMemoryOffsets = checkShapesOffsets(input0PerClusterOffsets, input1PerClusterOffsets);
                if (verifyMemoryOffsets.failed()) {
                    return verifyMemoryOffsets;
                }

                const auto input0PerClusterShapes =
                        vpux::parseIntArrayOfArrayAttr<int64_t>(curType.getDistribution().getMemoryShapes());
                const auto input1PerClusterShapes =
                        vpux::parseIntArrayOfArrayAttr<int64_t>(inDistribution.getMemoryShapes());

                const auto verifyMemoryShapes = checkShapesOffsets(input0PerClusterShapes, input1PerClusterShapes);
                if (verifyMemoryShapes.failed()) {
                    return verifyMemoryShapes;
                }
            }
        }
    } else {
        VPUX_THROW("Unsupported VPU::Concat type on input - `{0}`", input1DataType);
    }

    return mlir::success();
}

//
// NCEOpInterface
//

bool vpux::VPU::ConcatOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    // cmx_concat has accuracy issue with SOH inputs and Height concatenate. Add a workaround here to avoid
    // it happens in strategy manager pass.
    auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        auto outputShape = outputType.getShape();
        auto inputDataType = mlir::cast<vpux::NDTypeInterface>(getInputs().front().getType());
        auto inputShape = inputDataType.getShape();

        if (inputShape[Dims4D::Act::H] != outputShape[Dims4D::Act::H]) {
            return false;
        }
    }

    return strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::Clustering;
}

vpux::VPU::DistributionInfo vpux::VPU::ConcatOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return vpux::VPU::getConcatExplicitDistributedNative(shape, distributionMode, numTiles, numClusters, alignment,
                                                         uniformDistributedSegments, overlapParams);
}

bool VPU::ConcatOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& outputTile) {
    // Concat could have different H for input and output.
    // For example, there's a concat on H axis. It has two inputs with shape of 1x1x1x16 #NHWC and a output with shape
    // of 1x2x1x16 #NHWC. Then we can't assign SplitOverHeight to it because the input can't be tiled on H.
    if (auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(getOperation())) {
        auto moduleOp = concatOp->getParentOfType<mlir::ModuleOp>();
        auto tileOp = config::getTileExecutor(moduleOp);
        const auto minimumOutputHeightForSOH = tileOp.getCount();

        for (auto concatInput : concatOp.getInputs()) {
            auto IH = getShape(concatInput)[Dims4D::Act::H];
            if (IH < minimumOutputHeightForSOH) {
                return false;
            }
        }
    }

    return VPU::isOperationSplitOverHeightCompatible(getOperation(), outputTile);
}

bool VPU::ConcatOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::ConcatOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(moduleOp);
    const auto numTiles = tileOp.getCount();

    // Channel alignment is specific for NCE DPU operations and CMX CONCAT
    auto minChannelSize = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT * numTiles;

    // Concat could have different C for input and output.
    // For example, there's a concat on C axis. It has two inputs with shape of 1x1x1x16 #NHWC and a output with shape
    // of 1x1x1x32 #NHWC. Then we can't assign SplitOverKernel to it because the input can't be tiled on C.
    if (auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(getOperation())) {
        for (auto concatInput : concatOp.getInputs()) {
            auto IC = getShape(concatInput)[Dims4D::Act::C];
            if (IC < minChannelSize) {
                return false;
            }
        }
    }

    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::ConcatOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, SiblingOpsAnalysis& siblingsAnalysis,
                                        Byte reservedMem) {
    auto concatOp = mlir::cast<VPU::ConcatOp>(getOperation());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(concatOp->getResult(0).getType());
    auto numClusters = VPU::getOptimalNumClusters(concatOp, outputType.getShape(), strategy);

    SmallVector<Byte> buffersSize{VPU::getTotalAllocSizeWithDistribution(
            getOutput().getType(),
            getOutputDistributionAttrFromOp(concatOp, getOutput().getType(), numClusters, strategy, siblingsAnalysis))};

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::ConcatOp::fitIntoCMX(vpux::NDTypeInterface output, Byte reservedMem) {
    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers{output.getTotalAllocSize()};
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::ConcatOp::fitIntoCMX(vpux::NDTypeInterface output) {
    return fitIntoCMX(output, Byte(0));
}
