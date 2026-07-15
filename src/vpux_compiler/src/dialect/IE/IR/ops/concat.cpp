//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>

#include <numeric>
#include <unordered_set>

using namespace vpux;

//
// build
//

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                               ConcatAttr per_axis) {
    build(builder, state, inputs, per_axis, nullptr);
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                               mlir::IntegerAttr axis, mlir::IntegerAttr offset, mlir::IntegerAttr stride) {
    build(builder, state, inputs, ConcatAttr::get(builder.getContext(), axis, offset, stride));
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                               int64_t axis, int64_t offset, int64_t stride) {
    build(builder, state, inputs, getIntAttr(builder, axis), offset != 0 ? getIntAttr(builder, offset) : nullptr,
          stride != 1 ? getIntAttr(builder, stride) : nullptr);
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs, Dim axis,
                               int64_t offset, int64_t stride) {
    build(builder, state, inputs, axis.ind(), offset, stride);
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                               mlir::ValueRange inputs, mlir::ArrayAttr static_offsets) {
    build(builder, state, outType, inputs, nullptr, static_offsets);
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                               mlir::ValueRange inputs, ArrayRef<Shape> static_offsets) {
    const auto attrArr = to_small_vector(static_offsets | transformed([&](ShapeRef arr) -> mlir::Attribute {
                                             return getIntArrayAttr(builder, arr);
                                         }));

    build(builder, state, outType, inputs, builder.getArrayAttr(attrArr));
}

void vpux::IE::ConcatOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type outType,
                               mlir::ValueRange inputs, ArrayRef<ShapeRef> static_offsets) {
    const auto attrArr = to_small_vector(static_offsets | transformed([&](ShapeRef arr) -> mlir::Attribute {
                                             return getIntArrayAttr(builder, arr);
                                         }));

    build(builder, state, outType, inputs, builder.getArrayAttr(attrArr));
}

//
// inferReturnTypeComponents
//

namespace {

using GetShapeFunc = std::function<Shape(const mlir::Value)>;

Shape getDynamicShape(const mlir::Value val) {
    return mlir::cast<vpux::NDTypeInterface>(val.getType()).getShape().toValues();
}

Shape getUpperBounds(const mlir::Value val) {
    auto tensorType = val.getType();
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(tensorType)) {
        const auto bounds = boundedType.getBounds();
        return Shape(bounds.raw());
    }

    auto staticShape = mlir::cast<mlir::RankedTensorType>(tensorType).getShape();
    return Shape(staticShape);
}

mlir::FailureOr<Shape> inferOutShapeWithAxis(IE::ConcatOpAdaptor concat, const GetShapeFunc& getShapeFunctor,
                                             mlir::Location loc) {
    const auto axis = normalizeAxis(concat);

    Shape outShape = getShapeFunctor(concat.getInputs().front());

    for (const auto val : concat.getInputs().drop_front()) {
        const auto curShape = getShapeFunctor(val);

        if (curShape.size() != outShape.size()) {
            return errorAt(loc, "Concat inputs have mismatched ranks: '{0}' vs '{1}'", curShape.size(),
                           outShape.size());
        }

        if (outShape[axis] != mlir::ShapedType::kDynamic) {
            outShape[axis] += curShape[axis];
        }
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

    if (outShape[axis] == mlir::ShapedType::kDynamic) {
        const auto bounds = getUpperBounds(concat.getInputs().front());

        if (maxLatestIdx >= bounds[axis]) {
            return errorAt(loc, "Concat with offset '{0}' and stride '{1}' doesn't fit to output dimension '{2}'",
                           offset, stride, bounds[axis]);
        }
    } else {
        if (maxLatestIdx >= outShape[axis]) {
            return errorAt(loc, "Concat with offset '{0}' and stride '{1}' doesn't fit to output dimension '{2}'",
                           offset, stride, outShape[axis]);
        }
    }

    return outShape;
}

mlir::FailureOr<Shape> inferReturnShapeWithOffsets(IE::ConcatOpAdaptor concat, const GetShapeFunc& getShapeFunctor,
                                                   mlir::Location loc) {
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

    Shape outShape(checked_cast<size_t>(inType.getRank()), 0);

    for (const auto& p : zip(concat.getInputs(), allOffsets)) {
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

        for (const auto ind : irange(outShape.size())) {
            const auto d = Dim(ind);

            if (curShape[d] == mlir::ShapedType::kDynamic) {
                outShape[d] = curShape[d];
            } else {
                outShape[d] = std::max(outShape[d], curOffsets[d] + curShape[d]);
            }
        }
    }

    // TODO: validate that inputs+static_offsets fully covers the output without intersections

    return outShape;
}

mlir::FailureOr<mlir::Type> inferReturnElemTypeWithAxis(IE::ConcatOpAdaptor concat, mlir::Location loc) {
    SmallVector<mlir::Type> types;
    const auto getElemTypeFromValue = [](mlir::Value operand) {
        return mlir::cast<vpux::NDTypeInterface>(operand.getType()).getElementType();
    };
    std::transform(concat.getOperands().begin(), concat.getOperands().end(), std::back_inserter(types),
                   getElemTypeFromValue);

    const auto logCb = [loc](const formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };

    return inferOutElemTypeWithAxis(types, concat, logCb);
}

mlir::FailureOr<mlir::Type> inferReturnElemTypeWithOffsets(IE::ConcatOpAdaptor concat, ShapeRef outShape,
                                                           mlir::Location loc) {
    SmallVector<mlir::Type> types;
    const auto getElemTypeFromValue = [](mlir::Value operand) {
        return mlir::cast<vpux::NDTypeInterface>(operand.getType()).getElementType();
    };
    std::transform(concat.getOperands().begin(), concat.getOperands().end(), std::back_inserter(types),
                   getElemTypeFromValue);

    const auto logCb = [loc](const formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };

    return inferOutElemTypeWithOffsets(types, concat, outShape, logCb);
}

}  // namespace

mlir::LogicalResult vpux::IE::ConcatOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ConcatOpAdaptor concat(operands, attrs, prop);
    if (mlir::failed(concat.verify(loc))) {
        return mlir::failure();
    }

    if (concat.getInputs().empty()) {
        return errorAt(loc, "Missing inputs for '{0}'", IE::ConcatOp::getOperationName());
    }

    if (!concat.getPerAxis().has_value() && !concat.getStaticOffsets().has_value()) {
        return errorAt(loc, "Missing either 'per_axis' or 'static_offsets' attribute");
    }
    if (concat.getPerAxis().has_value() && concat.getStaticOffsets().has_value()) {
        return errorAt(loc, "Only one attribute ('per_axis' or 'static_offsets') should be provided");
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(concat.getInputs().front().getType());

    // Check consistent tensor attributes

    const auto inDesc = vpux::getTensorAttr(inType);

    for (const auto val : concat.getInputs().drop_front()) {
        const auto curType = mlir::cast<mlir::RankedTensorType>(val.getType());
        const auto curDesc = vpux::getTensorAttr(curType);

        if (curDesc != inDesc) {
            if ((!curType.hasStaticShape() && inType.hasStaticShape()) ||
                (curType.hasStaticShape() && !inType.hasStaticShape())) {
                continue;
            }
            // Bounds may be different.
            if (curDesc.getOrder() != inDesc.getOrder() || curDesc.getMemSpace() != inDesc.getMemSpace()) {
                return errorAt(loc, "Misaligned TensorType attributes for '{0}' inputs",
                               IE::ConcatOp::getOperationName());
            }
        }
    }

    // Infer output shape

    const auto outShape = concat.getPerAxis() ? inferOutShapeWithAxis(concat, getDynamicShape, loc)
                                              : inferReturnShapeWithOffsets(concat, getDynamicShape, loc);
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    // Infer output element type

    const auto outElemType = concat.getPerAxis() ? inferReturnElemTypeWithAxis(concat, loc)
                                                 : inferReturnElemTypeWithOffsets(concat, outShape.value(), loc);
    if (mlir::failed(outElemType)) {
        return mlir::failure();
    }

    // Return inferred components

    if (outShape.value().isStatic()) {
        // Return inferred components
        inferredReturnShapes.emplace_back(outShape.value().raw(), outElemType.value(), inDesc);
    } else {
        // Infer output bounds
        const auto outBounds = concat.getPerAxis() ? inferOutShapeWithAxis(concat, getUpperBounds, loc)
                                                   : inferReturnShapeWithOffsets(concat, getUpperBounds, loc);
        if (mlir::failed(outBounds)) {
            return mlir::failure();
        }

        // Return inferred components
        const auto outDesc =
                vpux::getTensorAttr(ctx, inDesc.getOrder(), inDesc.getMemSpace(), BoundsRef(outBounds.value().raw()));
        inferredReturnShapes.emplace_back(outShape.value().raw(), outElemType.value(), outDesc);
    }
    return mlir::success();
}

//
// ConvertPerAxisToOffsets
//

namespace {

class ConvertPerAxisToOffsets final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ConvertPerAxisToOffsets(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx, benefit) {
        this->setDebugName("ConvertPerAxisToOffsets");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertPerAxisToOffsets::matchAndRewrite(IE::ConcatOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (origOp.getStaticOffsetsAttr()) {
        return mlir::failure();
    }

    if (origOp.getPerAxisAttr().getStride() || origOp.getPerAxisAttr().getOffset()) {
        return mlir::failure();
    }

    const auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto axis = origOp.getPerAxisAttr().getAxis().getValue().getSExtValue();
    auto rank = mlir::cast<vpux::NDTypeInterface>(origOp.getInputs().front().getType()).getRank();
    // Negative value means counting dimension from the end
    if (axis < 0) {
        axis += rank;
    }
    if (outType.getShape()[Dim(axis)] == mlir::ShapedType::kDynamic) {
        return mlir::failure();
    }
    const auto finalOffsetsAttr = inferOffsetsAttrWithAxis(origOp, axis);

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, outType, origOp.getInputs(), finalOffsetsAttr);
    return mlir::success();
}

}  // namespace

//
// FuseConcat
//

namespace {

class FuseConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    FuseConcat(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx, benefit) {
        this->setDebugName("FuseConcat");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConcatOp op, mlir::PatternRewriter& rewriter) const final;
};

SmallVector<mlir::Value> getAllInputOp(IE::ConcatOp origOp, const std::unordered_set<Dim>& axis) {
    SmallVector<mlir::Value> inputOps;
    for (auto preOps : origOp.getInputs()) {
        auto producerConcatOp = preOps.getDefiningOp<IE::ConcatOp>();

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

mlir::LogicalResult FuseConcat::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.getPerAxisAttr()) {
        return mlir::failure();
    }

    const auto axis = getConcatAxesFromOffsets(origOp, getShape(origOp.getOutput()));
    if (axis.size() != 1) {
        return mlir::failure();
    }

    // Skip fuse multi-concat for dynamic case
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    if (!outputType.getShape().isStatic()) {
        return mlir::failure();
    }

    const auto fuseInputs = getAllInputOp(origOp, axis);
    if (fuseInputs.size() <= origOp.getInputs().size()) {
        return mlir::failure();
    }

    const auto axisValue = *axis.begin();
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, fuseInputs, axisValue.ind());

    return mlir::success();
}

}  // namespace

//
// FuseConstConcat
//

namespace {

class FuseConstConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    FuseConstConcat(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx, benefit) {
        this->setDebugName("FuseConstConcat");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConcatOp op, mlir::PatternRewriter& rewriter) const final;
};

SmallVector<mlir::Value> getAllConstInputOp(IE::ConcatOp origOp) {
    SmallVector<mlir::Value> inputOps;
    for (auto preOps : origOp.getInputs()) {
        auto constOp = preOps.getDefiningOp<Const::DeclareOp>();

        if (constOp != nullptr) {
            inputOps.emplace_back(constOp);
        }
    }
    return inputOps;
}

mlir::LogicalResult FuseConstConcat::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    //        Const  Const  Const
    //          |      |      |
    //           \     |     /         =>   Const [#const.Concat<...>]
    //              Concat
    //                 |
    //
    if (origOp.getPerAxisAttr()) {
        return mlir::failure();
    }

    const auto constInputs = getAllConstInputOp(origOp);
    if (constInputs.size() != origOp.getInputs().size()) {
        return mlir::failure();
    }

    if (!origOp.getStaticOffsets().has_value()) {
        return mlir::failure();
    }
    auto staticOffsets = origOp.getStaticOffsets().value();

    const auto axis = getConcatAxesFromOffsets(origOp, getShape(origOp.getOutput()));
    if (axis.size() != 1) {
        return mlir::failure();
    }

    std::vector<Const::ContentAttr> inputContents;
    inputContents.reserve(constInputs.size());
    for (auto input : constInputs) {
        auto declareOp = input.getDefiningOp<Const::DeclareOp>();
        inputContents.push_back(declareOp.getContentAttr());
    }

    const auto axisValue = axis.begin()->ind();
    auto contentAttr = Const::createConcatContentAttr(rewriter.getContext(), staticOffsets, axisValue, inputContents);

    rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, origOp.getType(), contentAttr);
    return mlir::success();
}

}  // namespace

//
// FuseSliceConcat
//

namespace {

class FuseSliceConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    FuseSliceConcat(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx, benefit) {
        this->setDebugName("FuseSliceConcat");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConcatOp op, mlir::PatternRewriter& rewriter) const final;
};

bool doSlicesRepresentFullParent(ArrayRef<IE::SliceOp> sliceOps, int32_t axis) {
    auto firstSlice = sliceOps[0];
    auto parentOp = firstSlice.getSource();
    const auto outShape = vpux::getShape(parentOp);
    auto processedShape = SmallVector<int64_t>(firstSlice.getStaticOffsets().size(), 0);
    auto compareCond = [](auto offset, auto procShape) {
        return (offset == 0 || offset == procShape);
    };
    auto processAxes = [](auto offset, auto dimShape) {
        return offset + dimShape;
    };
    auto isTrueCond = [](auto condition) {
        return condition;
    };
    for (auto sliceOp : sliceOps) {
        const auto offset = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
        const auto shape = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
        SmallVector<bool> cond(offset.size(), false);
        std::transform(offset.begin(), offset.end(), processedShape.begin(), cond.begin(), compareCond);
        auto greaterThan1DimCount = std::count_if(offset.begin(), offset.end(), [](auto item) {
            return item > 1;
        });
        if (!std::all_of(cond.begin(), cond.end(), isTrueCond) || greaterThan1DimCount > 1) {
            return false;
        }
        auto sliceAxes = vpux::IE::getDiffInOutSizeDims(getShape(sliceOp.getSource()), getShape(sliceOp.getResult()));
        if (sliceAxes.empty() || sliceAxes.size() != 1) {
            return false;
        }
        if (sliceAxes.front().ind() != axis) {
            return false;
        }
        std::transform(offset.begin(), offset.end(), shape.begin(), processedShape.begin(), processAxes);
    }

    SmallVector<int64_t> realInputShape = to_small_vector(outShape);
    SmallVector<bool> retCond(realInputShape.size(), false);
    std::transform(processedShape.begin(), processedShape.end(), realInputShape.begin(), retCond.begin(), compareCond);
    return std::all_of(retCond.begin(), retCond.end(), isTrueCond);
}

SmallVector<mlir::Value> getFoldInputsOp(IE::ConcatOp origOp, int32_t axis) {
    SmallVector<IE::SliceOp> sameParentSliceOps;
    SmallVector<mlir::Value> inputOps;
    mlir::Value parent = nullptr;
    auto handleLastSlice = [&](mlir::Value sliceParent) {
        if (sameParentSliceOps.empty()) {
            return;
        }
        if (doSlicesRepresentFullParent(sameParentSliceOps, axis)) {
            inputOps.emplace_back(sliceParent);
        } else {
            for (auto& sliceOp : sameParentSliceOps) {
                inputOps.emplace_back(sliceOp);
            }
        }
        sameParentSliceOps.clear();
    };
    for (const auto& perOps : origOp.getInputs()) {
        auto sliceOp = perOps.getDefiningOp<IE::SliceOp>();

        if (sliceOp != nullptr) {
            auto currentParent = sliceOp.getSource();
            if (currentParent != parent) {
                handleLastSlice(parent);
                parent = currentParent;
            }
            sameParentSliceOps.emplace_back(sliceOp);
        } else {
            handleLastSlice(parent);
            inputOps.emplace_back(perOps);
        }
    }
    // Process the concat's last parameter is SliceOp
    handleLastSlice(parent);
    return inputOps;
}

mlir::LogicalResult FuseSliceConcat::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    //
    // Delete the Slice to avoid the Stride DMA when the sliceOps can represent the slice input.
    //             OP1
    //          /      \                        OP1     OP2
    //          |      |                         |       |
    //        Slice  Slice   OP2           =>    |       |
    //          |      |      |                  \       /
    //           \     |     /                     Concat
    //              Concat                           |
    //                 |
    //
    if (origOp.getPerAxisAttr()) {
        return mlir::failure();
    }

    const auto axis = getConcatAxesFromOffsets(origOp, getShape(origOp.getOutput()));
    if (axis.size() != 1) {
        return mlir::failure();
    }

    const auto axisValue = (*axis.begin()).ind();
    auto newInputs = getFoldInputsOp(origOp, axisValue);
    if (newInputs.size() >= origOp.getInputs().size()) {
        return mlir::failure();
    }

    if (newInputs.size() > 1) {
        rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, newInputs, axisValue);
    } else {
        rewriter.replaceAllUsesWith(origOp, newInputs[0]);
    }

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ConcatOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ConvertPerAxisToOffsets>(ctx);
    results.add<FuseConcat>(ctx);
    results.add<FuseSliceConcat>(ctx);
    results.add<FuseConstConcat>(ctx);
}

void vpux::IE::registerConcatOpRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                         size_t index) {
    registry.registerRewriter<ConvertPerAxisToOffsets>("convert-per-axis-to-offsets", benefitLevels[index]);
    registry.registerRewriter<FuseConcat>("fuse-concat", benefitLevels[index]);
    registry.registerRewriter<FuseSliceConcat>("fuse-slice-concat", benefitLevels[index]);
    registry.registerRewriter<FuseConstConcat>("fuse-const-concat", benefitLevels[index]);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ConcatOp::fold(FoldAdaptor) {
    if (getInputs().size() == 1) {
        return getInputs().front();
    }

    return nullptr;
}

namespace {
mlir::Attribute dispatchStaticDim(mlir::OpBuilder& builder, const int64_t dimIdx, const int64_t concatAxis,
                                  const mlir::ValueRange operands) {
    if (concatAxis != dimIdx) {
        // Concatenation axis does not match current dimension:
        // IE.Concat(1x2x3x?, 1x2x3x?) {concatAxis = 1, dimIdx = 2} -> 1x4x3x?
        // Fetch the dimension value from any input (outShape[dimIdx] = inShape[dimIdx] = 3)
        const auto inputShapedType = mlir::cast<mlir::ShapedType>(operands.front().getType());
        return builder.getIndexAttr(inputShapedType.getDimSize(dimIdx));
    }
    // Concatenation axis matches current dimension:
    // IE.Concat(1x2x3x?, 1x2x3x?) {concatAxis = 1, dimIdx = 1} -> 1x4x3x?
    // Accumulate shape[dimIdx] values over all the inputs
    // outShape[dimIdx] = in1Shape[dimIdx] + in2Shape[dimIdx]= 2 + 2 = 4
    const auto accumulateDimSizes = [dimIdx](const int64_t acc, const mlir::Value operand) {
        return acc + mlir::cast<mlir::ShapedType>(operand.getType()).getDimSize(dimIdx);
    };
    const int64_t dimSize = std::accumulate(operands.begin(), operands.end(), int64_t{0}, accumulateDimSizes);
    return builder.getIndexAttr(dimSize);
}

mlir::Value dispatchDynamicDim(mlir::OpBuilder& builder, const int64_t dimIdx, const mlir::ValueRange operands) {
    // Concatenation over static dimension.
    // Apply DimOp to any dynamic input.
    const auto firstDynOpIter = std::find_if(operands.begin(), operands.end(), IE::hasDynamicShapeAttr);
    if (firstDynOpIter == operands.end()) {
        return nullptr;
    }
    const auto firstDynamicOperand = *firstDynOpIter;
    mlir::OpFoldResult firstDimOp = builder.createOrFold<mlir::tensor::DimOp>(
            appendLoc(firstDynamicOperand.getLoc(), "dim_{0}", dimIdx), firstDynamicOperand, dimIdx);
    return mlir::getValueOrCreateConstantIndexOp(builder, firstDynamicOperand.getLoc(), firstDimOp);
}
};  // namespace

mlir::LogicalResult vpux::IE::ConcatOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                          mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    const auto operands = getInputs();
    if (operands.empty()) {
        return mlir::failure();
    }
    const auto concatAxis = getConcatAxis(*this);
    if (!concatAxis.has_value()) {
        return mlir::failure();
    }
    const auto axisValue = concatAxis.value().ind();

    const auto outputShapedType = mlir::cast<mlir::ShapedType>(getOutput().getType());
    if (outputShapedType.isDynamicDim(axisValue)) {
        // Concatenation over dynamic dimension is not supported
        return mlir::failure();
    }

    SmallVector<mlir::OpFoldResult> shapes;
    for (const auto& dimIdx : irange(outputShapedType.getRank())) {
        if (outputShapedType.isDynamicDim(dimIdx)) {
            const mlir::Value dynamicDim = dispatchDynamicDim(builder, dimIdx, operands);
            if (dynamicDim == nullptr) {
                return mlir::failure();
            }
            shapes.push_back(dynamicDim);
        } else {
            // Technically, reification does not care much about the correctness of static dimensions.
            // This case might've returned inputShapedType.getDimSize(dimIdx)
            shapes.push_back(dispatchStaticDim(builder, dimIdx, axisValue, operands));
        }
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));
    return mlir::success();
}
