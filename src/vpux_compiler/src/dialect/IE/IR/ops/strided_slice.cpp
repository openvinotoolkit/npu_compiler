//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/PatternMatch.h>

#include <optional>

using namespace vpux;

namespace {

struct StridedSliceInputData final {
    SmallVector<int64_t> begins;
    SmallVector<int64_t> ends;
    SmallVector<int64_t> strides;
};

SmallVector<int64_t> extractValues(mlir::Location loc, mlir::Value input, mlir::ArrayAttr attr) {
    if (input != nullptr) {
        auto values = IE::constInputToData(loc, input);
        if (mlir::succeeded(values)) {
            return values.value();
        }
    }
    if (attr != nullptr) {
        return parseIntArrayAttr<int64_t>(attr);
    }
    return {};
}

StridedSliceInputData extractData(mlir::Location loc, IE::StridedSliceOpAdaptor stridedSlice) {
    auto beginsVal = extractValues(loc, stridedSlice.getBegins(), stridedSlice.getBeginsAttr().value_or(nullptr));
    auto endsVal = extractValues(loc, stridedSlice.getEnds(), stridedSlice.getEndsAttr().value_or(nullptr));
    auto stridesVal = extractValues(loc, stridedSlice.getStrides(), stridedSlice.getStridesAttr().value_or(nullptr));

    return StridedSliceInputData{std::move(beginsVal), std::move(endsVal), std::move(stridesVal)};
}
}  // namespace

mlir::LogicalResult vpux::IE::StridedSliceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::StridedSliceOpAdaptor slice(operands, attrs, prop);
    if (mlir::failed(slice.verify(loc))) {
        return mlir::failure();
    }

    const auto inputData = extractData(loc, slice);
    const auto beginsShape =
            slice.getBegins() != nullptr
                    ? SmallVector<int64_t>(mlir::cast<mlir::ShapedType>(slice.getBegins().getType()).getShape())
                    : SmallVector<int64_t>{};
    const auto endsShape =
            slice.getEnds() != nullptr
                    ? SmallVector<int64_t>(mlir::cast<mlir::ShapedType>(slice.getEnds().getType()).getShape())
                    : SmallVector<int64_t>{};
    const auto stridesShape =
            slice.getStrides() != nullptr
                    ? SmallVector<int64_t>(mlir::cast<mlir::ShapedType>(slice.getStrides().getType()).getShape())
                    : SmallVector<int64_t>{};

    const auto beginMask = parseIntArrayAttr<int64_t>(slice.getBeginMask());
    const auto endMask = parseIntArrayAttr<int64_t>(slice.getEndMask());
    const auto newAxisMask = parseIntArrayAttr<int64_t>(slice.getNewAxisMask());
    const auto shrinkAxisMask = parseIntArrayAttr<int64_t>(slice.getShrinkAxisMask());
    const auto ellipsisMask = parseIntArrayAttr<int64_t>(slice.getEllipsisMask());

    const auto inDataType = mlir::cast<NDTypeInterface>(slice.getInput().getType());
    const auto inShapeInfo = ShapeInfo::fromNDType(inDataType);
    auto outputShapeInfo = inferStridedSliceOutputShape(inShapeInfo, inputData.begins, inputData.ends,
                                                        inputData.strides, beginsShape, endsShape, stridesShape,
                                                        beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);

    auto outputShape = outputShapeInfo.shape;
    if (outputShape.empty()) {
        outputShape.push_back(1);
    }

    auto inType = mlir::cast<vpux::NDTypeInterface>(slice.getInput().getType());
    const auto outType = mlir::cast<mlir::RankedTensorType>(inType.changeShape(ShapeRef(outputShape)));

    const auto inDataTensorType = mlir::cast<mlir::RankedTensorType>(slice.getInput().getType());
    auto outBounds = !outputShapeInfo.bounds.empty() ? outputShapeInfo.bounds : SmallVector<int64_t>{};
    auto outDesc =
            vpux::getTensorAttr(ctx, vpux::getOrder(inDataTensorType), /*memSpace=*/nullptr, BoundsRef(outBounds));

    inferredReturnShapes.emplace_back(outType.getShape(), outType.getElementType(), outDesc);

    return mlir::success();
}

bool vpux::IE::StridedSliceOp::isSimplified() {
    auto isZero = [](auto val) {
        return val == 0;
    };
    auto isPositive = [](auto val) {
        return val >= 0;
    };

    return (llvm::all_of(parseIntArrayAttr<int64_t>(getNewAxisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getShrinkAxisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEllipsisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getBeginMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEndMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getBeginsAttr().value()), isPositive) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEndsAttr().value()), isPositive));
}

bool hasNegativeStride(ArrayRef<int64_t> strides) {
    auto isNegative = [](auto val) {
        return val < 0;
    };
    return (llvm::any_of(strides, isNegative));
}
//
// fold
//

mlir::OpFoldResult vpux::IE::StridedSliceOp::fold(FoldAdaptor) {
    auto strides = getStridesAttr();
    if (!strides.has_value()) {
        return nullptr;
    }
    // In case StridedSlice op has negative value in stride input, which means reversing
    // the data along the axis, we can not fold for such case.
    if (!hasNegativeStride(parseIntArrayAttr<int64_t>(strides.value())) &&
        getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    // TODO E#22568: attempt const folding but only if slice isSimplified()

    return nullptr;
}

//
// ComposeStridedSlice
//

namespace {

class ComposeStridedSlice final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ComposeStridedSlice::matchAndRewrite(IE::StridedSliceOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto producerSliceOp = origOp.getInput().getDefiningOp<IE::StridedSliceOp>();
    if (producerSliceOp == nullptr) {
        return mlir::failure();
    }

    if (!(origOp.isSimplified() && producerSliceOp.isSimplified())) {
        return mlir::failure();
    }

    const auto firstBegin = parseIntArrayAttr<int64_t>(producerSliceOp.getBeginsAttr().value());
    const auto nextBegin = parseIntArrayAttr<int64_t>(origOp.getBeginsAttr().value());
    auto resultBegin = SmallVector<int64_t>(nextBegin.size());

    const auto firstEnd = parseIntArrayAttr<int64_t>(producerSliceOp.getEndsAttr().value());
    const auto nextEnd = parseIntArrayAttr<int64_t>(origOp.getEndsAttr().value());
    auto resultEnd = SmallVector<int64_t>(nextEnd.size());

    const auto firstStride = parseIntArrayAttr<int64_t>(producerSliceOp.getStridesAttr().value());
    const auto nextStride = parseIntArrayAttr<int64_t>(origOp.getStridesAttr().value());
    auto resultStride = SmallVector<int64_t>(nextStride.size());

    for (auto i : irange(firstBegin.size())) {
        resultBegin[i] = firstBegin[i] + nextBegin[i] * firstStride[i];
        resultEnd[i] = firstBegin[i] + nextEnd[i] * firstStride[i];
        resultStride[i] = firstStride[i] * nextStride[i];
    }

    // Stride on more than 2 axis is not supported
    const auto greaterThanOne = [](auto ele) {
        return ele > 1;
    };
    auto stridesGreaterThanOne = llvm::count_if(resultStride, greaterThanOne);
    if (stridesGreaterThanOne > 2) {
        return mlir::failure();
    }

    const auto beginsAttr = getIntArrayAttr(getContext(), resultBegin);
    const auto endsAttr = getIntArrayAttr(getContext(), resultEnd);
    const auto stridesAttr = getIntArrayAttr(getContext(), resultStride);

    const auto fusedLoc = takeOpLoc(origOp, "composed");
    const auto newOp = rewriter.create<IE::StridedSliceOp>(
            fusedLoc, producerSliceOp.getInput(), origOp.getBegins(), origOp.getEnds(), origOp.getStrides(), beginsAttr,
            endsAttr, stridesAttr, origOp.getBeginMask(), origOp.getEndMask(), origOp.getNewAxisMask(),
            origOp.getShrinkAxisMask(), origOp.getEllipsisMask());
    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    using mlir::OpRewritePattern<IE::StridedSliceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp slice, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::StridedSliceOp slice,
                                                        mlir::PatternRewriter& rewriter) const {
    if (!slice.getBegins() || !slice.getEnds() || !slice.getStrides()) {
        return mlir::failure();
    }

    const auto inputData = extractData(slice.getLoc(), slice);
    const auto beginsAttr = !inputData.begins.empty() ? getIntArrayAttr(getContext(), inputData.begins) : nullptr;
    const auto endsAttr = !inputData.ends.empty() ? getIntArrayAttr(getContext(), inputData.ends) : nullptr;
    const auto stridesAttr = !inputData.strides.empty() ? getIntArrayAttr(getContext(), inputData.strides) : nullptr;

    const auto begins = beginsAttr == nullptr ? slice.getBegins() : mlir::TypedValue<mlir::RankedTensorType>{nullptr};
    const auto ends = endsAttr == nullptr ? slice.getEnds() : mlir::TypedValue<mlir::RankedTensorType>{nullptr};
    const auto strides =
            stridesAttr == nullptr ? slice.getStrides() : mlir::TypedValue<mlir::RankedTensorType>{nullptr};

    rewriter.replaceOpWithNewOp<IE::StridedSliceOp>(
            slice, slice.getInput(), begins, ends, strides, beginsAttr, endsAttr, stridesAttr, slice.getBeginMask(),
            slice.getEndMask(), slice.getNewAxisMask(), slice.getShrinkAxisMask(), slice.getEllipsisMask());
    return mlir::success();
}

//
// ConvertNegStrideStridedSlice2Reverse
//

class ConvertNegStrideStridedSlice2Reverse final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    using mlir::OpRewritePattern<IE::StridedSliceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp stridedSliceOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertNegStrideStridedSlice2Reverse::matchAndRewrite(IE::StridedSliceOp slice,
                                                                          mlir::PatternRewriter& rewriter) const {
    if (!slice.getBegins() || !slice.getEnds() || !slice.getStrides()) {
        return mlir::failure();
    }

    const auto inputData = extractData(slice.getLoc(), slice);
    const auto strides = inputData.strides;

    auto countNegativeStrides = [](ArrayRef<int64_t> strides) {
        auto isNegative = [](auto val) {
            return val < 0;
        };
        return std::count_if(strides.begin(), strides.end(), isNegative);
    };

    // Currently, only reverse along 1 axis (from the perspective of functionality and performance) supported, the
    // StridedSlice can be replaced by ReverseSequenceOp
    auto negativeStridesCount = countNegativeStrides(strides);
    if (negativeStridesCount != 1) {
        return mlir::failure();
    }

    const auto inDataType = mlir::cast<vpux::NDTypeInterface>(slice.getInput().getType());
    const auto inDataShape = inDataType.getShape().raw();
    const auto begins = inputData.begins;
    const auto ends = inputData.ends;
    SmallVector<int64_t> seqLen;
    int64_t seqAxis = 0;
    for (const auto& p : strides | indexed) {
        if (p.value() < 0) {
            if (std::abs(ends[p.index()]) < inDataShape[p.index()] || begins[p.index()] != -1) {
                return mlir::failure();
            }
            seqAxis = p.index();
            seqLen.push_back(inDataShape[p.index()]);
        }
    }

    const auto seqLenStorageType =
            mlir::RankedTensorType::get({static_cast<int64_t>(seqLen.size())}, getSInt64Type(rewriter.getContext()));
    const auto seqLenData = Const::createConst(rewriter, slice.getLoc(), seqLenStorageType, ArrayRef(seqLen),
                                               [&](Const::ContentSetup& setup) {
                                                   return setup.castElemType(getSInt32Type(rewriter.getContext()));
                                               });
    rewriter.replaceOpWithNewOp<IE::ReverseSequenceOp>(slice, slice.getInput(), seqLenData,
                                                       getIntAttr(rewriter.getContext(), seqAxis),
                                                       getIntAttr(rewriter.getContext(), 0));
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::StridedSliceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                           mlir::MLIRContext* context) {
    patterns.add<ConvertNegStrideStridedSlice2Reverse>(context);
    patterns.add<ConvertConstToAttr>(context);
    patterns.add<ComposeStridedSlice>(context);
}

//
// inferElemTypeInfo
//

void vpux::IE::StridedSliceOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
    propagateElementTypeDown(info);
}

void vpux::IE::StridedSliceOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
    propagateElementTypeUp(info);
}

//
// verify
//

mlir::LogicalResult vpux::IE::StridedSliceOp::verify() {
    auto beginsAttr = getBeginsAttr();
    auto endsAttr = getEndsAttr();
    auto stridesAttr = getStridesAttr();

    if (beginsAttr.has_value() && endsAttr.has_value()) {
        if (beginsAttr.value().size() != endsAttr.value().size()) {
            return errorAt(*this, "Lower bounds and Upper bounds needs to have same number of values");
        }
    }

    if (beginsAttr.has_value() && stridesAttr.has_value()) {
        if (beginsAttr.value().size() != stridesAttr.value().size()) {
            return errorAt(*this, "Lower bounds and strides needs to have same number of values");
        }
    }

    if (endsAttr.has_value() && stridesAttr.has_value()) {
        if (endsAttr.value().size() != stridesAttr.value().size()) {
            return errorAt(*this, "Upper bounds and strides needs to have same number of values");
        }
    }

    return mlir::success();
}
