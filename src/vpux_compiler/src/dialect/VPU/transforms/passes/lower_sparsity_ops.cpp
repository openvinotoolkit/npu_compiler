//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/sparsity.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Transforms/DialectConversion.h>

#include <limits>

namespace vpux::VPU {
#define GEN_PASS_DECL_LOWERSPARSITYOPS
#define GEN_PASS_DEF_LOWERSPARSITYOPS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

mlir::Value createActSparsityMap(mlir::PatternRewriter& rewriter, mlir::Type type) {
    auto dataType = mlir::cast<vpux::NDTypeInterface>(type);
    auto ctx = rewriter.getContext();
    if (auto sparseType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(type)) {
        dataType = mlir::cast<vpux::NDTypeInterface>(sparseType.getData());
    }
    const auto sparsityMapType = mlir::cast<mlir::RankedTensorType>(
            dataType.changeElemType(mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Signless)));

    return Const::createConst(rewriter, mlir::UnknownLoc::get(ctx), sparsityMapType,
                              /*splatValue=*/ArrayRef<bool>(true));
}

std::tuple<mlir::Value, Shape> insertExpandToAlign(mlir::PatternRewriter& rewriter, mlir::Value input,
                                                   int64_t alignment) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());

    const auto inputShape = inputType.getShape();
    const auto dimC = Dims4D::Act::C;

    auto expandedShape = inputShape.toValues();
    expandedShape[dimC] = alignValUp(inputShape[dimC], alignment);
    SmallVector<int64_t> padsBegin(inputShape.size(), 0);
    SmallVector<int64_t> padsEnd(inputShape.size(), 0);
    padsEnd[dimC.ind()] = expandedShape[dimC] - inputShape[dimC];

    const mlir::Type outputType = inputType.changeShape(expandedShape);
    auto ctx = rewriter.getContext();
    auto expandOp = rewriter.create<VPU::ExpandOp>(input.getLoc(), outputType, input, getIntArrayAttr(ctx, padsBegin),
                                                   getIntArrayAttr(ctx, padsEnd));
    return std::make_tuple(expandOp.getOutput(), expandedShape);
}

mlir::ArrayAttr getIdentityDimPermutation(mlir::MLIRContext* ctx) {
    SmallVector<SmallVector<int64_t>> permutation = {{0}, {1}, {2}, {3}};
    return getIntArrayOfArray(ctx, permutation);
}

std::tuple<mlir::Value, Shape> insertAlignmentReshape(mlir::PatternRewriter& rewriter, mlir::Value input,
                                                      int64_t alignment) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());

    const auto inputShape = inputType.getShape();
    const auto totalElements = inputShape.totalSize();
    // Trying to uniformely distribute elements
    const auto desiredAxisSize = checked_cast<int64_t>(std::cbrt(totalElements));
    auto numC = alignValUp(desiredAxisSize, alignment);
    if (totalElements % numC != 0) {
        numC = alignment;
    }
    const auto spatialRemainder = totalElements / numC;
    // approximate square shape of spatial
    int64_t numH = checked_cast<int64_t>(std::floor(std::sqrt(spatialRemainder)));
    while (numH > 1 && spatialRemainder % numH != 0) {
        --numH;
    }
    const auto numW = spatialRemainder / numH;
    const Shape newShape{1, numC, numH, numW};
    VPUX_THROW_WHEN(newShape.totalSize() != totalElements,
                    "New shape '{0}' doesnt contain same number of elements as original '{1}'", newShape, inputShape);
    const auto ctx = input.getContext();
    auto reshapeOp =
            rewriter.create<VPU::AffineReshapeOp>(input.getLoc(), inputType.changeShape(newShape), input,
                                                  getIdentityDimPermutation(ctx), getIntArrayAttr(ctx, newShape));
    return std::make_tuple(reshapeOp.getOutput(), newShape);
}

mlir::LogicalResult rewriteSparsityOpWithEltwiseOp(mlir::PatternRewriter& rewriter, mlir::Operation* origOp,
                                                   vpux::Logger log, StringRef debugName) {
    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(log, rewriter, origOp, "[{0}] {1}", debugName, msg.str());
    };

    auto ctx = origOp->getContext();

    auto input = origOp->getOperand(0);
    auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());

    auto output = origOp->getResult(0);
    auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
    const auto originalOutputType = outputType;

    auto defaultQuantElemType = mlir::quant::UniformQuantizedType::get(
            /*flags=*/0, getUInt8Type(ctx), mlir::Float16Type::get(ctx), /*scale=*/1.0,
            /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);
    auto quantizedPerAxis = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputType.getElementType());
    if (quantizedPerAxis) {
        auto quantCastOp = rewriter.create<VPU::QuantizeCastOp>(origOp->getLoc(), input, defaultQuantElemType);
        input = quantCastOp.getOutput();
        inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
        outputType = outputType.changeElemType(defaultQuantElemType);
    }

    const auto maybeRequantizedOutputType = outputType;

    auto alignment = vpux::VPU::NCEInvariant::getAlignment(inputType.getElementType());
    bool needAlignment = !vpux::VPU::NCEInvariant::isAligned(inputType, alignment, logCb);
    bool isAlignmentResolvedByReshape = false;
    const auto originalShape = outputType.getShape();
    Shape alignedShape;
    if (needAlignment) {
        if (originalShape.totalSize() % alignment == 0) {
            // Eltwise result do not depend on shape, so Eltwise can be wrapped by reshapes to fit alignment
            // requirements
            std::tie(input, alignedShape) = insertAlignmentReshape(rewriter, input, alignment);
            isAlignmentResolvedByReshape = true;
        } else {
            std::tie(input, alignedShape) = insertExpandToAlign(rewriter, input, alignment);
        }
        inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
        outputType = outputType.changeShape(alignedShape);
    }

    if (!VPU::isNCEEltwiseSupported(origOp, inputType, inputType, outputType,
                                    /*allowDifferentScales=*/true,
                                    /*allowDifferentZp=*/true, /*checkLayout=*/true, /*checkChannelAlignment=*/true,
                                    logCb)) {
        return matchFailed(log, rewriter, origOp, "Cannot lower operation to NCE Eltwise because of HW requirements");
    }

    // To keep values half of scale is needed
    auto outputTypeForPPEAttr = outputType;
    auto elementType = outputTypeForPPEAttr.getElementType();
    // dummy filled type with half of scale for fp16 case
    auto newType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, getInt32Type(ctx),
                                                          rewriter.getF16Type(),
                                                          /*scale=*/2.0,
                                                          /*zeroPoint=*/0,
                                                          /*storageTypeMin=*/std::numeric_limits<int32_t>::min(),
                                                          /*storageTypeMax=*/std::numeric_limits<int32_t>::max());

    if (auto qOutputType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elementType)) {
        const auto newScale = qOutputType.getScale() * 2.;
        // For real quantType we should use real values, except scale
        newType = mlir::quant::UniformQuantizedType::get(0, getUInt8Type(ctx), rewriter.getF16Type(), newScale,
                                                         qOutputType.getZeroPoint(), qOutputType.getStorageTypeMin(),
                                                         qOutputType.getStorageTypeMax());
    }
    outputTypeForPPEAttr = outputTypeForPPEAttr.changeElemType(newType);

    const auto opType = VPU::EltwiseType::ADD;
    const auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);

    auto inputPaddingAttr = origOp->hasAttr(VPU::INPUT_PADDING_ATTR_NAME)
                                    ? mlir::cast<mlir::ArrayAttr>(origOp->getAttr(VPU::INPUT_PADDING_ATTR_NAME))
                                    : nullptr;
    auto outputPaddingAttr = origOp->hasAttr(VPU::OUTPUT_PADDING_ATTR_NAME)
                                     ? mlir::cast<mlir::ArrayAttr>(origOp->getAttr(VPU::OUTPUT_PADDING_ATTR_NAME))
                                     : nullptr;

    auto nceOp = rewriter.create<VPU::NCEEltwiseOp>(origOp->getLoc(), outputType, input, input,
                                                    VPU::EltwiseTypeAttr::get(ctx, opType), ppeAttr,
                                                    /*multi_cluster_strategyAttr=*/nullptr,
                                                    /*is_inplace*/ nullptr, outputPaddingAttr, inputPaddingAttr);

    auto newOutput = nceOp.getOutput();

    if (needAlignment) {
        if (isAlignmentResolvedByReshape) {
            auto reshapeOp = rewriter.replaceOpWithNewOp<VPU::AffineReshapeOp>(
                    origOp, maybeRequantizedOutputType, nceOp.getOutput(), getIdentityDimPermutation(ctx),
                    getIntArrayAttr(ctx, maybeRequantizedOutputType.getShape()));
            newOutput = reshapeOp.getOutput();
        } else {
            SmallVector<int64_t> offsets(alignedShape.size(), 0);
            SmallVector<int64_t> sizes(originalShape.begin(), originalShape.end());

            auto sliceOp = rewriter.replaceOpWithNewOp<VPU::SliceOp>(origOp, maybeRequantizedOutputType,
                                                                     nceOp.getOutput(), getIntArrayAttr(ctx, offsets),
                                                                     getIntArrayAttr(ctx, sizes));
            newOutput = sliceOp.getResult();
        }
    }

    if (quantizedPerAxis) {
        auto quantCastOp =
                rewriter.create<VPU::QuantizeCastOp>(origOp->getLoc(), newOutput, originalOutputType.getElementType());
        newOutput = quantCastOp.getOutput();
    }

    rewriter.replaceOp(origOp, {newOutput});

    return mlir::success();
}

mlir::Value createFilter(mlir::PatternRewriter& rewriter, mlir::Location loc, vpux::NDTypeInterface filterType) {
    auto ctx = filterType.getContext();
    auto shape = filterType.getShape();
    auto elemType = filterType.getElementType();
    auto order = filterType.getDimsOrder();

    const auto OC = shape[Dims4D::Filter::OC];
    const auto IC = shape[Dims4D::Filter::IC];
    SmallVector<float> content(OC * IC, 0.0f);
    for (int64_t oc = 0; oc < OC; ++oc) {
        content[oc * IC + oc] = 1.0f;
    }
    auto dataStorageType = mlir::RankedTensorType::get(shape.raw(), mlir::Float32Type::get(ctx));
    auto dataAttr = Const::createConstContent(dataStorageType, ArrayRef(content));

    Const::ContentSetup contentAttrSetup(dataStorageType);
    if (auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
        contentAttrSetup = contentAttrSetup.castElemType(qElemType);
    } else if (mlir::isa<mlir::Float16Type>(elemType)) {
        contentAttrSetup = contentAttrSetup.castElemType(mlir::Float16Type::get(ctx));
    }
    if (order != DimsOrder::fromNumDims(shape.size())) {
        contentAttrSetup = contentAttrSetup.reorder(order);
    }

    auto filterContentAttr =
            Const::ContentAttr::get(dataAttr, contentAttrSetup.clone().sparsify(/*compressOutputType=*/false));
    auto filterConstOp = rewriter.create<Const::DeclareOp>(loc, filterType, std::move(filterContentAttr));

    // The sparsity map is introduced to avoid the compute with the numerous zero values in the filter
    auto sparsityMapContentAttr = Const::ContentAttr::get(dataAttr, contentAttrSetup.clone().getSparsityMap());
    auto sparsityMapConstOp =
            rewriter.create<Const::DeclareOp>(loc, sparsityMapContentAttr.getType(), std::move(sparsityMapContentAttr));

    auto contentAttr = Const::ContentAttr::get(dataAttr, std::move(contentAttrSetup));
    const auto numNonSparseElements = vpux::countNonSparseElementsPerOC(contentAttr.fold(), elemType);
    const auto numNonSparseElementsType =
            mlir::RankedTensorType::get({static_cast<int64_t>(numNonSparseElements.size())}, getInt64Type(ctx));
    const auto numElemsAttr = mlir::DenseElementsAttr::get(numNonSparseElementsType, ArrayRef(numNonSparseElements));
    const auto axisAttr = getIntAttr(ctx, Dims4D::Filter::OC.ind());
    const auto alignmentAttr = getIntAttr(ctx, VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT);
    auto sparsityCompressionAttr = VPU::SparsityCompressionAttr::get(ctx, axisAttr, numElemsAttr, alignmentAttr);

    auto groupOp =
            rewriter.create<VPU::GroupSparseTensorOp>(loc, filterConstOp.getOutput(), sparsityMapConstOp.getOutput(),
                                                      /*is_weights=*/true, sparsityCompressionAttr);

    return groupOp.getOutput();
}

// Hardware Convolution is chosen since it supports output sparsity while having the most flexibility
// in terms of variants configurations. Depthwise operations have limited support channel sizes for variants,
// while Eltwise operations cannot have variants tiled over Z which is often required for sparse producers.
mlir::LogicalResult rewriteSparsityOpWithConv(mlir::PatternRewriter& rewriter, mlir::Operation* origOp,
                                              vpux::Logger log, StringRef debugName) {
    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(log, rewriter, origOp, "[{0}] {1}", debugName, msg.str());
    };
    auto ctx = origOp->getContext();

    auto input = origOp->getOperand(0);
    auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto inputRank = inputType.getShape().size();
    if (inputRank != 4) {
        return matchFailed(log, rewriter, origOp, "Expected 4D input, but got '{0}' dimensions", inputRank);
    }

    auto output = origOp->getResult(0);
    auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
    const auto originalOutputType = outputType;

    auto defaultQuantElemType = mlir::quant::UniformQuantizedType::get(
            /*flags=*/0, getUInt8Type(ctx), mlir::Float16Type::get(ctx), /*scale=*/1.0,
            /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);

    auto quantizedPerAxis = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputType.getElementType());
    if (quantizedPerAxis) {
        auto quantCastOp = rewriter.create<VPU::QuantizeCastOp>(origOp->getLoc(), input, defaultQuantElemType);
        input = quantCastOp.getOutput();
        inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
        outputType = outputType.changeElemType(defaultQuantElemType);
    }

    const auto maybeRequantizedOutputType = outputType;

    const auto arch = config::getArch(origOp);
    auto alignment = vpux::VPU::NCEInvariant::getAlignment(inputType.getElementType());
    bool needAlignment = !vpux::VPU::NCEInvariant::isAligned(inputType, alignment, logCb);
    if (needAlignment) {
        Shape alignedShape;
        std::tie(input, alignedShape) = insertExpandToAlign(rewriter, input, alignment);
        inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
        outputType = outputType.changeShape(alignedShape);
    }

    const auto OC = outputType.getShape()[Dims4D::Act::C];
    const auto IC = inputType.getShape()[Dims4D::Act::C];
    const Shape filterShape({OC, IC, 1, 1});

    auto inputElemType = inputType.getElementType();
    mlir::Type filterElemType = mlir::Float16Type::get(ctx);
    if (auto qInputElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputElemType)) {
        filterElemType = mlir::quant::UniformQuantizedType::get(
                /*flags=*/0, getUInt8Type(ctx), mlir::Float16Type::get(ctx),
                /*scale=*/1.0, /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);
    }
    auto filterTensorAttr = vpux::getTensorAttr(ctx, DimsOrder::OYXI, nullptr);
    auto filterType = mlir::cast<vpux::NDTypeInterface>(
            mlir::RankedTensorType::get(filterShape.raw(), filterElemType, filterTensorAttr));

    const SmallVector<int64_t> dilations{1, 1};
    const auto pads = vpux::PadInfo(0, 0, 0, 0);

    if (!VPU::isNCEConvSupported(origOp, inputType, filterType, outputType, dilations, /*KY=*/1, /*KX=*/1, /*SY=*/1,
                                 /*SX=*/1, pads, /*checkLayout=*/true, /*checkChannelAlignment=*/true, logCb)) {
        return matchFailed(log, rewriter, origOp,
                           "Cannot lower operation to NCE Convolution because of HW requirements");
    }

    auto filter = createFilter(rewriter, origOp->getLoc(), filterType);

    // TODO: add weights sparsity map to save compute
    const auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch);
    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp);

    const auto adaptedOutElemType =
            VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    ppeAttr, outputType.getElementType());

    auto weightsTableVec = VPU::createWeightsTableData(origOp->getOperand(0), adaptedOutElemType, filter,
                                                       /*bias=*/{}, OC, ppeConverter, biasConverter,
                                                       /*constScale=*/nullptr, /*hasAutopad=*/false);
    const auto wtShape = VPU::NCESparsity::inferWeightsTableShape(OC);
    auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec, wtShape);

    auto stridesAttr = getIntArrayAttr(ctx, SmallVector<int64_t>({1, 1}));
    auto padAttr = VPU::getPaddingAttr(ctx, pads);
    auto rawFilterShape = getIntArrayAttr(ctx, filterType.getShape().raw());

    auto inputPaddingAttr = origOp->hasAttr(VPU::INPUT_PADDING_ATTR_NAME)
                                    ? mlir::cast<mlir::ArrayAttr>(origOp->getAttr(VPU::INPUT_PADDING_ATTR_NAME))
                                    : nullptr;
    auto outputPaddingAttr = origOp->hasAttr(VPU::OUTPUT_PADDING_ATTR_NAME)
                                     ? mlir::cast<mlir::ArrayAttr>(origOp->getAttr(VPU::OUTPUT_PADDING_ATTR_NAME))
                                     : nullptr;

    auto nceOp = rewriter.create<VPU::NCEConvolutionOp>(
            origOp->getLoc(), outputType, input, filter, weightsTable,
            /*weight_table_data_ptr=*/nullptr,
            /*weight_table_sp_ptr=*/nullptr, /*weight_table_scale=*/nullptr, /*weight_table_bias=*/nullptr,
            /*weight_zero_points=*/nullptr, stridesAttr, padAttr, ppeAttr, mpeEngineAttr, rawFilterShape,
            /*multi_cluster_strategyAttr=*/nullptr, outputPaddingAttr, inputPaddingAttr);

    auto newOutput = nceOp.getOutput();

    if (needAlignment) {
        auto origOutputShape = mlir::cast<vpux::NDTypeInterface>(maybeRequantizedOutputType).getShape();

        SmallVector<int64_t> offsets(origOutputShape.size(), 0);
        SmallVector<int64_t> sizes(origOutputShape.begin(), origOutputShape.end());

        auto sliceOp = rewriter.create<VPU::SliceOp>(origOp->getLoc(), maybeRequantizedOutputType, newOutput,
                                                     getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes));
        newOutput = sliceOp.getResult();
    }

    if (quantizedPerAxis) {
        auto quantCastOp =
                rewriter.create<VPU::QuantizeCastOp>(origOp->getLoc(), newOutput, originalOutputType.getElementType());
        newOutput = quantCastOp.getOutput();
    }

    rewriter.replaceOp(origOp, {newOutput});

    return mlir::success();
}

//
// LowerSparsityOpsPass
//

class LowerSparsityOpsPass final : public VPU::impl::LowerSparsityOpsBase<LowerSparsityOpsPass> {
public:
    explicit LowerSparsityOpsPass(std::optional<bool> maybeFakeSparsify, Logger log)
            : _maybeFakeSparsify(maybeFakeSparsify) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    std::optional<bool> _maybeFakeSparsify;
};

mlir::LogicalResult LowerSparsityOpsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (!fakeSparsify.hasValue()) {
        return mlir::success();
    }
    if (_maybeFakeSparsify.has_value()) {
        _log.trace("Overloading C++ createLowerSparsityOpsPass argument by MLIR variable");
    }
    _maybeFakeSparsify = fakeSparsify;
    return mlir::success();
}

//
// RewriteDesparsify
//

class RewriteDesparsify final : public mlir::OpRewritePattern<VPU::DesparsifyOp> {
public:
    RewriteDesparsify(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::DesparsifyOp>(ctx), _log(log) {
        setDebugName("RewriteDesparsify");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::DesparsifyOp desparsifyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RewriteDesparsify::matchAndRewrite(VPU::DesparsifyOp desparsifyOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", desparsifyOp->getName(), desparsifyOp->getLoc());
    return rewriteSparsityOpWithEltwiseOp(rewriter, desparsifyOp.getOperation(), _log, getDebugName());
}

//
// RewriteSparsify
//

class RewriteSparsify final : public mlir::OpRewritePattern<VPU::SparsifyOp> {
public:
    RewriteSparsify(mlir::MLIRContext* ctx, bool useFakeSparsify, Logger log)
            : mlir::OpRewritePattern<VPU::SparsifyOp>(ctx), _useFakeSparsify(useFakeSparsify), _log(log) {
        setDebugName("RewriteSparsify");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::SparsifyOp SparsifyOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _useFakeSparsify;
    Logger _log;
};

mlir::LogicalResult RewriteSparsify::matchAndRewrite(VPU::SparsifyOp sparsifyOp,
                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", sparsifyOp->getName(), sparsifyOp->getLoc());

    if (!_useFakeSparsify) {
        return rewriteSparsityOpWithConv(rewriter, sparsifyOp.getOperation(), _log, getDebugName());
    }

    const auto sparsityMap = createActSparsityMap(rewriter, sparsifyOp.getInput().getType());
    auto groupedView =
            rewriter.create<VPU::GroupSparseTensorOp>(sparsifyOp.getLoc(), sparsifyOp.getInput(), sparsityMap);
    // GroupSparseTensorOp result have new type, so cant just replaceOpWithNewOp
    sparsifyOp.getOutput().replaceAllUsesWith(groupedView->getResult(0));
    rewriter.eraseOp(sparsifyOp);

    return mlir::success();
}

//
// safeRunOnFunc
//

void LowerSparsityOpsPass::safeRunOnFunc() {
    using namespace VPU;
    using namespace VPU::NCESparsity;

    auto func = getOperation();
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPU::DesparsifyOp>();
    target.addIllegalOp<VPU::SparsifyOp>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RewriteDesparsify>(&ctx, _log);
    patterns.add<RewriteSparsify>(&ctx, _maybeFakeSparsify.value_or(true), _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLowerSparsityOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createLowerSparsityOpsPass(std::optional<bool> maybeFakeSparsify, Logger log) {
    return std::make_unique<LowerSparsityOpsPass>(maybeFakeSparsify, log);
}
