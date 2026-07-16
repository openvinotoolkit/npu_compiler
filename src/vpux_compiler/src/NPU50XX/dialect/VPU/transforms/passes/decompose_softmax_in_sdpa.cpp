//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/sprlut_generator.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <memory>

namespace vpux::VPU::arch50xx {
#define GEN_PASS_DECL_DECOMPOSESOFTMAXINSDPA
#define GEN_PASS_DEF_DECOMPOSESOFTMAXINSDPA
#include "vpux/compiler/NPU50XX/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU::arch50xx

using namespace vpux;

namespace {

class DecomposeSoftmax final : public mlir::OpRewritePattern<VPU::SoftMaxOp> {
public:
    DecomposeSoftmax(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("DecomposeSoftmax");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool isBeneficialAndFeasible(VPU::SoftMaxOp origOp) const;
    double getRescaler(VPU::SoftMaxOp origOp) const;
    mlir::Value constructUpStream(VPU::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const;
    vpux::VPU::NCEOpInterface constructLeftBranch(VPU::SoftMaxOp origOp, VPU::NCEOpInterface userOp, double rescaler,
                                                  const vpux::NDTypeInterface newUserOutputType) const;
    VPU::NCEConvolutionOp constructRightBranch(VPU::SoftMaxOp origOp, mlir::PatternRewriter& rewriter,
                                               mlir::Value expOutput, mlir::Operation* userOp,
                                               const vpux::NDTypeInterface userOutputType,
                                               const vpux::NDTypeInterface newUserOutputType) const;
    vpux::VPU::NCEEltwiseOp constructDownStream(VPU::SoftMaxOp origOp, mlir::PatternRewriter& rewriter,
                                                mlir::Value leftBrnach, mlir::Value rightBrnach, double descaler,
                                                const vpux::NDTypeInterface userOutputType,
                                                vpux::VPU::PPEAttr userOpPpeAttr) const;

private:
    Logger _log;
    mutable mlir::DenseElementsAttr _sprLUTAttr;
    mutable bool _sprLUTAttrInitialized = false;
};

//     Activation  Weights
//           \   /
//           MatMul0
//             |
//          SoftMax
//             |
//          MatMul1
//  To
//
//     Activation  Weights
//           \   /
//           MatMul0
//             |
//       Exp_stabilized --- [ReduceMax -> Subtract -> Convert -> Exp -> Convert]
//         |        |
//      MatMul1    ReduceSum
//     (rescaler)       |
//         |         Expand
//         |            |
//         |       Convolution (broadcast)
//          \           /
//         Eltwise(Multiply)
//           (1/rescaler)

bool DecomposeSoftmax::isBeneficialAndFeasible(VPU::SoftMaxOp origOp) const {
    const auto ctx = this->getContext();
    const auto output = origOp.getOutput();
    if (!output.hasOneUse()) {
        _log.info("decompose failed '{0}' at '{1}', more than one use", origOp->getName(), origOp->getLoc());
        return false;
    }

    const auto axis = origOp.getAxisIndAttr().getInt();
    if (axis != 1) {
        _log.info("decompose failed '{0}' at '{1}', axis of softmax != 1", origOp->getName(), origOp->getLoc());
        return false;
    }

    const auto firstUserOp = *output.getUsers().begin();
    if (!mlir::isa<VPU::NCEConvolutionOp>(firstUserOp)) {
        _log.info("decompose failed '{0}' at '{1}', not NCEOpInterface", origOp->getName(), origOp->getLoc());
        return false;
    }

    auto userOp = mlir::cast<VPU::NCEOpInterface>(firstUserOp);
    auto userOpOutput = userOp.getOperation()->getResult(0);
    const auto userOutputType = mlir::cast<vpux::NDTypeInterface>(userOpOutput.getType());

    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(userOutputType.getElementType())) {
        _log.info("decompose failed '{0}' at '{1}', per-channel output quantization", origOp->getName(),
                  origOp->getLoc());
        return false;
    }
    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto scaleAdapter = ppeConfig.getFactoryAs<VPU::IPpeAdapterScaleBias*>();
    const auto modeAdapter = ppeConfig.getFactoryAs<VPU::IPpeAdapterMode*>();

    auto ppeAttr = userOp.getPPE();
    const auto oldBias = scaleAdapter->getBias(ppeAttr);
    const auto mode = modeAdapter->getMode(ppeAttr);

    if (mode != vpux::VPU::PPEMode::NOOP) {
        _log.info("decompose failed '{0}' at '{1}', non NOOP mode in ppe", origOp->getName(), origOp->getLoc());
        return false;
    }

    // neutral bias allows delaying multiplication to after convolution
    if (oldBias.value_or(1.0f) != 0.0f) {
        _log.info("decompose failed '{0}' at '{1}', non neutral per-tensor bias in ppe", origOp->getName(),
                  origOp->getLoc());
        return false;
    }

    const auto userOpInputShape = getShape(userOp->getOperand(0));
    const auto userOpInputSize =
            std::accumulate(userOpInputShape.begin(), userOpInputShape.end(), 1, std::multiplies<int64_t>());
    const auto userOpOutputShape = getShape(userOp->getResult(0));
    const auto userOpOutputSize =
            std::accumulate(userOpOutputShape.begin(), userOpOutputShape.end(), 1, std::multiplies<int64_t>());
    const auto outputInputSizeMultiplier = 64;
    if (userOpInputSize <= userOpOutputSize * outputInputSizeMultiplier) {
        // this criterion is for identifying SDPA patterns that runtime of SoftMaxOp significantly exceeds both
        // MatMulOps
        _log.info("decompose failed '{0}' at '{1}', softmax isn't the bottleneck", origOp->getName(), origOp->getLoc());
        return false;
    }
    return true;
}

double DecomposeSoftmax::getRescaler(VPU::SoftMaxOp origOp) const {
    const auto output = origOp.getOutput();
    const auto axisLength = getShape(output)[Dim(origOp.getAxisIndAttr().getValue().getSExtValue())];
    double rescaler = 1 / sqrt(axisLength);
    const auto firstUserOp = *output.getUsers().begin();
    const auto userOpOutput = firstUserOp->getResult(0);
    const auto userOutputType = mlir::cast<vpux::NDTypeInterface>(userOpOutput.getType());
    const auto userOutputElemType = userOutputType.getElementType();
    // re-emplace the quant of output and output.scale from userOp to the last eltwiseOp
    if (auto quantUserOutType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(userOutputElemType)) {
        const auto scale = quantUserOutType.getScale();
        const auto bias = quantUserOutType.getZeroPoint();
        VPUX_THROW_UNLESS(bias == 0.0f, "Quantized type must have bias of 0.0f");
        rescaler = rescaler * scale;
    }
    return rescaler;
}

mlir::Value DecomposeSoftmax::constructUpStream(VPU::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    // Replace the ReduceMax -> Subtract -> Convert -> Exp -> Convert chain with a
    // single exp_stabilized SHAVE kernel dispatched via SoftMaxOp(SkipNormalization).
    // The kernel computes max per-row internally and outputs exp(input - max), which
    // avoids the intermediate DPU ReduceMax and the two precision-conversion ops.
    const auto input = origOp.getInput();
    const auto outputType = origOp.getOutput().getType();

    auto expStabilizedOp = rewriter.create<VPU::SoftMaxOp>(
            appendLoc(origOp->getLoc(), "_exp_stabilized"), outputType, input, /*max=*/mlir::Value{},
            origOp.getAxisIndAttr(), origOp.getPadSizeAttr(), origOp.getDstElemTypeAttr(), origOp.getMaskAwareAttr(),
            origOp.getMultiClusterStrategyAttr());
    expStabilizedOp->setAttr("SkipNormalization", rewriter.getUnitAttr());

    return expStabilizedOp.getOutput();
}

vpux::VPU::NCEOpInterface DecomposeSoftmax::constructLeftBranch(VPU::SoftMaxOp origOp, VPU::NCEOpInterface userOp,
                                                                double rescaler,
                                                                const vpux::NDTypeInterface newUserOutputType) const {
    auto ppeAttr = userOp.getPPE();
    const auto ctx = this->getContext();
    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto origPpeAttr = ppeConfig.retrievePPEAttribute(origOp);
    const auto scaleAdapter = ppeConfig.getFactoryAs<VPU::IPpeAdapterScaleBias*>();
    const auto clampAdapter = ppeConfig.getFactoryAs<VPU::IPpeAdapterClamp*>();
    ppeAttr = clampAdapter->updateClamps(ppeAttr, origPpeAttr);
    const auto oldScale = scaleAdapter->getScale(ppeAttr);
    VPUX_THROW_UNLESS(oldScale.has_value(), "Scale must have a value");

    // The following logic related scale is to prevent the result of second Matmul from overflowing
    const auto newScale = rescaler * oldScale.value()[0];
    ppeAttr = scaleAdapter->updateScale(ppeAttr, {newScale});
    userOp.setPPE(ppeAttr);

    auto userOpOutput = userOp.getOperation()->getResult(0);
    userOpOutput.setType(newUserOutputType);
    return userOp;
}

VPU::NCEConvolutionOp DecomposeSoftmax::constructRightBranch(VPU::SoftMaxOp origOp, mlir::PatternRewriter& rewriter,
                                                             mlir::Value expOutput, mlir::Operation* userOp,
                                                             const vpux::NDTypeInterface userOutputType,
                                                             const vpux::NDTypeInterface newUserOutputType) const {
    const auto ctx = this->getContext();
    const auto f16Type = mlir::Float16Type::get(ctx);
    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto origPpeAttr = ppeConfig.retrievePPEAttribute(origOp);
    const auto paddingLength = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT - 1;
    auto reduceSumInType = mlir::cast<vpux::NDTypeInterface>(expOutput.getType());
    const auto reduceSumInTypeShape = getShape(expOutput);
    llvm::SmallVector<int64_t> reduceSumOutTypeShape = {
            reduceSumInTypeShape[Dims4D::Act::N],
            1,
            reduceSumInTypeShape[Dims4D::Act::H],
            reduceSumInTypeShape[Dims4D::Act::W],
    };
    const auto reduceSumOutType = reduceSumInType.changeShape(ShapeRef(reduceSumOutTypeShape));

    auto origPpe = mlir::cast<VPU::PPEFpAttr>(origPpeAttr);
    if (!_sprLUTAttrInitialized) {
        _sprLUTAttr = [&]() -> mlir::DenseElementsAttr {
            using namespace VPU;
            const auto sprLUT = SprLUTGenerator(
                                        [](float x) {
                                            return 1.0f / x;
                                        },
                                        RelativeError(RCP_SFM_ERROR))
                                        .setIsSymmetric()
                                        .addSaturationRange(RCP_SFM_POS_SMALL_SAT_LOW, RCP_SFM_POS_SMALL_SAT_HIGH,
                                                            RCP_SFM_POS_SMALL_SAT_VALUE)
                                        .addSaturationRange(RCP_SFM_POS_LARGE_SAT_LOW, RCP_SFM_POS_LARGE_SAT_HIGH,
                                                            RCP_SFM_POS_LARGE_SAT_VALUE)
                                        .generate();
            const auto uint16Type = mlir::IntegerType::get(ctx, 16, mlir::IntegerType::SignednessSemantics::Unsigned);
            auto sprLUTType = mlir::RankedTensorType::get({checked_cast<int64_t>(sprLUT.size())}, uint16Type);
            return mlir::DenseElementsAttr::get(sprLUTType, ArrayRef(sprLUT));
        }();
        _sprLUTAttrInitialized = true;
    }

    auto reducePpeAttr =
            VPU::PPEFpAttr::get(ctx, VPU::PPEModeAttr::get(ctx, VPU::PPEMode::RCP_SFM), origPpe.getClampLow(),
                                origPpe.getClampHigh(), origPpe.getScale(), origPpe.getPreluAlpha(), origPpe.getBias(),
                                origPpe.getAdder(), origPpe.getIn1Mult(), origPpe.getIn2Mult(), _sprLUTAttr);

    auto reduceSumOp = rewriter.create<VPU::NCEReduceOp>(
            appendLoc(origOp->getLoc(), "_reduce_sum"), reduceSumOutType, expOutput,
            vpux::getIntArrayAttr(ctx, SmallVector<int64_t>({origOp.getAxisIndAttr().getValue().getSExtValue()})),
            reducePpeAttr, nullptr, VPU::ReduceTypeAttr::get(ctx, VPU::ReduceType::SUM), nullptr, nullptr,
            mlir::dyn_cast_or_null<mlir::ArrayAttr>(userOp->getAttr("input_padding")));

    rewriter.setInsertionPointAfter(userOp);

    auto expandOp = rewriter.create<VPU::ExpandOp>(appendLoc(origOp->getLoc(), "_expandOp"), reduceSumOp->getResult(0),
                                                   getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0, 0, 0}),
                                                   getIntArrayAttr(ctx, SmallVector<int64_t>{0, paddingLength, 0, 0}));

    // convolutionOp with dummy weights for broadcasting inverseOp's output
    const auto dummyWeightsShape = Shape({1, 1, 1, 1});
    const auto dummyWeightsType = mlir::RankedTensorType::get(dummyWeightsShape.raw(), f16Type);
    const auto dummyWeightsAttr = Const::createConstContent(dummyWeightsType, ArrayRef({1.0f}));
    Shape cstPadBegin = {0, 0, 0, 0};
    Shape cstPadEnd = {0, paddingLength, 0, 0};
    const auto userOC = userOutputType.getShape()[Dims4D::Act::C];
    const auto dummyWeightsContentAttr = Const::ContentAttr::get(dummyWeightsAttr)
                                                 .transform()
                                                 .broadcast(Dim(0), userOC)
                                                 .reshape({userOC, 1, 1, 1})
                                                 .padWithZero(cstPadBegin, cstPadEnd)
                                                 .reorder(DimsOrder::NHWC)
                                                 .get();
    auto dummyWeights =
            rewriter.create<Const::DeclareOp>(appendLoc(origOp->getLoc(), "dummy_weights"),
                                              dummyWeightsContentAttr.getType(), std::move(dummyWeightsContentAttr))
                    .getOutput();

    const auto arch = config::getArch(origOp);
    const auto useNewWtFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(userOp, /*isCompressConv=*/false);
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch, useNewWtFormat);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch, useNewWtFormat);

    const auto adaptedOutElemType =
            ppeConfig.getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    origPpeAttr, newUserOutputType.getElementType());
    const auto weightsTableParams = VPU::WeightsTableParams(
            /*op=*/nullptr, reduceSumOp->getResult(0), adaptedOutElemType, dummyWeights, {}, userOC, ppeConverter,
            biasConverter, /*constScale=*/nullptr, /*zeroPoints=*/nullptr);

    mlir::Value legacyWeightsTable = nullptr;
    mlir::Value wtScale = nullptr;
    mlir::Value wtBias = nullptr;
    if (useNewWtFormat) {
        const auto newWtShape = VPU::NCESparsity::inferWeightsTableShape(userOC, /*newFormat=*/true);
        const auto newWtTensors = VPU::NewWeightsTableTensors(/*useNewWeightTableFormat=*/true, weightsTableParams,
                                                              rewriter, origOp->getLoc(), newWtShape);
        wtScale = newWtTensors.scaleTensor;
        wtBias = newWtTensors.biasTensor;
    } else {
        const auto weightsTableVec = VPU::createWeightsTableData(weightsTableParams, VPU::canAutopadOutput(origOp));
        const auto wtShape = VPU::NCESparsity::inferWeightsTableShape(userOC);
        legacyWeightsTable = VPU::createTensorFromTableData<int32_t>(rewriter, origOp->getLoc(), weightsTableVec,
                                                                     wtShape, getSInt32Type(rewriter.getContext()));
    }

    const auto stridesAttr = getIntArrayAttr(origOp->getContext(), SmallVector<int64_t>{1, 1});
    const auto padAttr = VPU::getPaddingAttr(origOp->getContext(), PadInfo(0, 0, 0, 0));
    const auto rawFilterShape = getIntArrayAttr(rewriter, getShape(dummyWeights));
    const auto inputPaddingAttr = mlir::cast<mlir::ArrayAttr>(rewriter.getI64ArrayAttr({0, paddingLength, 0, 0}));
    const auto outputPaddingAttr = origOp->hasAttr(VPU::OUTPUT_PADDING_ATTR_NAME)
                                           ? mlir::cast<mlir::ArrayAttr>(origOp->getAttr(VPU::OUTPUT_PADDING_ATTR_NAME))
                                           : nullptr;
    const auto broadcastedInverseOutputType = newUserOutputType.changeElemType(f16Type);

    return rewriter.create<VPU::NCEConvolutionOp>(
            appendLoc(origOp->getLoc(), "_conv_tileOp"), broadcastedInverseOutputType,
            /*reduceXyMax*/ nullptr, /*reduceXyMin*/ nullptr,
            /*reduceGlobalMinMax*/ nullptr, expandOp.getOutput(), dummyWeights, legacyWeightsTable,
            /*weight_table_data_ptr=*/nullptr,
            /*weight_table_sp_ptr=*/nullptr, /*weight_table_scale=*/wtScale,
            /*weight_table_bias=*/wtBias,
            /*weight_zero_points=*/nullptr, stridesAttr, padAttr, origPpeAttr,
            /*mpeEngineAttr=*/nullptr, /*rawFilterShape=*/mlir::ValueRange{},
            parseIntArrayAttr<int64_t>(rawFilterShape),
            /*multi_cluster_strategyAttr=*/nullptr, outputPaddingAttr, inputPaddingAttr,
            /*axes_value=*/nullptr);
}

vpux::VPU::NCEEltwiseOp DecomposeSoftmax::constructDownStream(VPU::SoftMaxOp origOp, mlir::PatternRewriter& rewriter,
                                                              mlir::Value leftBrnach, mlir::Value rightBrnach,
                                                              double descaler,
                                                              const vpux::NDTypeInterface userOutputType,
                                                              vpux::VPU::PPEAttr userOpPpeAttr) const {
    const auto ctx = this->getContext();
    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto origPpeAttr = ppeConfig.retrievePPEAttribute(origOp);
    const auto scaleAdapter = ppeConfig.getFactoryAs<VPU::IPpeAdapterScaleBias*>();
    const auto clampAdapter = ppeConfig.getFactoryAs<VPU::IPpeAdapterClamp*>();
    auto newMultiplyOpPpeAttr = scaleAdapter->updateScale(origPpeAttr, descaler);
    newMultiplyOpPpeAttr = clampAdapter->updateClamps(newMultiplyOpPpeAttr, userOpPpeAttr);

    return rewriter.create<VPU::NCEEltwiseOp>(
            appendLoc(origOp->getLoc(), "_eltwiseOp"), userOutputType, leftBrnach, rightBrnach,
            /*weight_table_scale=*/nullptr, /*weight_table_bias=*/nullptr,
            VPU::EltwiseTypeAttr::get(ctx, VPU::EltwiseType::MULTIPLY), newMultiplyOpPpeAttr, nullptr,
            /*multi_cluster_strategyAttr=*/nullptr,
            /*is_inplace=*/
            rewriter.getBoolAttr(userOutputType.getDimsOrder() == DimsOrder::NHWC &&
                                 userOutputType.getElementType().isF16()),
            nullptr, nullptr);
}

mlir::LogicalResult DecomposeSoftmax::matchAndRewrite(VPU::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!isBeneficialAndFeasible(origOp)) {
        return mlir::failure();
    }
    _log.info("target identified '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto ctx = this->getContext();
    const auto output = origOp.getOutput();
    const auto firstUserOp = *output.getUsers().begin();
    auto userOp = mlir::cast<VPU::NCEOpInterface>(firstUserOp);
    auto userOpOutput = firstUserOp->getResult(0);
    const auto userOutputType = mlir::cast<vpux::NDTypeInterface>(userOpOutput.getType());
    const auto f16Type = mlir::Float16Type::get(ctx);
    const auto newUserOutputType = userOutputType.changeDimsOrder(DimsOrder::NHWC).changeElemType(f16Type);
    auto userOpPpeAttr = userOp.getPPE();
    rewriter.setInsertionPointAfter(origOp);

    double rescaler = getRescaler(origOp);
    const double descaler = 1 / rescaler;

    auto expOutput = constructUpStream(origOp, rewriter);
    origOp.replaceAllUsesWith(expOutput);

    auto newUserOp = constructLeftBranch(origOp, userOp, rescaler, newUserOutputType);

    auto broadcastedInverseOp =
            constructRightBranch(origOp, rewriter, expOutput, newUserOp, userOutputType, newUserOutputType);

    auto newMultiplyOp = constructDownStream(origOp, rewriter, newUserOp.getOperation()->getResult(0),
                                             broadcastedInverseOp.getOutput(), descaler, userOutputType, userOpPpeAttr);

    newUserOp.getOperation()->getResult(0).replaceAllUsesExcept(newMultiplyOp.getOutput(),
                                                                llvm::SmallPtrSet<mlir::Operation*, 1>{newMultiplyOp});
    _log.info("DecomposeSoftmax succeeded at '{0}'", origOp->getLoc());
    rewriter.eraseOp(origOp);
    return mlir::success();
}

class DecomposeSoftmaxInSdpaPass final :
        public VPU::arch50xx::impl::DecomposeSoftmaxInSdpaBase<DecomposeSoftmaxInSdpaPass> {
public:
    explicit DecomposeSoftmaxInSdpaPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final {
        auto& ctx = getContext();
        auto func = getOperation();

        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<DecomposeSoftmax>(&ctx, _log.nest());
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::arch50xx::createDecomposeSoftmaxInSdpaPass(const Logger& log) {
    return std::make_unique<DecomposeSoftmaxInSdpaPass>(log);
}
