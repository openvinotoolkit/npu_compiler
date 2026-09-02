//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux {
//
// ConvToNCE
//

class ConvToNCE final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("ConvToNCE");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    config::ArchKind _arch;
    Logger _log;
};

//
// MatMulToNCE
//

class MatMulToNCE final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    MatMulToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx), _arch(arch), _log(log) {
        setDebugName("MatMulToNCE");
    }

    mlir::LogicalResult matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    config::ArchKind _arch;
    Logger _log;
};

//
// DepthConvToNCE
//

class DepthConvToNCE final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    DepthConvToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("DepthConvToNCE");
    }

    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    config::ArchKind _arch;
    Logger _log;
};

//
// MaxPoolToNCE
//

class MaxPoolToNCE final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _arch(arch), _log(log) {
        setDebugName("MaxPoolToNCE");
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    config::ArchKind _arch;
    Logger _log;
};

//
// AveragePoolToNCE
//

class AveragePoolToNCE final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AveragePoolToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _arch(arch), _log(log) {
        setDebugName("AveragePoolToNCE");
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    config::ArchKind _arch;
    Logger _log;
};

//
// PermuteQuantizeToNCEPermute
//

class PermuteQuantizeToNCEPermute final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    PermuteQuantizeToNCEPermute(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteQuantizeOp>(ctx), _log(log) {
        setDebugName("PermuteQuantizeToNCEPermute");
    }

    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    Logger _log;
};

//
// EltwiseToNCE
//

template <class ConcreteOp>
class EltwiseToNCE final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    EltwiseToNCE(mlir::MLIRContext* ctx, VPU::EltwiseType opType, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _opType(opType), _arch(arch), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::EltwiseType _opType;
    config::ArchKind _arch;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult EltwiseToNCE<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto ppeAttr = VPU::getPpeConfig(origOp->getContext()).retrievePPEAttribute(origOp);

    const auto inputElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput1().getType()).getElementType();
    const auto outputElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();
    const bool isInputQuantizedPerAxis = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputElemType);
    const bool isOutputQuantizedPerAxis = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType);

    // Generate scale and bias tables for per-axis quantization.
    // For per-tensor or non-quantized cases, PPE handles quantization directly without requiring scale tables.
    const auto getScaleAndBias = [&]() -> std::pair<mlir::Value, mlir::Value> {
        if (!isInputQuantizedPerAxis && !isOutputQuantizedPerAxis && origOp.getScale() == nullptr) {
            return {nullptr, nullptr};
        }
        const auto output = origOp.getOutput();
        const auto outputShape = getShape(output);
        const auto OC = outputShape[Dims4D::Act::C];

        Const::ContentAttr bias = nullptr;

        const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, false);
        const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
        const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);

        const auto staticScale = origOp->template getAttrOfType<mlir::FloatAttr>("static_scale");

        const auto newWtShape = VPU::NCESparsity::inferWeightsTableShape(OC, /*newFormat=*/true);
        const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(
                isNewWeightTableFormat,
                VPU::WeightsTableParams(origOp, origOp.getInput1(), output, /*weights=*/nullptr, bias, OC, ppeConverter,
                                        biasConverter, staticScale, /*zeroPoints=*/nullptr),
                rewriter, origOp->getLoc(), newWtShape);

        auto scaleTensor = origOp.getScale() != nullptr ? origOp.getScale() : newWeightsTableTensors.scaleTensor;

        if (newWeightsTableTensors.scaleTensor != nullptr && origOp.getScale() != nullptr) {
            if (auto constOp = mlir::dyn_cast<Const::DeclareOp>(newWeightsTableTensors.scaleTensor.getDefiningOp())) {
                auto scaleTableContent = constOp.getContent();
                auto staticScaleEqualToOne =
                        scaleTableContent.isSplat() && isFloatEqual(scaleTableContent.getSplatValue<float>(), 1.0f);
                if (!staticScaleEqualToOne) {
                    scaleTensor = rewriter.create<IE::MultiplyOp>(
                            origOp.getLoc(), scaleTensor, newWeightsTableTensors.scaleTensor,
                            IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
                }
            }
        }

        return {scaleTensor, newWeightsTableTensors.biasTensor};
    };

    const auto [scaleTensor, biasTensor] = getScaleAndBias();

    VPU::MPEEngineAttr mpeEngineAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        const auto input1 = getPerTensorZeroPointAttr(origOp.getInput2());
        const auto input2 = getPerTensorZeroPointAttr(origOp.getInput1());

        mpeEngineAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineWithZP(input1, input2));
    }

    auto nceOp = rewriter.create<VPU::NCEEltwiseOp>(
            origOp->getLoc(), origOp.getType(), /*reduce_xy_max=*/nullptr,
            /*reduce_xy_min=*/nullptr, /*reduce_tensor_min_max=*/nullptr, origOp.getInput1(), origOp.getInput2(),
            scaleTensor, biasTensor, VPU::EltwiseTypeAttr::get(this->getContext(), _opType), ppeAttr, mpeEngineAttr,
            /*multi_cluster_strategyAttr=*/nullptr,
            /*is_inplace=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr(),
            /*axes_value=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}
}  // namespace vpux
