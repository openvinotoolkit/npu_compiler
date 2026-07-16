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
    EltwiseToNCE<ConcreteOp>(mlir::MLIRContext* ctx, VPU::EltwiseType opType, config::ArchKind arch, Logger log)
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

    // Scales-as-input for NCE eltwise is not yet implemented. The `scales` operand
    // is a reserved placeholder in the IR. Remove this check and implement the
    // scale tensor input path before enabling it.
    if (origOp.getScales() != nullptr) {
        VPUX_THROW("NCE eltwise op does not support scales-as-input; "
                   "implement scale tensor input support before enabling this path");
    }

    const auto ppeAttr = VPU::getPpeConfig(origOp->getContext()).retrievePPEAttribute(origOp);

    const auto inputElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput1().getType()).getElementType();
    const auto outputElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();
    const bool isInputQuantizedPerAxis = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputElemType);
    const bool isOutputQuantizedPerAxis = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType);

    // Generate scale and bias tables for per-axis quantization.
    // For per-tensor or non-quantized cases, PPE handles quantization directly without requiring scale tables.
    const auto getScaleAndBias = [&]() -> std::pair<mlir::Value, mlir::Value> {
        if (!isInputQuantizedPerAxis && !isOutputQuantizedPerAxis) {
            return {nullptr, nullptr};
        }

        const auto output = origOp.getOutput();
        const auto outputShape = getShape(output);
        const auto OC = outputShape[Dims4D::Act::C];

        Const::ContentAttr bias = nullptr;

        const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, false);
        const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
        const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);

        const auto newWtShape = VPU::NCESparsity::inferWeightsTableShape(OC, /*newFormat=*/true);
        const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(
                isNewWeightTableFormat,
                VPU::WeightsTableParams(origOp, origOp.getInput1(), output, /*weights=*/nullptr, bias, OC, ppeConverter,
                                        biasConverter, /*constScale=*/nullptr, /*zeroPoints=*/nullptr),
                rewriter, origOp->getLoc(), newWtShape);

        return {newWeightsTableTensors.scaleTensor, newWeightsTableTensors.biasTensor};
    };

    const auto [scaleTensor, biasTensor] = getScaleAndBias();

    VPU::MPEEngineAttr mpeEngineModeAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        mpeEngineModeAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineMode());
    }

    auto nceOp = rewriter.create<VPU::NCEEltwiseOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput1(), origOp.getInput2(), scaleTensor, biasTensor,
            VPU::EltwiseTypeAttr::get(this->getContext(), _opType), ppeAttr, mpeEngineModeAttr,
            /*multi_cluster_strategyAttr=*/nullptr,
            /*is_inplace=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}
}  // namespace vpux
