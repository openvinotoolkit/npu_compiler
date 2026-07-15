//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/IE/interfaces/channel_axis_reduction_with_dpu_parent_checker.hpp"
#include "vpux/compiler/dialect/IE/interfaces/convert_quantize_ops_to_nce_ops_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/convert_to_conv_alignment_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/convert_to_mixed_precision_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/dialect/IE/interfaces/expand_activation_channels_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/fuse_convert_to_dpu_checker.hpp"
#include "vpux/compiler/dialect/IE/interfaces/map_bilinear_interpolate_on_dpu_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/propagate_and_fuse_quantize_dequantize_strategy.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_strategies.hpp"

namespace vpux::IE {

class StrategyFactory {
public:
    virtual ~StrategyFactory() = default;

    virtual std::unique_ptr<IConvertQuantizeOpsToNceOpsStrategy> getConvertQuantizeOpsToNceOpsStrategy() = 0;
    virtual std::unique_ptr<IDynamicRewriterStrategy> getWeightsDequantizeToDynamicDequantizeStrategy(
            ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index) = 0;
    virtual std::unique_ptr<IDynamicRewriterStrategy> getWeightsDequantizeToFakeQuantizeStrategy(
            ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index) = 0;
    virtual std::unique_ptr<IMapBilinearInterpolateOnDPUStrategy> getMapBilinearInterpolateOnDPUStrategy(
            const bool interpolateAsSEOpInStrategy) = 0;
    virtual std::unique_ptr<IGreedilyPassStrategy> getFuseQuantizedOpsStrategy(const bool seOpsEnabled) = 0;
    virtual std::unique_ptr<IGreedilyPassStrategy> getFuseOutstandingQuantStrategy() = 0;
    virtual std::unique_ptr<IExpandActivationChannelsStrategy> getExpandActivationChannelsStrategy(bool _seOpsEnabled,
                                                                                                   Logger& _log) = 0;
    virtual std::unique_ptr<IConversionPassStrategy> getConvertToPalletizationLUTStrategy() = 0;
    virtual std::unique_ptr<IConvertToMixedPrecisionStrategy> getConvertToMixedPrecisionStrategy(
            const bool enableFloatInQuantWeightsMixedMode) = 0;
    virtual std::unique_ptr<IPropagateAndFuseQuantizeDequantizeStrategy> getPropagateAndFuseQuantizeDequantizeStrategy(
            const bool seOpsEnabled) = 0;
    virtual std::unique_ptr<D2SToTransposedConvVerifierBase> getD2SToTransposedConvVerifier() = 0;
    virtual std::unique_ptr<FuseConvertToDPUCheckerBase> getFuseConvertToDPUChecker() = 0;
    virtual std::unique_ptr<IDynamicRewriterStrategy> getInitialLowPrecisionTransformationsPipelineStrategy(
            mlir::func::FuncOp funcOp, bool enableDynamicQuantizationForStaticCase = false) = 0;

    virtual std::unique_ptr<IConvertToConvAlignmentStrategy> getConvertToConvAlignmentStrategy() {
        return nullptr;
    }
    virtual std::unique_ptr<ChannelAxisReductionWithDPUParentCheckerBase> getChannelAxisReductionWithDPUParentChecker(
            bool enableFuseReduceMinMaxToDpu) = 0;
};

class StrategyFactoryCache final : public mlir::DialectInterface::Base<StrategyFactoryCache> {
    std::unique_ptr<StrategyFactory> _strategyFactory = nullptr;

public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StrategyFactoryCache)

    StrategyFactoryCache(mlir::Dialect* dialect): Base(dialect) {
    }

    const std::unique_ptr<IE::StrategyFactory>& getStrategyFactory() const {
        return _strategyFactory;
    }

    void setStrategyFactory(std::unique_ptr<IE::StrategyFactory> strategyFactory) {
        _strategyFactory = std::move(strategyFactory);
    }
};

void setIEStrategyFactory(mlir::MLIRContext* context, std::unique_ptr<IE::StrategyFactory> factory);
const std::unique_ptr<IE::StrategyFactory>& getIEStrategyFactory(mlir::MLIRContext* context);

}  // namespace vpux::IE
