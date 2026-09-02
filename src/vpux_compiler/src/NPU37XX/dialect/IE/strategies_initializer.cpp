//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/strategies_initializer.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/adjust_quantized_conv_shape_verifier.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/channel_axis_reduction_with_dpu_parent_checker.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_quantize_ops_to_nce_ops_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_to_mixed_precision_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/expand_activation_channels_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_convert_to_dpu_checker.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_outstanding_quant_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_quantized_ops_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/initial_low_precision_transformations_pipeline_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/map_bilinear_interpolate_on_dpu_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/propagate_and_fuse_quantize_dequantize_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/se_pad_ic_perf_threshold_verifier.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/convert_weights_to_unsigned_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/IR/MLIRContext.h>

using namespace vpux;

namespace vpux::IE {
class StrategyFactory37XX : public IE::StrategyFactory {
    std::unique_ptr<IE::IConvertQuantizeOpsToNceOpsStrategy> getConvertQuantizeOpsToNceOpsStrategy() override {
        return std::make_unique<IE::arch37xx::ConvertQuantizeOpsToNceOpsStrategy>();
    }

    std::unique_ptr<IMapBilinearInterpolateOnDPUStrategy> getMapBilinearInterpolateOnDPUStrategy(
            const bool interpolateAsSEOpInStrategy) override {
        return std::make_unique<arch37xx::MapBilinearInterpolateOnDPUStrategy>(interpolateAsSEOpInStrategy);
    }

    std::unique_ptr<IGreedilyPassStrategy> getFuseQuantizedOpsStrategy(const bool seOpsEnabled) override {
        return std::make_unique<arch37xx::FuseQuantizedOpsStrategy>(seOpsEnabled);
    }

    std::unique_ptr<IGreedilyPassStrategy> getFuseOutstandingQuantStrategy() override {
        return std::make_unique<arch37xx::FuseOutstandingQuantStrategy>();
    }

    std::unique_ptr<IExpandActivationChannelsStrategy> getExpandActivationChannelsStrategy(const bool seOpsEnabled,
                                                                                           Logger& log) override {
        return std::make_unique<arch37xx::ExpandActivationChannelsStrategy>(seOpsEnabled, log);
    }

    std::unique_ptr<IConversionPassStrategy> getConvertToPalletizationLUTStrategy() override {
        VPUX_THROW("Unable to get ConvertToPalletizationLUTStrategy for NPU37XX");
        return nullptr;
    }

    std::unique_ptr<IConvertToMixedPrecisionStrategy> getConvertToMixedPrecisionStrategy(
            const bool enableFloatInQuantWeightsMixedMode) override {
        return std::make_unique<arch37xx::ConvertToMixedPrecisionStrategy>(enableFloatInQuantWeightsMixedMode);
    }

    std::unique_ptr<IPropagateAndFuseQuantizeDequantizeStrategy> getPropagateAndFuseQuantizeDequantizeStrategy(
            const bool seOpsEnabled) override {
        return std::make_unique<arch37xx::PropagateAndFuseQuantizeDequantizeStrategy>(seOpsEnabled);
    }

    std::unique_ptr<D2SToTransposedConvVerifierBase> getD2SToTransposedConvVerifier() override {
        return std::make_unique<IE::arch37xx::D2SToTransposedConvVerifier>();
    }

    std::unique_ptr<FuseConvertToDPUCheckerBase> getFuseConvertToDPUChecker() override {
        return std::make_unique<IE::arch37xx::FuseConvertToDPUChecker>();
    }

    std::unique_ptr<IDynamicRewriterStrategy> getInitialLowPrecisionTransformationsPipelineStrategy() override {
        return std::make_unique<IE::arch37xx::InitialLowPrecisionTransformationsPipelineStrategy>();
    }

    std::unique_ptr<ChannelAxisReductionWithDPUParentCheckerBase> getChannelAxisReductionWithDPUParentChecker(
            bool /*enableFuseReduceMinMaxToDpu*/) override {
        return std::make_unique<IE::arch37xx::ChannelAxisReductionWithDPUParentChecker>();
    }

    std::unique_ptr<AdjustQuantizedConvShapeVerifierBase> getAdjustQuantizedConvShapeVerifier() override {
        return std::make_unique<IE::arch37xx::AdjustQuantizedConvShapeVerifier>();
    }

    std::unique_ptr<IConvertWeightsToUnsignedStrategy> getConvertWeightsToUnsignedStrategy() override {
        return std::make_unique<IE::arch50xx::ConvertWeightsToUnsignedStrategy>();
    }

    std::unique_ptr<SEPadICPerfThresholdVerifierBase> getSEPadICPerfThresholdVerifier() override {
        return std::make_unique<IE::arch37xx::SEPadICPerfThresholdVerifier>();
    }
};
}  // namespace vpux::IE

void vpux::IE::StrategiesInitializer37XX::initialize(mlir::MLIRContext* context) {
    auto factory = std::make_unique<IE::StrategyFactory37XX>();
    IE::setIEStrategyFactory(context, std::move(factory));
}
