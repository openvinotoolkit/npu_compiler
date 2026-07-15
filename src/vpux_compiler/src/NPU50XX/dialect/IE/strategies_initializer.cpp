//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/strategies_initializer.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/channel_axis_reduction_with_dpu_parent_checker.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_quantize_ops_to_nce_ops_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/propagate_and_fuse_quantize_dequantize_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/impl/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/impl/map_bilinear_interpolate_on_dpu_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/convert_to_mixed_precision_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/convert_to_palletization_lut_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/expand_activation_channels_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/fuse_outstanding_quant_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/fuse_quantized_ops_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/initial_low_precision_transformations_pipeline_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/weights_dequantize_to_dynamic_dequantize_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/impl/weights_dequantize_to_fakequantize_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"

#include <mlir/IR/MLIRContext.h>

using namespace vpux;

namespace vpux::IE {
class StrategyFactory50XX : public IE::StrategyFactory {
    std::unique_ptr<IE::IConvertQuantizeOpsToNceOpsStrategy> getConvertQuantizeOpsToNceOpsStrategy() override {
        return std::make_unique<IE::arch37xx::ConvertQuantizeOpsToNceOpsStrategy>();
    }

    std::unique_ptr<IDynamicRewriterStrategy> getWeightsDequantizeToDynamicDequantizeStrategy(
            ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index) override {
        return std::make_unique<IE::arch50xx::WeightsDequantizeToDynamicDequantizeStrategy>(benefitLevels, index);
    }

    std::unique_ptr<IDynamicRewriterStrategy> getWeightsDequantizeToFakeQuantizeStrategy(
            ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index) override {
        return std::make_unique<IE::arch50xx::WeightsDequantizeToFakeQuantizeStrategy>(benefitLevels, index);
    }

    std::unique_ptr<IMapBilinearInterpolateOnDPUStrategy> getMapBilinearInterpolateOnDPUStrategy(
            const bool interpolateAsSEOpInStrategy) override {
        return std::make_unique<arch40xx::MapBilinearInterpolateOnDPUStrategy>(interpolateAsSEOpInStrategy);
    }

    std::unique_ptr<IGreedilyPassStrategy> getFuseQuantizedOpsStrategy(const bool seOpsEnabled) override {
        return std::make_unique<arch50xx::FuseQuantizedOpsStrategy>(seOpsEnabled);
    }

    std::unique_ptr<IGreedilyPassStrategy> getFuseOutstandingQuantStrategy() override {
        return std::make_unique<arch50xx::FuseOutstandingQuantStrategy>();
    }

    std::unique_ptr<IExpandActivationChannelsStrategy> getExpandActivationChannelsStrategy(const bool seOpsEnabled,
                                                                                           Logger& log) override {
        return std::make_unique<arch50xx::ExpandActivationChannelsStrategy>(seOpsEnabled, log);
    }

    std::unique_ptr<IConversionPassStrategy> getConvertToPalletizationLUTStrategy() override {
        return std::make_unique<arch50xx::ConvertToPalletizationLUTStrategy>();
    }

    std::unique_ptr<IConvertToMixedPrecisionStrategy> getConvertToMixedPrecisionStrategy(
            const bool enableFloatInQuantWeightsMixedMode) override {
        return std::make_unique<arch50xx::ConvertToMixedPrecisionStrategy>(enableFloatInQuantWeightsMixedMode);
    }

    std::unique_ptr<IPropagateAndFuseQuantizeDequantizeStrategy> getPropagateAndFuseQuantizeDequantizeStrategy(
            const bool seOpsEnabled) override {
        return std::make_unique<arch37xx::PropagateAndFuseQuantizeDequantizeStrategy>(seOpsEnabled);
    }

    std::unique_ptr<D2SToTransposedConvVerifierBase> getD2SToTransposedConvVerifier() override {
        return std::make_unique<IE::arch40xx::D2SToTransposedConvVerifier>();
    }

    std::unique_ptr<FuseConvertToDPUCheckerBase> getFuseConvertToDPUChecker() override {
        return std::make_unique<IE::FuseConvertToDPUCheckerBase>();
    }

    std::unique_ptr<IDynamicRewriterStrategy> getInitialLowPrecisionTransformationsPipelineStrategy(
            mlir::func::FuncOp func, const bool enableDynamicQuantizationForStaticCase) override {
        return std::make_unique<IE::arch50xx::InitialLowPrecisionTransformationsPipelineStrategy>(
                enableDynamicQuantizationForStaticCase, func);
    }

    std::unique_ptr<ChannelAxisReductionWithDPUParentCheckerBase> getChannelAxisReductionWithDPUParentChecker(
            bool /*enableFuseReduceMinMaxToDpu*/) override {
        return std::make_unique<IE::arch37xx::ChannelAxisReductionWithDPUParentChecker>();
    }
};
}  // namespace vpux::IE

void vpux::IE::StrategiesInitializer50XX::initialize(mlir::MLIRContext* context) {
    auto factory = std::make_unique<IE::StrategyFactory50XX>();
    IE::setIEStrategyFactory(context, std::move(factory));
}
