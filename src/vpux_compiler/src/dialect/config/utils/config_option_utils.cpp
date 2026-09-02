//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// **
// * @file config_option_utils.cpp
// * @brief Configuration option management utilities for Config Dialect
//
// * @note
// * This file should exclusively contain functions that interact with config::OptionOp
// * and the Config Dialect. Utility functions not related to the configuration system
// * should be placed in appropriate alternative modules.
// *

#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

using namespace vpux;

mlir::func::FuncOp config::getOwningFuncOp(mlir::Operation* operation) {
    if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(operation)) {
        return func;
    }
    auto func = operation->getParentOfType<mlir::func::FuncOp>();
    VPUX_THROW_WHEN(func == nullptr, "Failed to find parent function for operation '{0}'", operation->getName());
    return func;
}

// Adaptive Stripping
bool config::hasEnableAdaptiveStripping(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, ENABLE_ADAPTIVE_STRIPPING).value_or(false);
}

// Asymmetric Quantization
bool config::asymmetricPerTensorZeroPointSupported(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, ASYMMETRIC_PER_TENSOR_ZP).value_or(false);
}

bool config::asymmetricPerChannelZeroPointSupported(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, ASYMMETRIC_PER_CHANNEL_ZP).value_or(false);
}

// Auto Padding
bool config::hasAutoPadding(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, AUTO_PADDING_IDU).value_or(false) ||
           config::tryGetBoolPassOption(module, AUTO_PADDING_ODU).value_or(false);
}

bool config::hasAutoPaddingIDU(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, AUTO_PADDING_IDU).value_or(false);
}

bool config::hasAutoPaddingODU(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, AUTO_PADDING_ODU).value_or(false);
}

// Compressed Convolution
bool config::hasFP16CompressedConv(mlir::Operation* op) {
    return config::getConstraint<bool>(op, FP16_COMPRESSED_CONV);
}

// Reduce Operation Support
bool config::isReduceOpSupportedOnNCE(mlir::Operation* op) {
    return config::getConstraint<bool>(op, REDUCE_SUPPORTED);
}

// QDQ Optimization Aggressive
bool config::hasEnableQDQOptimizationAggressive(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, ENABLE_QDQ_OPTIMIZATION_AGGRESSIVE).value_or(false);
}

// SE Ptrs Operations
bool config::hasEnableSEPtrsOperations(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, ENABLE_SE_PTRS_OPERATIONS).value_or(false);
}

bool config::hasEnableExperimentalSEPtrsOperations(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, ENABLE_EXPERIMENTAL_SE_PTRS_OPERATIONS).value_or(false);
}

// Extra Static Shape Operations
bool config::hasEnableExtraStaticShapeOps(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, ENABLE_EXTRA_STATIC_SHAPE_OPS).value_or(false);
}

// Workload Management Mode
WorkloadManagementMode config::getWorkloadManagementMode(mlir::ModuleOp moduleOp) {
    auto pipelineOptionOp = moduleOp.lookupSymbol<config::PipelineOptionsOp>(config::PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch workload management mode");

    auto wlmModeConfigOp = pipelineOptionOp.lookupSymbol<config::OptionOp>(WORKLOAD_MANAGEMENT_MODE);
    VPUX_THROW_WHEN(wlmModeConfigOp == nullptr, "Failed to find config.OptionOp to fetch workload management mode");

    auto wlmModeString = mlir::dyn_cast<mlir::StringAttr>(wlmModeConfigOp.getOptionValue());
    VPUX_THROW_WHEN(wlmModeString == nullptr, "{0} config.OptionOp is expected to be a string, got {1}",
                    WORKLOAD_MANAGEMENT_MODE, wlmModeConfigOp);

    auto wlmMode = vpux::symbolizeWorkloadManagementMode(wlmModeString.getValue());
    VPUX_THROW_WHEN(!wlmMode.has_value(), "Failed to symbolize workload management mode from string '{0}'",
                    wlmModeString.getValue());

    return wlmMode.value();
}

void config::setWorkloadManagementMode(mlir::ModuleOp moduleOp, WorkloadManagementMode value) {
    auto context = moduleOp.getContext();
    auto pipelineOptionsOp = config::getPipelineOptionsOp(*context, moduleOp);
    const auto attrName = mlir::StringAttr::get(context, WORKLOAD_MANAGEMENT_MODE);
    auto attrValue = mlir::StringAttr::get(context, stringifyEnum(value));

    if (auto wlmModeConfigOp = pipelineOptionsOp.lookupSymbol<config::OptionOp>(attrName)) {
        wlmModeConfigOp.setOptionValueAttr(attrValue);
    } else {
        auto optionsBuilder = mlir::OpBuilder::atBlockBegin(&pipelineOptionsOp.getOptions().front());
        optionsBuilder.create<config::OptionOp>(optionsBuilder.getUnknownLoc(), attrName, attrValue);
    }
}

bool config::hasEnableWeightsDynamicDequantization(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, ENABLE_WEIGHTS_DYNAMIC_DEQUANTIZATION).value_or(false);
}

// VPUNN Configurations
bool config::hasVPUNNPreSplit(mlir::Operation* op) {
    return config::getConstraint<bool>(op, VPUNN_PRE_SPLIT);
}

// Preferred Spatial (H W) Alignment
int64_t config::getPreferredSpatialAlignment(mlir::Operation* op) {
    // The option may not be set for every pipeline (e.g. IE-only flows), so fall back to
    // VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT instead of throwing when the option is absent.
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<config::PipelineOptionsOp>(config::PIPELINE_OPTIONS);
    if (pipelineOptionOp == nullptr) {
        return VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
    }
    auto attrValue = pipelineOptionOp.lookupSymbol<config::OptionOp>(PREFERRED_SPATIAL_ALIGNMENT);
    if (attrValue == nullptr) {
        return VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
    }
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attrValue.getOptionValue());
    if (intAttr == nullptr) {
        return VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
    }
    const auto value = intAttr.getValue().getSExtValue();
    return value > 0 ? value : VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT;
}

// ODU Configurations
bool config::hasODULocalRegion(mlir::Operation* op) {
    return config::getConstraint<bool>(op, ODU_LOCAL_REGION);
}

// Profiling Configurations
bool config::isProfilingEnabled(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, ENABLE_PROFILING).value_or(false);
}

// Weights Table Reuse Mode
WeightsTableReuseMode config::getWeightsTableReuseMode(mlir::Operation* op) {
    return static_cast<WeightsTableReuseMode>(config::getConstraint(op, WEIGHTS_TABLE_REUSE_MODE));
}

bool config::isWeightsTableReuseEnabled(mlir::Operation* op) {
    mlir::func::FuncOp func = getOwningFuncOp(op);
    const auto weightsTableReuseMode = getWeightsTableReuseMode(func);
    return weightsTableReuseMode == WeightsTableReuseMode::ENABLED ||
           (weightsTableReuseMode == WeightsTableReuseMode::VF_ENABLED &&
            func->hasAttr(VPU::PureVerticalFusionRegionAttrName));
}

// SHAVE Engine FIFO
bool config::isFifoPerShaveEngineEnabled(mlir::Operation* op) {
    return config::getConstraint<bool>(op, config::USE_DEDICATED_FIFO_PER_SHAVE_ENGINE);
}

bool config::hasSupportForFifoPerShaveEngine(config::ArchKind arch) {
    if (arch == config::ArchKind::NPU37XX) {
        return false;
    }

    // Enable support for separate FIFO per each SHAVE engine by default.
    return true;
}

// SPRLUT Configurations
bool config::isSprLUTEnabled(mlir::Operation* op) {
    return config::getConstraint<bool>(op, SPRLUT_ENABLED);
}

// Legacy Barriers
bool config::hasUseLegacyBarriers(mlir::Operation* op) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<config::PipelineOptionsOp>(config::PIPELINE_OPTIONS);
    if (pipelineOptionOp == nullptr) {
        return false;
    }
    auto attrValue = pipelineOptionOp.lookupSymbol<config::OptionOp>(USE_LEGACY_BARRIERS);
    if (attrValue == nullptr) {
        return false;
    }
    auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attrValue.getOptionValue());
    return boolAttr != nullptr && boolAttr.getValue();
}

// Enable SoftmaxMaskAware to account for an upward adjustment of the FP16 minimum threshold.
bool config::isSoftmaxMaskAwareEnabled(mlir::Operation* op) {
    return config::getConstraint<bool>(op, SOFTMAX_MASK_AWARE);
}
double config::getSoftmaxMaskAwareThreshold(mlir::Operation* op) {
    return config::getConstraint<double>(op, SOFTMAX_MASK_AWARE_THRESHOLD);
}

bool config::isBarrierFifoDummyEntrySupported([[maybe_unused]] config::ArchKind arch) {
    return false;
}

bool config::isFinalBarrierConsumerRequired([[maybe_unused]] config::ArchKind arch,
                                            [[maybe_unused]] bool is4kWlmBarrierProgrammingMode) {
    return true;
}
