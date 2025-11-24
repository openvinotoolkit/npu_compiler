//
// Copyright (C) 2025 Intel Corporation.
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
#include "vpux/compiler/utils/VPU/function_outlining_splitter.hpp"

using namespace vpux;

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

// Max Kernel Size
bool config::hasMaxKernelSize(mlir::Operation* op) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<config::PipelineOptionsOp>(config::PIPELINE_OPTIONS);
    if (pipelineOptionOp != nullptr) {
        auto attrValue = pipelineOptionOp.lookupSymbol<config::OptionOp>(config::MAX_KERNEL_SIZE);
        if (attrValue != nullptr) {
            return true;
        }
    }
    return false;
}

int64_t config::getMaxKernelSize(mlir::Operation* op) {
    return config::getConstraint<int64_t>(op, MAX_KERNEL_SIZE);
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

// Workload Management Status
WorkloadManagementStatus config::getWorkloadManagementStatus(mlir::ModuleOp moduleOp) {
    auto pipelineOptionOp = moduleOp.lookupSymbol<config::PipelineOptionsOp>(config::PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch workload management status");

    auto wlmStatusConfigOp = pipelineOptionOp.lookupSymbol<config::OptionOp>(WORKLOAD_MANAGEMENT_STATUS);
    VPUX_THROW_WHEN(wlmStatusConfigOp == nullptr, "Failed to find config.OptionOp to fetch workload management status");

    auto wlmStatusString = mlir::dyn_cast<mlir::StringAttr>(wlmStatusConfigOp.getOptionValue());
    VPUX_THROW_WHEN(wlmStatusString == nullptr, "{0} config.OptionOp is expected to be a string, got {1}",
                    WORKLOAD_MANAGEMENT_STATUS, wlmStatusConfigOp);

    auto wlmStatus = vpux::symbolizeWorkloadManagementStatus(wlmStatusString.getValue());
    VPUX_THROW_WHEN(!wlmStatus.has_value(), "Failed to symbolize workload management status from string '{0}'",
                    wlmStatusString.getValue());

    return wlmStatus.value();
}

void config::setWorkloadManagementStatus(mlir::ModuleOp moduleOp, WorkloadManagementStatus value) {
    auto context = moduleOp.getContext();
    auto pipelineOptionsOp = config::getPipelineOptionsOp(*context, moduleOp);
    const auto attrName = mlir::StringAttr::get(context, WORKLOAD_MANAGEMENT_STATUS);
    auto attrValue = mlir::StringAttr::get(context, stringifyEnum(value));

    if (auto wlmStatusConfigOp = pipelineOptionsOp.lookupSymbol<config::OptionOp>(attrName)) {
        wlmStatusConfigOp.setOptionValueAttr(attrValue);
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

// Profiling Configurations
bool config::isProfilingEnabled(mlir::ModuleOp module) {
    return config::tryGetBoolPassOption(module, ENABLE_PROFILING).value_or(false);
}

// Weights Table Reuse Mode
WeightsTableReuseMode config::getWeightsTableReuseMode(mlir::Operation* op) {
    return static_cast<WeightsTableReuseMode>(config::getConstraint(op, WEIGHTS_TABLE_REUSE_MODE));
}

bool config::isWeightsTableReuseEnabled(mlir::Operation* op) {
    mlir::func::FuncOp func;
    if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
        func = funcOp;
    } else {
        func = op->getParentOfType<mlir::func::FuncOp>();
    }
    VPUX_THROW_WHEN(func == nullptr, "Cannot find parent function for operation '{0}'", op->getName());
    const auto weightsTableReuseMode = getWeightsTableReuseMode(func);
    return weightsTableReuseMode == WeightsTableReuseMode::ENABLED ||
           (weightsTableReuseMode == WeightsTableReuseMode::VF_ENABLED &&
            func->hasAttr(VPU::PureVerticalFusionRegionAttrName));
}

// SHAVE Engine FIFO
bool config::isFifoPerShaveEngineEnabled(mlir::Operation* op) {
    return config::getConstraint<bool>(op, config::USE_DEDICATED_FIFO_PER_SHAVE_ENGINE);
}

bool config::hasSupportForFifoPerShaveEngine(config::ArchKind arch, bool enableWorkloadManagement) {
    if (!enableWorkloadManagement) {
        return false;
    }

    if (arch == config::ArchKind::NPU37XX) {
        return false;
    }

    // Enable support for separate FIFO per each SHAVE engine by default.
    return true;
}
