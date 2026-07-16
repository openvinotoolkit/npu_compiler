//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/utils/options.hpp"

#include <mlir/IR/BuiltinOps.h>

namespace vpux {
namespace config {

mlir::func::FuncOp getOwningFuncOp(mlir::Operation* operation);

// Adaptive Stripping
constexpr StringRef ENABLE_ADAPTIVE_STRIPPING = "config.EnableAdaptiveStripping";
bool hasEnableAdaptiveStripping(mlir::ModuleOp module);

// Asymmetric Quantization
constexpr StringRef ASYMMETRIC_PER_TENSOR_ZP = "config.AsymmetricPerTensorZP";
constexpr StringRef ASYMMETRIC_PER_CHANNEL_ZP = "config.AsymmetricPerChannelZP";
bool asymmetricPerTensorZeroPointSupported(mlir::ModuleOp);
bool asymmetricPerChannelZeroPointSupported(mlir::ModuleOp);

// Auto Padding
constexpr StringRef AUTO_PADDING_ODU = "config.AutoPaddingODU";
constexpr StringRef AUTO_PADDING_IDU = "config.AutoPaddingIDU";
bool hasAutoPadding(mlir::ModuleOp);
bool hasAutoPaddingODU(mlir::ModuleOp);
bool hasAutoPaddingIDU(mlir::ModuleOp);

// Compressed Convolution
constexpr StringRef FP16_COMPRESSED_CONV = "config.FP16CompressedConv";
bool hasFP16CompressedConv(mlir::Operation* op);

// Reduce Operation Support
constexpr StringRef REDUCE_SUPPORTED = "config.ReduceSupported";
bool isReduceOpSupportedOnNCE(mlir::Operation* op);

// QDQ Optimization Aggressive
constexpr StringRef ENABLE_QDQ_OPTIMIZATION_AGGRESSIVE = "config.EnableQDQOptimizationAggressive";
bool hasEnableQDQOptimizationAggressive(mlir::ModuleOp module);

// SE Ptrs Operations
constexpr StringRef ENABLE_SE_PTRS_OPERATIONS = "config.EnableSEPtrsOperations";
constexpr StringRef ENABLE_EXPERIMENTAL_SE_PTRS_OPERATIONS = "config.EnableExperimentalSEPtrsOperations";
bool hasEnableSEPtrsOperations(mlir::ModuleOp module);
bool hasEnableExperimentalSEPtrsOperations(mlir::ModuleOp module);

// Extra Static Shape Operations
constexpr StringRef ENABLE_EXTRA_STATIC_SHAPE_OPS = "config.EnableExtraStaticShapeOps";
bool hasEnableExtraStaticShapeOps(mlir::ModuleOp module);

// Fragmentation Avoid Ratio for Pipelining Large Weights
constexpr StringRef FRAGMENTATION_AVOID_RATIO_PIPELINING_LARGE_WEIGHTS =
        "config.FragmentationAvoidRatioPipeliningLargeWeights";

// Workload Management Mode
constexpr StringRef WORKLOAD_MANAGEMENT_MODE = "config.WorkloadManagementMode";
WorkloadManagementMode getWorkloadManagementMode(mlir::ModuleOp module);
void setWorkloadManagementMode(mlir::ModuleOp module, WorkloadManagementMode value);

// Weights Dynamic Dequantization
constexpr StringRef ENABLE_WEIGHTS_DYNAMIC_DEQUANTIZATION = "config.EnableWeightsDynamicDequantization";
bool hasEnableWeightsDynamicDequantization(mlir::ModuleOp module);

// Workload Management Constraints

constexpr StringRef BARR_MAX_SLOT_COUNT = "config.BarrierMaxSlotCount";
constexpr StringRef METADATA_MAX_VARIANT_COUNT = "config.MetadataMaxVariantCount";
constexpr StringRef METADATA_MAX_INVARIANT_COUNT = "config.MetadataMaxInvariantCount";
constexpr StringRef METADATA_MAX_KERNEL_INVOCATION_COUNT = "config.MetadataMaxKernelInvocationCount";
constexpr StringRef METADATA_MAX_KERNEL_RANGE_COUNT = "config.MetadataMaxKernelRangeCount";
constexpr StringRef METADATA_MAX_MEDIA_COUNT = "config.MetadataMaxMediaCount";
constexpr StringRef METADATA_MAX_DMA_COUNT = "config.MetadataMaxDMACount";

constexpr StringRef SHV_FIFO_ADDRS = "config.ShvFIFOAddrs";
constexpr StringRef DPU_FIFO_ADDRS = "config.DpuFIFOAddrs";
constexpr StringRef BARRIER_FIFO_ADDR = "config.BarrierFIFOAddr";
constexpr StringRef BARRIER_FIFO_DEPTH = "config.BarrierFIFODepth";
constexpr StringRef MAX_BARRIERS_PER_TASK = "config.MaxBarriersPerTask";
constexpr StringRef USE_LEGACY_BARRIERS = "config.UseLegacyBarriers";
bool hasUseLegacyBarriers(mlir::Operation* op);

// VPUNN Configurations
constexpr StringRef VPUNN_PRE_SPLIT = "config.EnableVPUNNPreSplit";
bool hasVPUNNPreSplit(mlir::Operation* op);

// ODU Configurations
constexpr StringRef ODU_LOCAL_REGION = "config.EnableODULocalRegion";
bool hasODULocalRegion(mlir::Operation* op);

// Profiling Configurations
constexpr StringRef ENABLE_PROFILING = "config.EnableProfiling";
bool isProfilingEnabled(mlir::ModuleOp module);

// Weights Table Reuse Mode
constexpr StringRef WEIGHTS_TABLE_REUSE_MODE = "config.WeightsTableReuseMode";
WeightsTableReuseMode getWeightsTableReuseMode(mlir::Operation* op);
bool isWeightsTableReuseEnabled(mlir::Operation* op);

// SHAVE Engine FIFO
constexpr StringRef USE_DEDICATED_FIFO_PER_SHAVE_ENGINE = "config.UseDedicatedFifoPerShaveEngine";
bool isFifoPerShaveEngineEnabled(mlir::Operation* op);
bool hasSupportForFifoPerShaveEngine(config::ArchKind arch);

// SPRLUT Configurations
constexpr StringRef SPRLUT_ENABLED = "config.SprLUTEnabled";
bool isSprLUTEnabled(mlir::Operation* op);

// Enable SoftmaxMaskAware and configure the FP16 minimum-threshold adjustment.
constexpr StringRef SOFTMAX_MASK_AWARE = "config.EnableSoftmaxMaskAware";
constexpr StringRef SOFTMAX_MASK_AWARE_THRESHOLD = "config.SoftmaxMaskAwareThreshold";
bool isSoftmaxMaskAwareEnabled(mlir::Operation* op);
double getSoftmaxMaskAwareThreshold(mlir::Operation* op);
bool isBarrierFifoDummyEntrySupported(config::ArchKind arch);

}  // namespace config
}  // namespace vpux
