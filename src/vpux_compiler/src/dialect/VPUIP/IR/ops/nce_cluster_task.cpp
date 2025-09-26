//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Support/LLVM.h>
#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// NCEClusterTaskOp::build
//

void vpux::VPUIP::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                          vpux::VPUIP::NCEClusterTaskOp task, mlir::Type profilingOutput) {
    auto operands = task->getOperands();
    assert(operands.size() >= 4u && "mismatched number of parameters");
    state.addOperands(operands);

    auto& props = state.getOrAddProperties<vpux::VPUIP::NCEClusterTaskOp::Properties>();

    auto attributes = task->getAttrs();
    auto dict = mlir::DictionaryAttr::get(builder.getContext(), attributes);
    VPUX_THROW_UNLESS(vpux::VPUIP::NCEClusterTaskOp::setPropertiesFromAttr(props, dict, nullptr).succeeded(),
                      "Cannot initialize NCEClusterTaskOp::Properties from attribute '{0}'!", dict);

    // Compute value for resultSegmentSizes attribute and add it to the properties
    int32_t outputSMVal = (task.getOutputSparsityMap() != nullptr) ? 1 : 0;
    props.setResultSegmentSizes({1, outputSMVal, 1});

    for (unsigned i = 0; i != 2; ++i) {
        state.addRegion();
    }

    state.addTypes(task.getOutput().getType());

    if (task.getOutputSparsityMap() != nullptr) {
        state.addTypes(task.getOutputSparsityMap().getType());
    }

    state.addTypes(profilingOutput);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input, mlir::Value weights,
        mlir::Value weight_table, mlir::Value weight_table_data_ptr, mlir::Value weight_table_sp_ptr,
        mlir::Value weight_table_scale, mlir::Value weight_table_bias, mlir::Value weight_zero_points,
        mlir::Value spr_lookup_table, mlir::Value pallet_lookup_table, mlir::Value parent_input,
        mlir::Value parent_output, mlir::Value output_buff, vpux::VPUIP::NCETaskType task_type,
        mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides, vpux::VPU::PaddingAttr kernel_padding,
        mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented,
        mlir::IntegerAttr out_channel_offset, mlir::UnitAttr input_channels_compression,
        mlir::UnitAttr is_zero_offset_weights_table, mlir::UnitAttr is_superdense, mlir::BoolAttr is_inplace,
        mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size, mlir::UnitAttr isPermuteQuantize,
        mlir::UnitAttr isSmallKernelOptimized, VPU::MPEEngineAttr mpeEngineAttr, VPU::EltwiseTypeAttr eltwiseType) {
    build(builder, state, output_buff.getType(), nullptr, nullptr, input, nullptr, nullptr, weights, nullptr,
          weight_table, weight_table_data_ptr, weight_table_sp_ptr, weight_table_scale, weight_table_bias,
          weight_zero_points, spr_lookup_table, pallet_lookup_table, parent_input, nullptr, nullptr, parent_output,
          nullptr, mlir::ValueRange(), output_buff, nullptr, nullptr, nullptr, nullptr, nullptr, mlir::ValueRange(),
          task_type, kernel_size, kernel_strides, kernel_padding, is_continued, cm_sp_pattern, is_segmented,
          out_channel_offset, input_channels_compression, is_zero_offset_weights_table, is_superdense, is_inplace,
          input_se_size, output_se_size, isPermuteQuantize, isSmallKernelOptimized, mpeEngineAttr, eltwiseType, nullptr,
          nullptr);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output, mlir::Value input,
        mlir::Value weights, mlir::Value weight_table, mlir::Value weight_table_data_ptr,
        mlir::Value weight_table_sp_ptr, mlir::Value weight_table_scale, mlir::Value weight_table_bias,
        mlir::Value weight_zero_points, mlir::Value spr_lookup_table, mlir::Value pallet_lookup_table,
        mlir::Value parent_input, mlir::Value parent_output, mlir::Value output_buff,
        vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides,
        vpux::VPU::PaddingAttr kernel_padding, mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern,
        mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset, mlir::UnitAttr input_channels_compression,
        mlir::UnitAttr is_zero_offset_weights_table, mlir::UnitAttr is_superdense, mlir::BoolAttr is_inplace,
        mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size, mlir::UnitAttr isPermuteQuantize,
        mlir::UnitAttr isSmallKernelOptimized, VPU::MPEEngineAttr mpeEngineAttr, VPU::EltwiseTypeAttr eltwiseType) {
    build(builder, state, output, nullptr, nullptr, input, nullptr, nullptr, weights, nullptr, weight_table,
          weight_table_data_ptr, weight_table_sp_ptr, weight_table_scale, weight_table_bias, weight_zero_points,
          spr_lookup_table, pallet_lookup_table, parent_input, nullptr, nullptr, parent_output, nullptr,
          mlir::ValueRange(), output_buff, nullptr, nullptr, nullptr, nullptr, nullptr, mlir::ValueRange(), task_type,
          kernel_size, kernel_strides, kernel_padding, is_continued, cm_sp_pattern, is_segmented, out_channel_offset,
          input_channels_compression, is_zero_offset_weights_table, is_superdense, is_inplace, input_se_size,
          output_se_size, isPermuteQuantize, isSmallKernelOptimized, mpeEngineAttr, eltwiseType, nullptr, nullptr);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input, mlir::Value weights,
        mlir::Value weight_table, mlir::Value weight_table_data_ptr, mlir::Value weight_table_sp_ptr,
        mlir::Value weight_table_scale, mlir::Value weight_table_bias, mlir::Value weight_zero_points,
        mlir::Value spr_lookup_table, mlir::Value pallet_lookup_table, mlir::Value parent_input,
        mlir::Value parent_output, mlir::Value output_buff, mlir::Value profiling_data,
        vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides,
        vpux::VPU::PaddingAttr kernel_padding, mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern,
        mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset, mlir::UnitAttr input_channels_compression,
        mlir::UnitAttr is_zero_offset_weights_table, mlir::UnitAttr is_superdense, mlir::BoolAttr is_inplace,
        mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size, mlir::UnitAttr isPermuteQuantize,
        mlir::UnitAttr isSmallKernelOptimized, VPU::MPEEngineAttr mpeEngineAttr, VPU::EltwiseTypeAttr eltwiseType) {
    build(builder, state, output_buff.getType(), nullptr, profiling_data ? profiling_data.getType() : nullptr, input,
          nullptr, nullptr, weights, nullptr, weight_table, weight_table_data_ptr, weight_table_sp_ptr,
          weight_table_scale, weight_table_bias, weight_zero_points, spr_lookup_table, pallet_lookup_table,
          parent_input, nullptr, nullptr, parent_output, nullptr, mlir::ValueRange(), output_buff, nullptr,
          profiling_data, nullptr, nullptr, nullptr, mlir::ValueRange(), task_type, kernel_size, kernel_strides,
          kernel_padding, is_continued, cm_sp_pattern, is_segmented, out_channel_offset, input_channels_compression,
          is_zero_offset_weights_table, is_superdense, is_inplace, input_se_size, output_se_size, isPermuteQuantize,
          isSmallKernelOptimized, mpeEngineAttr, eltwiseType, nullptr, nullptr);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output, mlir::Type profiling_output,
        mlir::Value input, mlir::Value weights, mlir::Value weight_table, mlir::Value weight_table_data_ptr,
        mlir::Value weight_table_sp_ptr, mlir::Value weight_table_scale, mlir::Value weight_table_bias,
        mlir::Value weight_zero_points, mlir::Value spr_lookup_table, mlir::Value pallet_lookup_table,
        mlir::Value parent_input, mlir::Value parent_output, mlir::Value output_buff, mlir::Value profiling_data,
        vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides,
        vpux::VPU::PaddingAttr kernel_padding, mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern,
        mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset, mlir::UnitAttr input_channels_compression,
        mlir::UnitAttr is_zero_offset_weights_table, mlir::UnitAttr is_superdense, mlir::BoolAttr is_inplace,
        mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size, mlir::UnitAttr isPermuteQuantize,
        mlir::UnitAttr isSmallKernelOptimized, VPU::MPEEngineAttr mpeEngineAttr, VPU::EltwiseTypeAttr eltwiseType) {
    build(builder, state, output, nullptr, profiling_output, input, nullptr, nullptr, weights, nullptr, weight_table,
          weight_table_data_ptr, weight_table_sp_ptr, weight_table_scale, weight_table_bias, weight_zero_points,
          spr_lookup_table, pallet_lookup_table, parent_input, nullptr, nullptr, parent_output, nullptr,
          mlir::ValueRange(), output_buff, nullptr, profiling_data, nullptr, nullptr, nullptr, mlir::ValueRange(),
          task_type, kernel_size, kernel_strides, kernel_padding, is_continued, cm_sp_pattern, is_segmented,
          out_channel_offset, input_channels_compression, is_zero_offset_weights_table, is_superdense, is_inplace,
          input_se_size, output_se_size, isPermuteQuantize, isSmallKernelOptimized, mpeEngineAttr, eltwiseType, nullptr,
          nullptr);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input, mlir::Value input_sparsity_map,
        mlir::Value input_storage_element_table, mlir::Value weights, mlir::Value weights_sparsity_map,
        mlir::Value weight_table, mlir::Value weight_table_data_ptr, mlir::Value weight_table_sp_ptr,
        mlir::Value weight_table_scale, mlir::Value weight_table_bias, mlir::Value weight_zero_points,
        mlir::Value spr_lookup_table, mlir::Value pallet_lookup_table, mlir::Value parent_input,
        mlir::Value parent_input_sparsity_map, mlir::Value parent_input_storage_element_table,
        mlir::Value parent_output, mlir::Value parent_output_sparsity_map, mlir::Value output_buff,
        mlir::Value output_sparsity_map_buff, mlir::Value profiling_data, mlir::Value dynamic_sequence_length,
        mlir::Value max_per_xy, mlir::Value min_per_xy, mlir::ValueRange min_max_per_tensor,
        vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides,
        vpux::VPU::PaddingAttr kernel_padding, mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern,
        mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset, mlir::UnitAttr input_channels_compression,
        mlir::UnitAttr is_zero_offset_weights_table, mlir::UnitAttr is_superdense, mlir::BoolAttr is_inplace,
        mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size, mlir::UnitAttr isPermuteQuantize,
        mlir::UnitAttr isSmallKernelOptimized, VPU::MPEEngineAttr mpeEngineAttr, VPU::EltwiseTypeAttr eltwiseType,
        vpux::VPUIP::DynamicScaleConfigAttr dynamicScaleConfig, vpux::VPUIP::LocalRegionAttr localRegion) {
    build(builder, state, output_buff.getType(),
          output_sparsity_map_buff ? output_sparsity_map_buff.getType() : nullptr,
          (profiling_data != nullptr) ? profiling_data.getType() : nullptr, input, input_sparsity_map,
          input_storage_element_table, weights, weights_sparsity_map, weight_table, weight_table_data_ptr,
          weight_table_sp_ptr, weight_table_scale, weight_table_bias, weight_zero_points, spr_lookup_table,
          pallet_lookup_table, parent_input, parent_input_sparsity_map, parent_input_storage_element_table,
          parent_output, parent_output_sparsity_map, mlir::ValueRange(), output_buff, output_sparsity_map_buff,
          profiling_data, dynamic_sequence_length, max_per_xy, min_per_xy, min_max_per_tensor, task_type, kernel_size,
          kernel_strides, kernel_padding, is_continued, cm_sp_pattern, is_segmented, out_channel_offset,
          input_channels_compression, is_zero_offset_weights_table, is_superdense, is_inplace, input_se_size,
          output_se_size, isPermuteQuantize, isSmallKernelOptimized, mpeEngineAttr, eltwiseType, dynamicScaleConfig,
          localRegion);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output, mlir::Type output_sparsity_map,
        mlir::Type profiling_output, mlir::Value input, mlir::Value input_sparsity_map,
        mlir::Value input_storage_element_table, mlir::Value weights, mlir::Value weights_sparsity_map,
        mlir::Value weight_table, mlir::Value weight_table_data_ptr, mlir::Value weight_table_sp_ptr,
        mlir::Value weight_table_scale, mlir::Value weight_table_bias, mlir::Value weight_zero_points,
        mlir::Value spr_lookup_table, mlir::Value pallet_lookup_table, mlir::Value parent_input,
        mlir::Value parent_input_sparsity_map, mlir::Value parent_input_storage_element_table,
        mlir::Value parent_output, mlir::Value parent_output_sparsity_map, mlir::Value output_buff,
        mlir::Value output_sparsity_map_buff, mlir::Value profiling_data, mlir::Value dynamic_sequence_length,
        mlir::Value max_per_xy, mlir::Value min_per_xy, mlir::ValueRange min_max_per_tensor,
        vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides,
        vpux::VPU::PaddingAttr kernel_padding, mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern,
        mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset, mlir::UnitAttr input_channels_compression,
        mlir::UnitAttr is_zero_offset_weights_table, mlir::UnitAttr is_superdense, mlir::BoolAttr is_inplace,
        mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size, mlir::UnitAttr isPermuteQuantize,
        mlir::UnitAttr isSmallKernelOptimized, VPU::MPEEngineAttr mpeEngineAttr, VPU::EltwiseTypeAttr eltwiseType,
        vpux::VPUIP::DynamicScaleConfigAttr dynamicScaleConfig, vpux::VPUIP::LocalRegionAttr localRegion) {
    build(builder, state, output, output_sparsity_map, profiling_output, input, input_sparsity_map,
          input_storage_element_table, weights, weights_sparsity_map, weight_table, weight_table_data_ptr,
          weight_table_sp_ptr, weight_table_scale, weight_table_bias, weight_zero_points, spr_lookup_table,
          pallet_lookup_table, parent_input, parent_input_sparsity_map, parent_input_storage_element_table,
          parent_output, parent_output_sparsity_map, mlir::ValueRange(), output_buff, output_sparsity_map_buff,
          profiling_data, dynamic_sequence_length, max_per_xy, min_per_xy, min_max_per_tensor, task_type, kernel_size,
          kernel_strides, kernel_padding, is_continued, cm_sp_pattern, is_segmented, out_channel_offset,
          input_channels_compression, is_zero_offset_weights_table, is_superdense, is_inplace, input_se_size,
          output_se_size, isPermuteQuantize, isSmallKernelOptimized, mpeEngineAttr, eltwiseType, dynamicScaleConfig,
          localRegion);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output, mlir::Type output_sparsity_map,
        mlir::Type profiling_output, mlir::Value input, mlir::Value input_sparsity_map,
        mlir::Value input_storage_element_table, mlir::Value weights, mlir::Value weights_sparsity_map,
        mlir::Value weight_table, mlir::Value weight_table_data_ptr, mlir::Value weight_table_sp_ptr,
        mlir::Value weight_table_scale, mlir::Value weight_table_bias, mlir::Value weight_zero_points,
        mlir::Value spr_lookup_table, mlir::Value pallet_lookup_table, mlir::Value parent_input,
        mlir::Value parent_input_sparsity_map, mlir::Value parent_input_storage_element_table,
        mlir::Value parent_output, mlir::Value parent_output_sparsity_map, mlir::ValueRange output_ITI_buff,
        mlir::Value output_buff, mlir::Value output_sparsity_map_buff, mlir::Value profiling_data,
        mlir::Value dynamic_sequence_length, mlir::Value max_per_xy, mlir::Value min_per_xy,
        mlir::ValueRange min_max_per_tensor, vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
        mlir::ArrayAttr kernel_strides, vpux::VPU::PaddingAttr kernel_padding, mlir::UnitAttr is_continued,
        mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset,
        mlir::UnitAttr input_channels_compression, mlir::UnitAttr is_zero_offset_weights_table,
        mlir::UnitAttr is_superdense, mlir::BoolAttr is_inplace, mlir::IntegerAttr input_se_size,
        mlir::IntegerAttr output_se_size, mlir::UnitAttr isPermuteQuantize, mlir::UnitAttr isSmallKernelOptimized,
        VPU::MPEEngineAttr mpeEngineAttr, VPU::EltwiseTypeAttr eltwiseType,
        vpux::VPUIP::DynamicScaleConfigAttr dynamicScaleConfig, vpux::VPUIP::LocalRegionAttr localRegion) {
    auto taskTypeAttr = vpux::VPUIP::NCETaskTypeAttr::get(builder.getContext(), task_type);

    build(builder, state, output, output_sparsity_map, profiling_output, input, input_sparsity_map,
          input_storage_element_table, weights, weights_sparsity_map, weight_table, weight_table_data_ptr,
          weight_table_sp_ptr, weight_table_scale, weight_table_bias, weight_zero_points, spr_lookup_table,
          pallet_lookup_table, parent_input, parent_input_sparsity_map, parent_input_storage_element_table,
          parent_output, parent_output_sparsity_map, output_ITI_buff, output_buff, output_sparsity_map_buff,
          profiling_data, dynamic_sequence_length, max_per_xy, min_per_xy, min_max_per_tensor, taskTypeAttr,
          eltwiseType, kernel_size, kernel_strides, kernel_padding, is_continued, cm_sp_pattern, is_segmented,
          out_channel_offset, input_channels_compression, is_zero_offset_weights_table, is_superdense, is_inplace,
          input_se_size, output_se_size, isPermuteQuantize, isSmallKernelOptimized,
          /*profilingMetadata=*/nullptr, mpeEngineAttr, dynamicScaleConfig, localRegion);

    // The auto-generated builders don't populate the regions even if SizedRegion<1> is specified.
    for (auto& region : state.regions) {
        region->emplaceBlock();
    }
}

//
// NCEClusterTaskOp::addDPUTask
//

VPUIP::DPUTaskOp vpux::VPUIP::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr outStart,
                                                           mlir::ArrayAttr outEnd, mlir::ArrayAttr inStart,
                                                           mlir::ArrayAttr inEnd, VPU::PaddingAttr pad,
                                                           VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId) {
    if (getVariants().empty()) {
        getVariants().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&getVariants().front());

    return builder.create<VPUIP::DPUTaskOp>(getLoc(), outStart, outEnd, inStart, inEnd, pad, mpeMode, clusterId);
}

VPUIP::DPUTaskOp vpux::VPUIP::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr outStart,
                                                           mlir::ArrayAttr outEnd, mlir::ArrayAttr inStart,
                                                           mlir::ArrayAttr inEnd, VPU::PaddingAttr pad,
                                                           VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId,
                                                           mlir::ArrayAttr haloRegions) {
    if (getVariants().empty()) {
        getVariants().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&getVariants().front());

    return builder.create<VPUIP::DPUTaskOp>(getLoc(), outStart, outEnd, inStart, inEnd, pad, mpeMode, clusterId,
                                            haloRegions);
}

VPUIP::DPUTaskOp vpux::VPUIP::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr outStart,
                                                           mlir::ArrayAttr outEnd, VPU::PaddingAttr pad,
                                                           VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId) {
    if (getVariants().empty()) {
        getVariants().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&getVariants().front());

    return builder.create<VPUIP::DPUTaskOp>(getLoc(), outStart, outEnd, pad, mpeMode, clusterId);
}

VPUIP::DPUTaskOp vpux::VPUIP::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr outStart,
                                                           mlir::ArrayAttr outEnd, mlir::ArrayAttr inStart,
                                                           mlir::ArrayAttr inEnd, VPU::PaddingAttr pad,
                                                           VPU::MPEMode mpeMode, mlir::ArrayAttr variant_primitive_id) {
    if (getVariants().empty()) {
        getVariants().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&getVariants().front());

    return builder.create<VPUIP::DPUTaskOp>(getLoc(), outStart, outEnd, inStart, inEnd, pad, mpeMode,
                                            /*cluster_id=*/nullptr,
                                            /*halo_regions=*/nullptr, /*workload_id =*/nullptr, variant_primitive_id);
}

//
// NCEClusterTaskOp::getNumVariants
//

int64_t vpux::VPUIP::NCEClusterTaskOp::getNumVariants() {
    return getVariants().getBlocks().front().getOperations().size();
}

size_t vpux::VPUIP::NCEClusterTaskOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto nceOp = mlir::cast<VPUIP::NCEClusterTaskOp>(this->getOperation());

    auto module = nceOp->getParentOfType<mlir::ModuleOp>();
    // TODO: Expose API to get arch from cost model
    auto arch = config::getArch(module);
    auto tileOp = config::getTileExecutor(module);
    VPUX_THROW_WHEN(tileOp == nullptr, "Couldn't get TileExecutor for module");

    vpux::Logger log = Logger::global();
    auto dpuCount = tileOp.getSubExecutor(VPU::ExecutorKind::DPU).getCount();
    return checked_cast<size_t>(calculateNceCycles(nceOp, costModel, arch, log, dpuCount));
}

//
// NCEClusterTaskOp::inferReturnTypes
//

mlir::LogicalResult vpux::VPUIP::NCEClusterTaskOp::inferReturnTypes(
        mlir::MLIRContext*, std::optional<mlir::Location>, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::OpaqueProperties props, mlir::RegionRange ranges,
        llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    VPUIP::NCEClusterTaskOpAdaptor adaptor(operands, attrs, props, ranges);
    inferredReturnTypes.push_back(adaptor.getOutputBuff().getType());
    if (adaptor.getOutputSparsityMapBuff() != nullptr) {
        inferredReturnTypes.push_back(adaptor.getOutputSparsityMapBuff().getType());
    }
    if (adaptor.getProfilingData() != nullptr) {
        inferredReturnTypes.push_back(adaptor.getProfilingData().getType());
    }
    return mlir::success();
}

//
// verify
//

namespace {

mlir::LogicalResult verifyInOutOrder(mlir::Operation* op, const std::string& opName) {
    const auto inOrder = DimsOrder::fromValue(op->getOperand(0));
    if (inOrder != DimsOrder::NHWC) {
        return errorAt(op, "{0} input must have NHWC layout, got '{1}'", opName, inOrder);
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEConv(VPUIP::NCEClusterTaskOp op, config::ArchKind arch) {
    VPUX_THROW_UNLESS(op.getTaskType() == VPUIP::NCETaskType::CONV, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::CONV, op.getTaskType());

    bool doAccuLoad = false;

    if (!doAccuLoad && op.getWeights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if ((op.getWeightTableDataPtr() || op.getWeightTableSpPtr() || op.getWeightTableScale() ||
         op.getWeightTableBias() || op.getWeightZeroPoints())) {
        return errorAt(op, "Only weight_table can be populated for NCETaskType : '{0}'", op.getTaskType());
    } else if (doAccuLoad == false && op.getWeightTableDataPtr() && op.getWeightZeroPoints()) {
        return errorAt(op,
                       "weight_table data pointers and zero points only are mutually exclusive for NCETaskType : '{0}'",
                       op.getTaskType());
    } else if (doAccuLoad == false && ((op.getWeightTableScale() && op.getWeightTableBias() == nullptr) ||
                                       (op.getWeightTableScale() == nullptr && op.getWeightTableBias()))) {
        return errorAt(op, "weight_table scale and bias are required together for NCETaskType : '{0}'",
                       op.getTaskType());
    }
    if (op.getKernelSizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelStridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelPaddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.getTaskType());
    }
    const auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.getKernelStridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = op.getKernelPaddingAttr();
    const auto padLeft = kernelPadding.getLeft().getInt();
    const auto padRight = kernelPadding.getRight().getInt();
    const auto padTop = kernelPadding.getTop().getInt();
    const auto padBottom = kernelPadding.getBottom().getInt();
    if (mlir::failed(VPU::NCEInvariant::verifyKernel(op, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight))) {
        return errorAt(op, "Kernel verification failed");
    }

    {
        const auto weightsShape = getShape(op.getWeights());
        auto OC = weightsShape[Dims4D::Filter::OC];

        const auto weightTableShape = getShape(op.getWeightTable());
        const auto weightTableNumElements = weightTableShape.totalSize();

        if (VPU::canAutopadOutput(op)) {
            OC = vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
        }

        if (OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC > weightTableNumElements) {
            return errorAt(op, "Weight table must have elements greater than or equal to '{0}', got '{1}'",
                           OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, weightTableNumElements);
        }
    }

    const auto inOrder = DimsOrder::fromValue(op.getInput());
    if (op.getTaskType() == VPUIP::NCETaskType::CONV && (inOrder != DimsOrder::NHWC && inOrder != DimsOrder::GNHWC)) {
        return errorAt(op, "For NCE z-major convolution input must have NHWC or GNHWC layout, got '{0}'", inOrder);
    }

    if (doAccuLoad == false) {
        const auto weightsOrder = DimsOrder::fromValue(op.getWeights());

        if (weightsOrder != DimsOrder::OYXI && weightsOrder != DimsOrder::GOYXI) {
            return errorAt(op, "For NCE convolution weights must have OYXI layout, got '{0}'", weightsOrder);
        }
    }

    const auto outputShape = getShape(op.getOutput());

    const auto isOutput5d = outputShape.size() == 5;

    if (isOutput5d) {  // if 5D, then that is grouped matmul and checks below are not applicable
        return mlir::success();
    }

    const auto batch = outputShape[Dims4D::Act::N];
    if (batch != vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
        if (arch < config::ArchKind::NPU37XX) {
            return errorAt(op, "Got unsupported input batch '{0}' expected '{1}'", batch,
                           vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE);
        }
        if (batch > vpux::VPU::getMaxArchDPUClusterNum(arch)) {
            return errorAt(op, "Got unsupported input batch '{0}' expected to be less than or equal to '{1}'", batch,
                           vpux::VPU::getMaxArchDPUClusterNum(arch));
        }
    }

    return mlir::success();
}
mlir::LogicalResult verifyNCEPool(VPUIP::NCEClusterTaskOp op) {
    VPUX_THROW_UNLESS(
            op.getTaskType() == VPUIP::NCETaskType::AVEPOOL || op.getTaskType() == VPUIP::NCETaskType::MAXPOOL,
            "Expected task type '{0}' or '{1}', but got '{2}'", VPUIP::NCETaskType::AVEPOOL,
            VPUIP::NCETaskType::MAXPOOL, op.getTaskType());

    if (op.getKernelSizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelStridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelPaddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.getTaskType());
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.getKernelStridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = op.getKernelPaddingAttr();
    const auto padLeft = kernelPadding.getLeft().getInt();
    const auto padRight = kernelPadding.getRight().getInt();
    const auto padTop = kernelPadding.getTop().getInt();
    const auto padBottom = kernelPadding.getBottom().getInt();

    if (mlir::failed(VPU::NCEInvariant::verifyKernel(op, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight))) {
        return errorAt(op, "Kernel verification failed");
    }

    return verifyInOutOrder(op, "Pooling");
}

mlir::LogicalResult verifyNCEEltwise(VPUIP::NCEClusterTaskOp op, config::ArchKind) {
    VPUX_THROW_UNLESS(op.getTaskType() == VPUIP::NCETaskType::ELTWISE, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::ELTWISE, op.getTaskType());
    if (op.getKernelSizeAttr() != nullptr) {
        return errorAt(op, "kernel_size should be empty for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelStridesAttr() != nullptr) {
        return errorAt(op, "kernel_strides should be empty for NCETaskType : '{0}'", op.getTaskType());
    }
    if (!VPU::hasZeroPadding(op.getKernelPaddingAttr())) {
        return errorAt(op, "kernel_padding should be empty for NCETaskType : '{0}'", op.getTaskType());
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEDWConv(VPUIP::NCEClusterTaskOp op, [[maybe_unused]] config::ArchKind arch) {
    VPUX_THROW_UNLESS(op.getTaskType() == VPUIP::NCETaskType::DWCONV, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::DWCONV, op.getTaskType());

    if (op.getWeights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if ((op.getWeightTableDataPtr() || op.getWeightTableSpPtr() || op.getWeightTableScale() ||
         op.getWeightTableBias() || op.getWeightZeroPoints())) {
        return errorAt(op, "Only weight_table can be populated for NCETaskType : '{0}'", op.getTaskType());
    } else if (op.getWeightTableDataPtr() && op.getWeightZeroPoints()) {
        return errorAt(op,
                       "weight_table data pointers and zero points only are mutually exclusive for NCETaskType : '{0}'",
                       op.getTaskType());
    } else if ((op.getWeightTableScale() && op.getWeightTableBias() == nullptr) ||
               (op.getWeightTableScale() == nullptr && op.getWeightTableBias())) {
        return errorAt(op, "weight_table scale and bias are required together for NCETaskType : '{0}'",
                       op.getTaskType());
    }
    if (op.getKernelSizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelStridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelPaddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.getTaskType());
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.getKernelStridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = op.getKernelPaddingAttr();
    const auto padLeft = kernelPadding.getLeft().getInt();
    const auto padRight = kernelPadding.getRight().getInt();
    const auto padTop = kernelPadding.getTop().getInt();
    const auto padBottom = kernelPadding.getBottom().getInt();

    if (mlir::failed(VPU::NCEInvariant::verifyKernel(op, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight))) {
        return errorAt(op, "Kernel verification failed");
    }

    {
        const auto weightsShape = getShape(op.getWeights());
        auto OC = weightsShape[Dims4D::Filter::OC];

        const auto weightTableShape = getShape(op.getWeightTable());
        const auto weightTableNumElements = weightTableShape.totalSize();

        if (VPU::canAutopadOutput(op)) {
            OC = vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
        }

        if (OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC > weightTableNumElements) {
            return errorAt(op, "Weight table must have elements greater than or equal to '{0}' elements, got '{1}'",
                           OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, weightTableNumElements);
        }
    }

    const auto weightsLayout = DimsOrder::fromValue(op.getWeights());
    if (weightsLayout != DimsOrder::NHWC) {
        return errorAt(op, "weights layout must be NHWC, got {0}", weightsLayout);
    }

    return verifyInOutOrder(op, "DWCONV");
}

mlir::LogicalResult verifyDimensions(mlir::Operation* op, mlir::Value operand) {
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operand.getType());
    if (distributedType != nullptr) {
        // check per cluster shape
        auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
        for (const auto& clusterShape : perClusterShapes) {
            if (vpux::VPU::NCEInvariant::hasDimensionExceedingVPULimit(clusterShape)) {
                return errorAt(op, "Op dimensions exceed VPU_DIMENSION_LIMIT: {0}", clusterShape);
            }
        }
    } else {
        auto operandShape = getShape(operand);
        if (vpux::VPU::NCEInvariant::hasDimensionExceedingVPULimit(operandShape)) {
            return errorAt(op, "Op dimensions exceed VPU_DIMENSION_LIMIT: {0}", operandShape);
        }
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::DPUTaskOp::verify() {
    const auto op = getOperation();
    static const size_t NUM_WORKLOAD_DIMS = 3;

    if (getOutStart().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "output start coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, getOutStart().size());
    }
    if (getOutEnd().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "output end coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, getOutEnd().size());
    }
    if (getInStart().has_value() && getInStart().value().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "input start coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS,
                       getInStart().value().size());
    }
    if (getInEnd().has_value() && getInEnd().value().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "input end coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS,
                       getInEnd().value().size());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEClusterTaskOp::verify() {
    const auto op = getOperation();
    auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(module);

    // Check dimensions of input and output
    if (getInput()) {
        if (mlir::failed(verifyDimensions(op, getInput()))) {
            return mlir::failure();
        }
    }
    if (mlir::failed(verifyDimensions(op, getOutput()))) {
        return mlir::failure();
    }

    if (getTaskType() == VPUIP::NCETaskType::CONV) {
        if (mlir::failed(verifyNCEConv(*this, arch))) {
            return mlir::failure();
        }
    } else if (getTaskType() == VPUIP::NCETaskType::MAXPOOL || getTaskType() == VPUIP::NCETaskType::AVEPOOL) {
        if (mlir::failed(verifyNCEPool(*this))) {
            return mlir::failure();
        }
    } else if (getTaskType() == VPUIP::NCETaskType::ELTWISE) {
        if (mlir::failed(verifyNCEEltwise(*this, arch))) {
            return mlir::failure();
        }
    } else if (getTaskType() == VPUIP::NCETaskType::DWCONV) {
        if (mlir::failed(verifyNCEDWConv(*this, arch))) {
            return mlir::failure();
        }
    } else {
        return errorAt(op, "NCE Task Type '{0}' is not supported", getTaskType());
    }

    size_t numDPUTasks = 0;
    for (auto& dpuOp : getVariants().getOps()) {
        if (!mlir::isa<VPUIP::DPUTaskOp>(dpuOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'variants' region", dpuOp.getName());
        }

        ++numDPUTasks;
    }

    static const size_t MIN_NUM_DPUS_PER_CLUSTER = 1;

    if (numDPUTasks < MIN_NUM_DPUS_PER_CLUSTER) {
        return errorAt(op, "There should be at least {0} DPU Tasks per NCEClusterTask, but got {1}",
                       MIN_NUM_DPUS_PER_CLUSTER, numDPUTasks);
    }

    for (auto& ppeOp : getPpe().getOps()) {
        if (!mlir::isa<VPUIP::PPETaskOp>(ppeOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'PPE' region", ppeOp.getName());
        }
    }

    const auto appendToVector = [](SmallVector<mlir::Value>& operands, mlir::Value val) {
        if (val != nullptr) {
            operands.push_back(val);
        }
    };
    auto nnCMXOperands = SmallVector<mlir::Value>();

    ShapeRef inputShape;
    if (getInput() != nullptr) {
        inputShape = getShape(getInput());
    }

    const auto isInput4d = inputShape.size() == 4;
    if (isInput4d) {
        const auto inputBatch = inputShape[Dims4D::Act::N];
        if (inputBatch != vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
            if (arch < config::ArchKind::NPU37XX) {
                return errorAt(op, "Got unsupported input batch '{0}' expected '{1}'", inputBatch,
                               vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE);
            }

            // Verify NPU37XX, NPU40XX and future version with batch tiling feature
            if (inputBatch > vpux::VPUIP::getNumTilesUsed(module)) {
                return errorAt(op, "Got unsupported input batch '{0}' expected to be less than or equal to '{1}'",
                               inputBatch, getNumTilesUsed(module));
            }
            if (vpux::VPUIP::hasDistributedOperand(op)) {
                auto outputType = mlir::cast<vpux::VPUIP::DistributedBufferType>(op->getResult(0).getType());
                const auto numClusters = outputType.getDistribution().getNumClusters().getInt();
                if (inputBatch != numClusters) {
                    return errorAt(op, "Got unsupported input batch '{0}' expected '{1}'", inputBatch, numClusters);
                }
            } else if (auto outputType = mlir::dyn_cast_or_null<VPUIP::DistributedBufferType>(getOutput().getType())) {
                const auto numClusters = outputType.getDistribution().getNumClusters().getInt();
                if (inputBatch != numClusters) {
                    return errorAt(op, "Got unsupported input batch '{0}' expected '{1}'", inputBatch, numClusters);
                }
            }
        }
    }  // else if (is5D) no limitation

    appendToVector(nnCMXOperands, getInput());
    appendToVector(nnCMXOperands, getWeights());
    appendToVector(nnCMXOperands, getWeightTable());
    appendToVector(nnCMXOperands, getOutputBuff());
    appendToVector(nnCMXOperands, getProfilingData());

    const auto checkMemoryKind = [&op](mlir::ValueRange operands, EnumSet<VPU::MemoryKind> acceptedMemoryKinds) {
        for (const auto& val : operands) {
            const auto type = mlir::cast<vpux::NDTypeInterface>(val.getType());

            const auto mem = type.getMemoryKind();
            if (llvm::find(acceptedMemoryKinds, mem) == acceptedMemoryKinds.end()) {
                return errorAt(op, "Can't operate with '{0}' MemoryKind.", mem);
            }
        }
        return mlir::success();
    };

    const auto nncmxStatus = checkMemoryKind(
            nnCMXOperands, EnumSet<VPU::MemoryKind>({VPU::MemoryKind::CMX_NN, VPU::MemoryKind::Register}));
    if (nncmxStatus.failed()) {
        return nncmxStatus;
    }

    // TODO revisit memory checks for parent operands

    for (const auto& val : getOperands()) {
        const auto type = mlir::cast<vpux::NDTypeInterface>(val.getType());
        const auto strideReqs = StrideReqs().add(DimStrideReq::compact(MemDim(type.getRank() - 1)));

        if (!strideReqs.checkStrides(val)) {
            return errorAt(op, "Value '{0}' strides do not match requirements '{1}'", val, strideReqs);
        }
    }

    if (arch >= config::ArchKind::NPU40XX) {
        auto outputType = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(getOutput().getType());
        auto outputItiBuffs = getOutput_ITIBuff();

        if (outputType == nullptr && !outputItiBuffs.empty()) {
            return errorAt(op, "Output is not of VPUIP::ITIBufferType, but output_ITI_buffs is not empty.");
        }

        for (const auto itiOutput : outputItiBuffs) {
            if (!mlir::isa<vpux::VPUIP::ITIBufferType>(itiOutput.getType())) {
                return errorAt(op, "ITI Output is not of VPUIP::ITIBufferType: {0}", itiOutput);
            }
        }

        if (getOutputSparsityMap() != nullptr && outputType != nullptr) {
            if (!mlir::isa<vpux::VPUIP::ITIBufferType>(getOutputSparsityMap().getType())) {
                return errorAt(op, "Output is of VPUIP::ITIBufferType, but output sparsity map is not.");
            }
        }
    }

    return mlir::success();
}
