//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

namespace vpux {
namespace VPUIP {

// TODO: E60214, need support more sw kernel task type. Currently only enable MVN
const SmallVector<StringLiteral> SW_KERNELS_SUPPORTING_TILING = {"mvn1",
                                                                 "mvn1_sum",
                                                                 "mvn1_norm",
                                                                 "mvn6",
                                                                 "interpolate",
                                                                 "activation_swish",
                                                                 "activation_gelu",
                                                                 "softmax",
                                                                 "log_softmax",
                                                                 "matmul",
                                                                 "activation_hswish",
                                                                 "activation_hardsigmoid",
                                                                 "convert",
                                                                 "activation_tanh",
                                                                 "topk",
                                                                 "gather",
                                                                 "gather_elements",
                                                                 "activation_sigmoid",
                                                                 "depth_to_space",
                                                                 "activation_clamp",
                                                                 "eltwise_add",
                                                                 "eltwise_sub",
                                                                 "eltwise_power",
                                                                 "eltwise_mul",
                                                                 "eltwise_div",
                                                                 "eltwise_min",
                                                                 "eltwise_max",
                                                                 "eltwise_greater",
                                                                 "eltwise_less",
                                                                 "eltwise_equal",
                                                                 "eltwise_select",
                                                                 "eltwise_and",
                                                                 "activation_sin",
                                                                 "activation_cos",
                                                                 "activation_exp",
                                                                 "activation_abs",
                                                                 "prelu_fp16",
                                                                 "normalize_l2",
                                                                 "reduce_l1",
                                                                 "reduce_l2",
                                                                 "reduce_logical_and",
                                                                 "reduce_logical_or",
                                                                 "reduce_max",
                                                                 "reduce_mean",
                                                                 "reduce_min",
                                                                 "reduce_prod",
                                                                 "reduce_sum",
                                                                 "gru_sequence",
                                                                 "gru_sequence_last_part",
                                                                 "activation_floor",
                                                                 "activation_ceil",
                                                                 "activation_log",
                                                                 "activation_sqrt",
                                                                 "fake_quantize",
                                                                 "detection_output_sort",
                                                                 "lstm_gates",
                                                                 "lstm_cell",
                                                                 "lstm_sequence",
                                                                 "round_fp16",
                                                                 "accumulate",
                                                                 "populate_weight_table",
                                                                 "dequantize",
                                                                 "activation_mish",
                                                                 "dynamic_dequantize",
                                                                 "rms_norm",
                                                                 "rope"};

const SmallVector<StringLiteral> SW_KERNELS_SUPPORTING_STRIDE = {"mvn1", "lstm_cell", "lstm_sequence"};

const SmallVector<std::string_view> SW_KERNELS_SUPPORTING_SHAVE_BALANCING = {
        "softmax",     "eltwise_mul", "activation_sin", "activation_cos", "activation_swish", "activation_clamp",
        "eltwise_min", "eltwise_max", "round_fp16",     "activation_exp", "prelu_fp16"};

const SmallVector<StringLiteral> SW_KERNELS_LAYOUT_AGNOSTIC = {
        "activation_swish", "activation_gelu",    "activation_hswish", "activation_hardsigmoid",
        "activation_tanh",  "activation_sigmoid", "activation_clamp",  "activation_sin",
        "activation_cos",   "activation_exp",     "activation_abs",    "activation_log",
        "activation_sqrt",  "hswish_fp16",        "round_fp16",        "eltwise_mul",
        "activation_mish"};

// TODO: E#117136, use heuristic for tile dim
const SmallVector<StringLiteral> SW_ACTIVATION_KERNELS = {
        "activation_swish",   "activation_gelu",  "activation_hardsigmoid", "activation_tanh",
        "activation_sigmoid", "activation_clamp", "activation_abs",         "activation_floor",
        "activation_sin",     "activation_cos",   "activation_exp",         "hswish_fp16",
        "prelu_fp16",         "activation_mish"};

constexpr StringLiteral SW_KERNEL_NAME_PREFIX = "builtin_";
constexpr StringLiteral populateWeightTableWithShave = "populateWeightTable";
constexpr StringLiteral weightsPtrsPerClusterAttr = "weightsPtrsPerClusterAttr";

constexpr Byte SIGMOID_SW_KERNEL_TILING_THRESHOLD = Byte(4096);

// SwKernel list can get better performance with tiling alignment configured
const SmallVector<StringLiteral> SW_KERNELS_NEED_TILING_ALIGNMENT = {"mvn1",
                                                                     "mvn6",
                                                                     "activation_swish",
                                                                     "activation_gelu",
                                                                     "softmax",
                                                                     "activation_hswish",
                                                                     "eltwise_mul",
                                                                     "activation_hardsigmoid",
                                                                     "convert",
                                                                     "activation_tanh",
                                                                     "activation_sigmoid",
                                                                     "activation_clamp",
                                                                     "eltwise_min",
                                                                     "eltwise_max",
                                                                     "eltwise_power",
                                                                     "activation_abs",
                                                                     "eltwise_div",
                                                                     "prelu_fp16",
                                                                     "normalize_l2",
                                                                     "eltwise_greater",
                                                                     "eltwise_less",
                                                                     "eltwise_sub",
                                                                     "eltwise_add",
                                                                     "activation_floor",
                                                                     "activation_log",
                                                                     "activation_sqrt",
                                                                     "fake_quantize",
                                                                     "eltwise_select",
                                                                     "activation_sin",
                                                                     "activation_cos",
                                                                     "lstm_cell",
                                                                     "activation_mish"};

SmallVector<mlir::Attribute> kernelArgsRange(VPUIP::SwKernelOp swKernelOp);

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, mlir::StringRef builtInFunctionName,
                                          const ArrayRef<mlir::Type> inputTypes, mlir::StringRef kernelEntryName,
                                          mlir::StringRef kernelSourceFileName, mlir::StringRef layerName,
                                          const vpux::Logger& log);

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, VPU::LayerOpInterface origOp,
                                          ArrayRef<mlir::Value> operands, ArrayRef<mlir::Value> results,
                                          const VPUIP::KernelInfo& kernelInfo, const Logger& log);

void createRuntimeKernelDefinition(mlir::ModuleOp module, const Logger& log, vpux::VPU::ArchKind arch);

void initSwKernel(vpux::VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs, mlir::ValueRange outputBuffs,
                  mlir::ArrayRef<mlir::Attribute> args, const vpux::Logger& log,
                  VPUIP::SwKernelRun swKernelRunOp = nullptr);

void initSwKernel(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRunOp, const vpux::Logger& log);

SmallString getSwKernelEntryName(VPUIP::SwKernelOp swKernelOp);
mlir::ModuleOp getVPUSWModule(mlir::ModuleOp module, const Logger& log);
bool isActivationSwKernelOp(VPUIP::SwKernelOp swKernelOp);
bool isSwKernelTilingSupported(VPUIP::SwKernelOp swKernelOp);
bool isStridedDataAccessSupported(VPUIP::SwKernelOp swKernelOp);

InputTiling backInferSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const SmallVector<vpux::TileInfo>& outputTiles,
                                       int tileId, Logger log);

SmallVector<mlir::Attribute> getSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                            ArrayRef<mlir::Attribute> origAttr,
                                                            const TilingInfo& inputTiling, const TileInfo& outTile,
                                                            Logger log);

SmallVector<int64_t> getPopulateWeightTableSwKernelEntries(VPUIP::SwKernelOp swKernelOp);
void updatePopulateWeightTableSwKernel(VPUIP::SwKernelOp swKernelOp, int64_t currOffset, Logger log);

SmallVector<vpux::NDTypeInterface> getSwKernelTiledTypes(VPUIP::SwKernelOp swKernelOp, Dim tileDim);

bool isCacheOpTaskType(std::optional<::mlir::SymbolRefAttr> kernelTaskType, bool includePrefetch = true);
bool isCacheOpTaskType(mlir::SymbolRefAttr kernelTaskType, bool includePrefetch = true);

bool isCacheHandlingOp(VPUIP::SwKernelOp swKernelOp);

mlir::SmallVector<mlir::Value> getDDRBuffers(mlir::ValueRange buffers);
bool hasInputsInDDR(VPUIP::SwKernelOp swKernelTask);

int64_t getSwKernelTilingAddressAlignment(VPUIP::SwKernelOp swkernelOp, VPU::ArchKind arch);
}  // namespace VPUIP
}  // namespace vpux
