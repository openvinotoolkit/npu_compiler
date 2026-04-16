//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_fwd.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops_fwd.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>

#include <cstdint>

namespace vpux::config {
enum class ArchKind : uint64_t;
}  // namespace vpux::config
namespace vpux::VPU {
enum class ActShaveTaskType : uint64_t;
class LayerOpInterface;
}  // namespace vpux::VPU
namespace vpux::VPUIP {
struct KernelInfo;
}  // namespace vpux::VPUIP

namespace vpux {
namespace VPUIP {

// TODO: E60214, need support more sw kernel task type. Currently only enable MVN
const SmallVector<StringLiteral> SW_KERNELS_SUPPORTING_TILING = {"mvn1",
                                                                 "mvn1_sum",
                                                                 "mvn1_norm",
                                                                 "mvn6",
                                                                 "interpolate",
                                                                 "deformable_convolution",
                                                                 "activation_swish",
                                                                 "activation_gelu",
                                                                 "softmax",
                                                                 "log_softmax",
                                                                 "log_softmax_topk",
                                                                 "log_softmax_peak",
                                                                 "matmul",
                                                                 "activation_hswish",
                                                                 "activation_hardsigmoid",
                                                                 "convert",
                                                                 "activation_tanh",
                                                                 "topk",
                                                                 "gather",
                                                                 "gatherND",
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
                                                                 "eltwise_greater_equal",
                                                                 "eltwise_less",
                                                                 "eltwise_equal",
                                                                 "eltwise_not_equal",
                                                                 "eltwise_less_equal",
                                                                 "eltwise_select",
                                                                 "eltwise_and",
                                                                 "eltwise_bitwise_or",
                                                                 "eltwise_bitwise_and",
                                                                 "eltwise_bitwise_not",
                                                                 "eltwise_bitwise_xor",
                                                                 "activation_sin",
                                                                 "activation_cos",
                                                                 "activation_exp",
                                                                 "activation_abs",
                                                                 "activation_sign",
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
                                                                 "reduce_square",
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
                                                                 "lstm_dpu",
                                                                 "round_fp16",
                                                                 "accumulate",
                                                                 "populate_weight_table",
                                                                 "dequantize",
                                                                 "activation_mish",
                                                                 "dynamic_dequantize",
                                                                 "reverse",
                                                                 "reverse_sequence",
                                                                 "rms_norm",
                                                                 "rope",
                                                                 "rope_ilv",
                                                                 "rope_pairwise",
                                                                 "sdpa",
                                                                 "random_uniform",
                                                                 "grid_sample",
                                                                 "roll",
                                                                 "reorder",
                                                                 "activation_negative",
                                                                 "activation_softplus",
                                                                 "eltwise_logical_not",
                                                                 "cum_sum",
                                                                 "sdpa_extended",
                                                                 "flash_sdpa",
                                                                 "nv12_to_rgb",
                                                                 "i420_to_rgb",
                                                                 "activation_relu"};

const SmallVector<StringLiteral> SW_KERNELS_SUPPORTING_STRIDE = {
        "mvn1", "lstm_cell", "lstm_sequence", "lstm_dpu", "reorder", "sdpa_extended", "flash_sdpa"};

const SmallVector<std::string_view> SW_KERNELS_SUPPORTING_SHAVE_BALANCING = {"softmax",
                                                                             "eltwise_mul",
                                                                             "activation_sin",
                                                                             "activation_cos",
                                                                             "activation_swish",
                                                                             "activation_softplus",
                                                                             "activation_clamp",
                                                                             "convert",
                                                                             "eltwise_min",
                                                                             "eltwise_max",
                                                                             "round_fp16",
                                                                             "activation_exp",
                                                                             "eltwise_greater",
                                                                             "eltwise_greater_equal",
                                                                             "eltwise_less",
                                                                             "eltwise_equal",
                                                                             "eltwise_not_equal",
                                                                             "eltwise_less_equal",
                                                                             "eltwise_div",
                                                                             "prelu_fp16",
                                                                             "eltwise_logical_not",
                                                                             "activation_relu"};

const SmallVector<StringLiteral> SW_KERNELS_LAYOUT_AGNOSTIC = {
        "activation_swish", "activation_gelu",     "activation_hswish",   "activation_hardsigmoid",
        "activation_tanh",  "activation_sigmoid",  "activation_clamp",    "activation_sin",
        "activation_cos",   "activation_exp",      "activation_abs",      "activation_log",
        "activation_sqrt",  "hswish_fp16",         "round_fp16",          "eltwise_mul",
        "activation_mish",  "activation_softplus", "eltwise_logical_not", "activation_relu"};

// TODO: E#117136, use heuristic for tile dim
const SmallVector<StringLiteral> SW_ACTIVATION_KERNELS = {
        "activation_swish",   "activation_gelu",  "activation_hardsigmoid", "activation_tanh",
        "activation_sigmoid", "activation_clamp", "activation_abs",         "activation_floor",
        "activation_sin",     "activation_cos",   "activation_exp",         "hswish_fp16",
        "prelu_fp16",         "activation_mish",  "activation_softplus",    "activation_negative",
        "activation_relu"};

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
                                                                     "activation_mish",
                                                                     "eltwise_logical_not",
                                                                     "nv12_to_rgb",
                                                                     "i420_to_rgb",
                                                                     "activation_relu"};

const SmallVector<StringLiteral> SW_KERNELS_USE_DPU = {"lstm_dpu", "sdpa_extended", "flash_sdpa"};

constexpr StringLiteral vpuTaskTypeAttrName = "VPU.task_type";

SmallVector<mlir::Attribute> kernelArgsRange(VPUIP::SwKernelOp swKernelOp);

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, mlir::StringRef builtInFunctionName,
                                          const ArrayRef<mlir::Type> inputTypes, mlir::StringRef kernelEntryName,
                                          mlir::StringRef kernelSourceFileName, mlir::StringRef layerName,
                                          const vpux::Logger& log);

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, VPU::LayerOpInterface origOp,
                                          ArrayRef<mlir::Value> operands, ArrayRef<mlir::Value> results,
                                          const VPUIP::KernelInfo& kernelInfo, const Logger& log);

void createRuntimeKernelDefinition(mlir::ModuleOp module, const Logger& log, vpux::config::ArchKind arch);

void initSwKernel(vpux::VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs, mlir::ValueRange outputBuffs,
                  mlir::ArrayRef<mlir::Attribute> args, const vpux::Logger& log, VPUIP::SwKernelRun swKernelRunOp);

void initSwKernel(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRunOp, const vpux::Logger& log);

// This is used to convert map to a format used by Swkernels. For general purpose
// reverse permutation use mlir::inversePermutation.
SmallVector<int64_t> reversePermutation(mlir::AffineMap map);

SmallString getSwKernelEntryName(VPUIP::SwKernelOp swKernelOp);
mlir::ModuleOp getVPUSWModule(mlir::ModuleOp module, const Logger& log);
bool isActivationSwKernelOp(VPUIP::SwKernelOp swKernelOp);
bool isSwKernelTilingSupported(VPUIP::SwKernelOp swKernelOp);
bool isSwKernelUseDpu(VPUIP::SwKernelOp swKernelOp);
bool isDpuShaveKernelType(VPURT::TaskOp taskOp);
bool isStridedDataAccessSupported(VPUIP::SwKernelOp swKernelOp);

InputTiling backInferSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const SmallVector<vpux::TileInfo>& outputTiles,
                                       int tileId, Logger log);

SmallVector<mlir::Attribute> getSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                            ArrayRef<mlir::Attribute> origAttr,
                                                            const TilingInfo& inputTiling, const TileInfo& outTile,
                                                            Logger log);

SmallVector<int64_t> getPopulateWeightTableSwKernelEntries(VPUIP::SwKernelOp swKernelOp);
void updatePopulateWeightTableSwKernel(VPUIP::SwKernelOp swKernelOp, int64_t currOffset, Logger log);

int64_t computeReverseMemDim(mlir::Value tensorArg, int64_t dimIdx);
void getQuantParamsAttr(mlir::Value qValue, mlir::Type pType, mlir::ArrayAttr& paramsAttr, int64_t tileSize = 0,
                        int64_t tileOffset = 0);

SmallVector<vpux::NDTypeInterface> getSwKernelTiledTypes(VPUIP::SwKernelOp swKernelOp, Dim tileDim);

bool isCacheOpTaskType(std::optional<::mlir::SymbolRefAttr> kernelTaskType, bool includePrefetch = true);
bool isCacheOpTaskType(mlir::SymbolRefAttr kernelTaskType, bool includePrefetch = true);

bool isCacheHandlingOp(VPUIP::SwKernelOp swKernelOp);

bool isJitKernelOp(VPUIP::SwKernelOp swKernelOp);

mlir::SmallVector<mlir::Value> getDDRBuffers(mlir::ValueRange buffers);
bool hasInputsInDDR(VPUIP::SwKernelOp swKernelTask);

int64_t getSwKernelTilingAddressAlignment(VPUIP::SwKernelOp swkernelOp, config::ArchKind arch);
std::pair<bool, size_t> getSwKernelInstructionPrefetchConfig(config::ArchKind arch);

mlir::SymbolRefAttr createCacheHandlingFunction(mlir::MLIRContext* ctx, OpBuilderLogger& builderLog, Logger log,
                                                VPUIP::SwKernelOp origOp, mlir::StringRef functionName,
                                                VPU::ActShaveTaskType type);

}  // namespace VPUIP
}  // namespace vpux
