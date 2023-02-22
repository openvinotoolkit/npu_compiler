//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"

namespace vpux {
namespace hwtest {

// NumericsBench padding definition
// NB ref: NumericsBench/operators/op_utils/sliding_window.py#L47
// IERT Conv pads_begin/pads_end ref: vpux-plugin/src/vpux_compiler/src/core/tiling.cpp#L308
static constexpr auto PAD_NB_TOP = 0;
static constexpr auto PAD_NB_LEFT = 1;
static constexpr auto PAD_NB_BOTTOM = 2;
static constexpr auto PAD_NB_RIGHT = 3;

// NCETask padding definition
// IERT::ConvolutionOp -> VPUIP::NCEInvariant ref:
// vpux-plugin/src/vpux_compiler/src/conversion/passes/convert_to_nce_ops.cpp#L185
static constexpr auto PAD_NCETASK_LEFT = 0;
static constexpr auto PAD_NCETASK_RIGHT = 1;
static constexpr auto PAD_NCETASK_TOP = 2;
static constexpr auto PAD_NCETASK_BOTTOM = 3;

mlir::DenseElementsAttr generateWeights(llvm::ArrayRef<int64_t> wt_shape, mlir::Type dtype, mlir::MLIRContext* ctx,
                                        const char* weight_file_name);

std::size_t totalTensorSize(llvm::ArrayRef<int64_t> shape, mlir::Type elementtype);

std::vector<int64_t> convertNBPadtoNCETaskPad(const std::array<int64_t, 4>& nb_pad);

mlir::Type parseInputType(mlir::OpBuilder builder, const nb::InputLayer& input);
mlir::Type parseOutputType(mlir::OpBuilder builder, const nb::OutputLayer& output);
mlir::Type parseWeightsType(mlir::OpBuilder builder, const nb::WeightLayer& weight);

void buildCNNOp(mlir::OpBuilder& builder, llvm::StringRef mainFuncName, llvm::ArrayRef<mlir::Type> inputs,
                llvm::ArrayRef<mlir::Type> outputs);

void buildDMA(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder, Logger& log,
              mlir::Type inputType, mlir::Type outputType);
void buildSimpleZMajorConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                           Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);
void buildContinuedConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                        Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);
void buildDoubleConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                     Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);
void buildSparseZMajorConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                           Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);
void buildEltwiseAdd(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                     Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);
void buildEltwiseMultWithDwConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                mlir::Type outputType);
void buildMaxPool(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type input0Type, mlir::Type outputType);
void buildAvgpoolWithDwConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                            Logger& log, mlir::Type inputType, mlir::Type outputType);
void buildAvgpool(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type outputType);
void buildDifferentClustersDPUTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                   mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                   mlir::Type outputType);
void buildMultiClustersDPUTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                               mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                               mlir::Type outputType);
void buildDWConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                 Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);
void buildActShave(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                   Logger& log, const SmallVector<mlir::Type>& inputType, mlir::Type outputType);
void buildReadAfterWriteDPUDMATest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                   mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                   mlir::Type outputType);
void buildReadAfterWriteDMADPUTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                   mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                   mlir::Type outputType);
void buildReadAfterWriteACTDMATest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                   mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type outputType);
void buildReadAfterWriteDMAACTTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                   mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type outputType);
void buildReadAfterWriteDPUACTTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                   mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                   mlir::Type outputType);
void buildReadAfterWriteACTDPUTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                   mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                   mlir::Type outputType);
void buildSETableTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                      Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);

// TODO: remove buildRaceConditionDMATest and buildRaceConditionDPUTest once buildRaceConditionTest is able to cover
// all testcases [Track number: E#31468]
void buildRaceConditionDMATest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                               mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type outputType);
void buildRaceConditionDPUTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                               mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                               mlir::Type outputType);
void buildRaceConditionTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                            Logger& log, mlir::Type inputType, mlir::Type outputType);

void buildRaceConditionDPUDMATest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                  mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                  mlir::Type outputType);
void buildRaceConditionDPUACTTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                  mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                  mlir::Type outputType);
void buildRaceConditionDPUDMAACTTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                     mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                     mlir::Type outputType);
void buildDualChannelDMATest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                             Logger& log, mlir::Type inputType, mlir::Type outputType);
mlir::DenseElementsAttr splitWeightsOverC(mlir::DenseElementsAttr wt_vec, ArrayRef<int64_t> wt_shape, mlir::Type dtype,
                                          mlir::MLIRContext* ctx, size_t startC, size_t endC);
template <typename T>
mlir::DenseElementsAttr splitWeightsOverCLoop(mlir::DenseElementsAttr wt_vec, ArrayRef<int64_t> wt_shape,
                                              mlir::Type dtype, T elementType, mlir::MLIRContext* ctx, size_t start_C,
                                              size_t end_C);

mlir::MemRefType getMemRefType(VPURT::BufferSection section, ArrayRef<int64_t> shape, mlir::Type elemType,
                               DimsOrder order);
mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ArrayRef<int64_t> shape,
                               mlir::Type elemType, DimsOrder order);
mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ShapeRef shape, mlir::Type elemType,
                               DimsOrder order);
mlir::MemRefType getMemRefType(VPURT::BufferSection section, ArrayRef<int64_t> shape, mlir::Type elemType,
                               DimsOrder order, StridesRef strides);
mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ArrayRef<int64_t> shape,
                               mlir::Type elemType, DimsOrder order, StridesRef strides);
mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ArrayRef<int64_t> shape,
                               mlir::Type elemType, DimsOrder order, StridesRef strides,
                               VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr);

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, VPURT::BufferSection section,
                                                   ArrayRef<int64_t> shape, mlir::Type elemType, DimsOrder order,
                                                   int64_t locale, size_t offset);
vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, mlir::Type type,
                                                   VPURT::BufferSection section, ArrayRef<int64_t> locale,
                                                   size_t offset);
vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, VPURT::BufferSection section,
                                                   ShapeRef shape, mlir::Type elemType, DimsOrder order, int64_t locale,
                                                   size_t offset);
vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder builder, VPURT::BufferSection section,
                                                   ArrayRef<int64_t> shape, mlir::Type elemType, DimsOrder order,
                                                   StridesRef strides, int64_t locale, size_t offset);
vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, mlir::MemRefType type,
                                                   VPURT::BufferSection section, int64_t locale, size_t offset);
vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, mlir::Type type,
                                                   VPURT::BufferSection section, ArrayRef<int64_t> locale,
                                                   size_t offset);
vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder builder, VPURT::BufferSection section,
                                                   ArrayRef<int64_t> shape, mlir::Type elemType, DimsOrder order,
                                                   StridesRef strides, int64_t locale, size_t offset,
                                                   VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr);

mlir::OpResult getTensorResult(VPURT::DeclareBufferOp op);

mlir::OpResult getConstResult(vpux::Const::DeclareOp op);

vpux::VPUIP::DPUTaskOp createDPUTaskOp(mlir::OpBuilder builder, mlir::OpBuilder variantbuilder,
                                       ArrayRef<int64_t> outputShape, ArrayRef<int64_t> inputShape,
                                       const std::vector<int64_t>& paddingVec, VPU::MPEMode mpeMode);

vpux::DimsOrder oduPermutationToLayout(const MVCNN::Permutation oduPermutation);
vpux::Dim getInnermostDim(const vpux::DimsOrder& order);

VPU::PaddingAttr getMulticlusteringPaddings(mlir::MLIRContext* ctx, const int64_t cluster, const int64_t numClusters,
                                            nb::SegmentationType segmentationType, VPU::PaddingAttr globalPadding);

}  // namespace hwtest
}  // namespace vpux
