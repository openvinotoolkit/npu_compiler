//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include "vpux/compiler/backend/VPUIP.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"

namespace vpux {
namespace hwtest {

// NumericsBench padding definition
// NB ref: NumericsBench/operators/op_utils/sliding_window.py#L47
// IERT Conv pads_begin/pads_end ref: kmb-plugin/src/vpux_compiler/src/core/tiling.cpp#L308
static constexpr auto PAD_NB_TOP = 0;
static constexpr auto PAD_NB_LEFT = 1;
static constexpr auto PAD_NB_BOTTOM = 2;
static constexpr auto PAD_NB_RIGHT = 3;

// NCETask padding definition
// IERT::ConvolutionOp -> VPUIP::NCEInvariant ref:
// kmb-plugin/src/vpux_compiler/src/conversion/passes/convert_to_nce_ops.cpp#L185
static constexpr auto PAD_NCETASK_LEFT = 0;
static constexpr auto PAD_NCETASK_RIGHT = 1;
static constexpr auto PAD_NCETASK_TOP = 2;
static constexpr auto PAD_NCETASK_BOTTOM = 3;

mlir::DenseElementsAttr generateWeights(llvm::ArrayRef<int64_t> wt_shape, mlir::Type dtype, mlir::MLIRContext* ctx,
                                        const char* weight_file_name);

std::size_t totalTensorSize(llvm::ArrayRef<std::int64_t> shape, mlir::Type elementtype);

std::vector<std::int64_t> convertNBPadtoNCETaskPad(const std::array<std::int64_t, 4>& nb_pad);

mlir::Type parseInputType(mlir::OpBuilder builder, const nb::InputLayer& input);
mlir::Type parseOutputType(mlir::OpBuilder builder, const nb::OutputLayer& output);
mlir::Type parseWeightsType(mlir::OpBuilder builder, const nb::WeightLayer& weight);

void buildCNNOp(mlir::OpBuilder& builder, llvm::StringRef mainFuncName, llvm::ArrayRef<mlir::Type> inputs,
                llvm::ArrayRef<mlir::Type> outputs);

void buildSimpleZMajorConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                           Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);
void buildContinuedConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                        Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);
void buildEltwiseAdd(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                     Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType);
void buildEltwiseMultWithDwConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type weightsType,
                                mlir::Type outputType);
void buildMaxpool(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type input0Type, mlir::Type outputType);
mlir::DenseElementsAttr splitWeightsOverC(mlir::DenseElementsAttr wt_vec, ArrayRef<int64_t> wt_shape, mlir::Type dtype,
                                          mlir::MLIRContext* ctx, size_t startC, size_t endC);
template <typename T>
mlir::DenseElementsAttr splitWeightsOverCLoop(mlir::DenseElementsAttr wt_vec, ArrayRef<int64_t> wt_shape,
                                              mlir::Type dtype, T elementType, mlir::MLIRContext* ctx, size_t start_C,
                                              size_t end_C);
mlir::DenseElementsAttr generateZeroPadForEltwiseMultWeights(ArrayRef<int64_t> wt_shape_padded, mlir::Type dtype,
                                                             mlir::MLIRContext* ctx);
mlir::MemRefType getMemRefType(mlir::OpBuilder builder, VPUIP::MemoryLocation memlocation, SmallVector<int64_t> shape,
                               mlir::Type type, SmallVector<mlir::AffineMap> affineMaps);

vpux::VPUIP::DeclareTensorOp createDeclareTensorOp(mlir::OpBuilder builder, VPUIP::MemoryLocation memlocation,
                                                   SmallVector<int64_t> shape, mlir::Type type,
                                                   SmallVector<mlir::AffineMap> affineMaps, int locale, size_t offset);

mlir::OpResult getTensorResult(VPUIP::DeclareTensorOp op);

mlir::OpResult getConstResult(vpux::Const::DeclareOp op);

vpux::VPUIP::DPUTaskOp createDPUTaskOp(mlir::OpBuilder builder, mlir::OpBuilder variantbuilder,
                                       llvm::SmallVector<int64_t> output_shape, std::vector<int64_t> padding_vec);

}  // namespace hwtest
}  // namespace vpux
