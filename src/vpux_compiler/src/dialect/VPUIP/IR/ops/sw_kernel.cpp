//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/bounded_buffer.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/asm.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/bit.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <algorithm>

#define REGION_YOLO_MAX_MASK_SIZE 9   // max mask size for region yolo op
#define PROPOSAL_MAX_RATIO 3          // max ratio size for proposal op
#define PROPOSAL_MAX_SCALE 6          // max scale size for proposal op
#define MAX_AXES_DIMS 4               // max axes size for reduce ops
#define SPACETOBATCH_MAX_INPUT_DIM 5  // max input size for spacetobatch op
#define BATCHTOSPACE_MAX_INPUT_DIM 5  // max input size for batchtospace op
#define T_HF8 1
#define T_BF8 2

using namespace vpux;
using namespace mlir;

namespace {

// permute int array attribute in the physical order
static SmallVector<int64_t> permuteIntArrayAttr(DimsOrder inOrder, mlir::ArrayAttr arrayAttr) {
    const auto origPerm = inOrder.toPermutation();
    const auto origArray = parseIntArrayAttr<int64_t>(arrayAttr);
    SmallVector<int64_t> permArray(arrayAttr.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = dstInd;
        permArray[revSrcInd] = origArray[revDstInd];
    }
    return permArray;
}

// pad the int array attribute with 0 to 4 to match StridedSlice begins ends and strides param
static SmallVector<int64_t> padIntArrayAttr(mlir::ArrayAttr arrayAttr) {
    const auto origArray = parseIntArrayAttr<int64_t>(arrayAttr);
    SmallVector<int64_t> padArray(4);
    for (const auto srcInd : irange(arrayAttr.size())) {
        padArray[srcInd] = origArray[srcInd];
    }
    return padArray;
}

static SmallVector<int64_t> getAxesArrayRevertAndOrderAware(mlir::Value tensorArg, mlir::ArrayAttr arrayAttr) {
    const auto axes = parseIntArrayAttr<int64_t>(arrayAttr);
    SmallVector<int64_t> revertedAxesArray(MAX_NUM_DIMS, 0);
    for (const auto srcInd : irange(arrayAttr.size())) {
        revertedAxesArray[srcInd] = VPUIP::computeReverseMemDim(tensorArg, axes[srcInd]);
    }
    return revertedAxesArray;
}

// Build a bit-mask to indicate present inputs/outputs
// (for SW kernels with optional IOs)
mlir::ArrayAttr optionalIoAttr(mlir::Operation* op) {
    int32_t mask = 0;

    llvm::TypeSwitch<mlir::Operation*, void>(op)
            .Case<VPU::MVN6Op>([&](VPU::MVN6Op mvn) {
                mask |= 1 << 0;  // main input
                mask |= (mvn.getScale() ? 1 : 0) << 1;
                mask |= (mvn.getBias() ? 1 : 0) << 2;
                mask |= 1 << 3;  // output
            })
            .Case<VPU::SDPAOp>([&](VPU::SDPAOp sdpa) {
                mask |= 1 << 0;                               // InputQ
                mask |= 1 << 1;                               // InputK
                mask |= 1 << 2;                               // InputV
                mask |= (sdpa.getInputMask() ? 1 : 0) << 3;   // InputMask
                mask |= (sdpa.getInputScale() ? 1 : 0) << 4;  // InputScale
                mask |= (sdpa.getInputBias() ? 1 : 0) << 5;   // InputBias
                mask |= 1 << 6;                               // InputDataStorage
                mask |= 1 << 7;                               // output
            })
            .Case<VPU::SDPAExtendedOp>([&](VPU::SDPAExtendedOp sdpaExtended) {
                mask |= 1 << 0;                                       // InputQ
                mask |= 1 << 1;                                       // InputK
                mask |= 1 << 2;                                       // InputV
                mask |= (sdpaExtended.getInputMask() ? 1 : 0) << 3;   // InputMask
                mask |= (sdpaExtended.getInputScale() ? 1 : 0) << 4;  // InputScale
                mask |= 1 << 5;                                       // InputDataStorage
                mask |= (sdpaExtended.getDpuStorage() ? 1 : 0) << 6;  // InputDpuStorage
                mask |= 1 << 7;                                       // output
            })
            .Default([](mlir::Operation* op) {
                VPUX_THROW("Bit-mask for '{0}' not implemented", op->getName());
            });

    llvm::SmallVector<int64_t> params;
    params.push_back(INT64_MIN);  // marker after all in/out MemRefs
    params.push_back(mask);
    return getIntArrayAttr(op->getContext(), std::move(params));
}

// isSupportedElemType checks whether strided_slice kernel supports element type of given value.
// 1. strided_slice kernel supports i64 attributes
// 2. strided_slice kernel supports i32 values
// 3. isAttribute denotes if the value is read from an attribute or from a value
// For example:
// if isAttribute is true for slice begins, the kernel reads begins from stridedSlice.getBegins()
// if isAttribute is false for slice ends, the kernel reads ends from stridedSlice.getEndsAttr()
bool isSupportedElemType(const bool isAttribute, const mlir::Value value) {
    if (isAttribute) {
        return true;
    }
    const auto ndType = mlir::cast<NDTypeInterface>(value.getType());
    return ndType.getElementType().isInteger(32);
}
}  // namespace

namespace vpux {
namespace VPUIP {

void vpux::VPUIP::SwKernelOp::print(mlir::OpAsmPrinter& p) {
    p.printOptionalAttrDict(
            getOperation()->getAttrs(),
            /*elidedAttrs=*/{getOperandSegmentSizesAttrName().strref(), getKernelFunctionAttrName().strref(),
                             getTileIndexAttrName().strref(), getListIndexAttrName().strref(),
                             getInputStridesAttrName().strref(), getOutputStridesAttrName().strref()});
    p << ' ';
    p.printAttributeWithoutType(getKernelFunctionAttr());

    auto& opBody = getBody();

    if (!opBody.empty()) {
        auto* entry = &opBody.front();

        unsigned opIdx = 0;
        printGroupOfOperands(p, entry, "inputs", getInputs(), opIdx);
        if (!getDynamicInputShapes().empty()) {
            p << ' ' << "dynamicInputShapes";
            p << "(";
            llvm::interleaveComma(getDynamicInputShapes(), p, [&](const auto& value) {
                p << value << " : " << value.getType();
            });
            p << ")";
        }
        printGroupOfOperands(p, entry, "outputs", getOutputBuffs(), opIdx);
        if (!getDynamicOutputShapeBuffs().empty()) {
            p << ' ' << "dynamicOutputShapes";
            p << "(";
            llvm::interleaveComma(getDynamicOutputShapeBuffs(), p, [&](const auto& value) {
                p << value << " : " << value.getType();
            });
            p << ")";
        }
    }

    auto profData = getProfilingData();
    if (profData) {
        p << ' ' << "profiling_data";
        p << "(";
        p << profData;
        p << ' ' << ":";
        p << ' ';
        p << ArrayRef<mlir::Type>(profData.getType());
        p << ")";
    }

    auto opInputStrides = getInputStrides();
    if (opInputStrides) {
        p << ' ' << "inputStrides";
        p << "(";
        p << opInputStrides;
        p << ")";
    }

    auto opOutputStrides = getOutputStrides();
    if (opOutputStrides) {
        p << ' ' << "outputStrides";
        p << "(";
        p << opOutputStrides;
        p << ")";
    }

    if (getTileIndex().has_value()) {
        p << ' ' << "on";
        p << ' ' << "tile";
        p << ' ';
        p.printAttributeWithoutType(getTileIndexAttr());
        if (getListIndex().has_value()) {
            p << ' ' << "list";
            p << ' ';
            p.printAttributeWithoutType(getListIndexAttr());
        }
    }

    p.printOptionalArrowTypeList(getResultTypes());
    p.printRegion(opBody, false);
}

mlir::ParseResult vpux::VPUIP::SwKernelOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    SmallVector<mlir::OpAsmParser::Argument> blockArgs;
    SmallVector<mlir::Type> blockTypes;

    if (parser.parseOptionalAttrDict(result.attributes)) {
        return mlir::failure();
    }

    mlir::SymbolRefAttr kernelFunctionAttr;
    if (parser.parseAttribute(kernelFunctionAttr, parser.getBuilder().getType<mlir::NoneType>(), "kernelFunction",
                              result.attributes)) {
        return mlir::failure();
    }

    int32_t inCount = 0;
    if (mlir::failed(parseGroupOfOperands(parser, result, blockArgs, blockTypes, "inputs", inCount))) {
        return mlir::failure();
    }

    int32_t inDynamicShapesCount = 0;
    if (mlir::failed(parseTensorDims(parser, result, "dynamicInputShapes", inDynamicShapesCount))) {
        return mlir::failure();
    }

    int32_t outCount = 0;
    if (mlir::failed(parseGroupOfOperands(parser, result, blockArgs, blockTypes, "outputs", outCount))) {
        return mlir::failure();
    }

    int32_t outDynamicShapesCount = 0;
    if (mlir::failed(parseTensorDims(parser, result, "dynamicOutputShapes", outDynamicShapesCount))) {
        return mlir::failure();
    }

    SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> profiling_dataOperands;
    mlir::SmallVector<mlir::Type, 1> profiling_dataTypes;
    mlir::OptionalParseResult parseResult;

    if (succeeded(parser.parseOptionalKeyword("profiling_data"))) {
        if (parser.parseLParen()) {
            return mlir::failure();
        }

        mlir::OpAsmParser::UnresolvedOperand operand;
        parseResult = parser.parseOptionalOperand(operand);
        if (parseResult.has_value()) {
            if (failed(*parseResult)) {
                return mlir::failure();
            }
            profiling_dataOperands.push_back(operand);
        }

        if (parser.parseColon()) {
            return mlir::failure();
        }

        mlir::Type optionalType;
        parseResult = parser.parseOptionalType(optionalType);
        if (parseResult.has_value()) {
            if (failed(*parseResult)) {
                return mlir::failure();
            }
            profiling_dataTypes.push_back(optionalType);
        }

        if (parser.parseRParen()) {
            return mlir::failure();
        }

        auto profiling_dataOperandsLoc = parser.getCurrentLocation();
        if (parser.resolveOperands(profiling_dataOperands, profiling_dataTypes, profiling_dataOperandsLoc,
                                   result.operands)) {
            return mlir::failure();
        }
    }

    if (succeeded(parser.parseOptionalKeyword("inputStrides"))) {
        if (parser.parseLParen()) {
            return mlir::failure();
        }

        mlir::ArrayAttr inputStridesAttr;
        parseResult = parser.parseOptionalAttribute(inputStridesAttr);
        if (!parseResult.has_value() || failed(*parseResult)) {
            return mlir::failure();
        }

        if (parser.parseRParen()) {
            return mlir::failure();
        }
        result.attributes.set("inputStrides", inputStridesAttr);
    }

    if (succeeded(parser.parseOptionalKeyword("outputStrides"))) {
        if (parser.parseLParen()) {
            return mlir::failure();
        }

        mlir::ArrayAttr outputStridesAttr;
        parseResult = parser.parseOptionalAttribute(outputStridesAttr);
        if (!parseResult.has_value() || failed(*parseResult)) {
            return mlir::failure();
        }

        if (parser.parseRParen()) {
            return mlir::failure();
        }
        result.attributes.set("outputStrides", outputStridesAttr);
    }

    if (succeeded(parser.parseOptionalKeyword("on"))) {
        if (parser.parseKeyword("tile")) {
            return mlir::failure();
        }

        mlir::IntegerAttr tileIndexAttr;
        parseResult = parser.parseOptionalAttribute(tileIndexAttr, parser.getBuilder().getIntegerType(64), "tileIndex",
                                                    result.attributes);
        if (parseResult.has_value() && failed(*parseResult)) {
            return mlir::failure();
        }

        if (succeeded(parser.parseOptionalKeyword("list"))) {
            mlir::IntegerAttr listIndexAttr;
            parseResult = parser.parseOptionalAttribute(listIndexAttr, parser.getBuilder().getIntegerType(64),
                                                        "listIndex", result.attributes);
            if (parseResult.has_value() && failed(*parseResult)) {
                return mlir::failure();
            }
        }
    }

    // Add derived `operandSegmentSizes` attribute based on parsed operands.
    auto operandSegmentSizes =
            parser.getBuilder().getDenseI32ArrayAttr({inCount, inDynamicShapesCount, outCount, outDynamicShapesCount,
                                                      static_cast<int32_t>(profiling_dataOperands.size())});
    result.addAttribute("operandSegmentSizes", operandSegmentSizes);

    SmallVector<mlir::Type> resultTypes;
    if (parser.parseOptionalArrowTypeList(resultTypes)) {
        return mlir::failure();
    }
    result.addTypes(resultTypes);

    // Parse region.
    auto* body = result.addRegion();
    if (parser.parseRegion(*body, blockArgs)) {
        return mlir::failure();
    }

    return mlir::success();
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex) {
    mlir::Value profilingOutput = nullptr;

    build(builder, opState, inputs, results, profilingOutput, kernelFunction, tileIndex);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex,
                       mlir::ArrayAttr outputStrides) {
    mlir::Value profilingOutput = nullptr;

    build(builder, opState, results.getTypes(), /*dynamicOutputShapes types*/ TypeRange{},
          /*profiling_output_type*/ nullptr, kernelFunction, inputs, /*dynamicInputShapes*/ ValueRange{},
          /*dynamicInputShapesMap*/ nullptr, results, /*dynamicOutputShapes*/ ValueRange{},
          /*dynamicOutputShapesMap*/ nullptr, profilingOutput, tileIndex, /* listIndex */ nullptr,
          /* inputStrides */ nullptr, outputStrides,
          /*profilingMetadata*/ nullptr);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex,
                       mlir::ArrayAttr inputStrides, mlir::ArrayAttr outputStrides) {
    mlir::Value profilingOutput = nullptr;

    build(builder, opState, results.getTypes(), /*dynamicOutputShapes types*/ TypeRange{},
          /*profiling_output_type*/ nullptr, kernelFunction, inputs, /*dynamicInputShapes*/ ValueRange{},
          /*dynamicInputShapesMap*/ nullptr, results, /*dynamicOutputShapes*/ ValueRange{},
          /*dynamicOutputShapesMap*/ nullptr, profilingOutput, tileIndex, /* listIndex */ nullptr, inputStrides,
          outputStrides,
          /*profilingMetadata*/ nullptr);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::Value profilingOutput, mlir::SymbolRefAttr kernelFunction,
                       mlir::IntegerAttr tileIndex) {
    build(builder, opState, results.getTypes(), /*dynamicOutputShapes types*/ TypeRange{},
          (profilingOutput ? profilingOutput.getType() : nullptr), kernelFunction, inputs,
          /*dynamicInputShapes*/ ValueRange{}, /*dynamicInputShapesMap*/ nullptr, results,
          /*dynamicOutputShapes*/ ValueRange{}, /*dynamicOutputShapesMap*/ nullptr, profilingOutput, tileIndex,
          /* listIndex */ nullptr, /* inputStrides */ nullptr, /*outputStrides*/ nullptr,
          /*profilingMetadata*/ nullptr);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::Value profilingOutput, mlir::SymbolRefAttr kernelFunction,
                       mlir::IntegerAttr tileIndex, mlir::ArrayAttr inputStrides, mlir::ArrayAttr outputStrides) {
    build(builder, opState, results.getTypes(), /*dynamicOutputShapes types*/ TypeRange{},
          (profilingOutput ? profilingOutput.getType() : nullptr), kernelFunction, inputs,
          /*dynamicInputShapes*/ ValueRange{}, /*dynamicInputShapesMap*/ nullptr, results,
          /*dynamicOutputShapes*/ ValueRange{}, /*dynamicOutputShapesMap*/ nullptr, profilingOutput, tileIndex,
          /* listIndex */ nullptr, inputStrides, outputStrides, /*profilingMetadata*/ nullptr);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::ValueRange dynamicInputShapes,
                       llvm::ArrayRef<int32_t> dynamicInputShapesMap, mlir::ValueRange dynamicOutputShapes,
                       llvm::ArrayRef<int32_t> dynamicOutputShapesMap, mlir::SymbolRefAttr kernelFunction,
                       mlir::IntegerAttr tileIndex) {
    build(builder, opState, results.getTypes(), dynamicOutputShapes.getTypes(), /*profilingOutputType*/ nullptr,
          kernelFunction, inputs, dynamicInputShapes,
          DenseI32ArrayAttr::get(builder.getContext(), dynamicInputShapesMap), results, dynamicOutputShapes,
          DenseI32ArrayAttr::get(builder.getContext(), dynamicOutputShapesMap), /*profilingOutput*/ nullptr, tileIndex,
          /* listIndex */ nullptr, /* inputStrides */ nullptr, /*outputStrides*/ nullptr,
          /*profilingMetadata*/ nullptr);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::ValueRange dynamicInputShapes,
                       llvm::ArrayRef<int32_t> dynamicInputShapesMap, mlir::ValueRange dynamicOutputShapes,
                       llvm::ArrayRef<int32_t> dynamicOutputShapesMap, mlir::SymbolRefAttr kernelFunction,
                       mlir::IntegerAttr tileIndex, mlir::ArrayAttr outputStrides) {
    build(builder, opState, results.getTypes(), dynamicOutputShapes.getTypes(), /*profilingOutputType*/ nullptr,
          kernelFunction, inputs, dynamicInputShapes,
          DenseI32ArrayAttr::get(builder.getContext(), dynamicInputShapesMap), results, dynamicOutputShapes,
          DenseI32ArrayAttr::get(builder.getContext(), dynamicOutputShapesMap), /*profilingOutput*/ nullptr, tileIndex,
          /* listIndex */ nullptr, /* inputStrides */ nullptr, outputStrides, /*profilingMetadata*/ nullptr);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::ValueRange dynamicInputShapes,
                       llvm::ArrayRef<int32_t> dynamicInputShapesMap, mlir::ValueRange dynamicOutputShapes,
                       llvm::ArrayRef<int32_t> dynamicOutputShapesMap, mlir::Value profilingOutput,
                       mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex) {
    build(builder, opState, results.getTypes(), dynamicOutputShapes.getTypes(),
          (profilingOutput ? profilingOutput.getType() : nullptr), kernelFunction, inputs, dynamicInputShapes,
          DenseI32ArrayAttr::get(builder.getContext(), dynamicInputShapesMap), results, dynamicOutputShapes,
          DenseI32ArrayAttr::get(builder.getContext(), dynamicOutputShapesMap), profilingOutput, tileIndex,
          /* listIndex */ nullptr, /* inputStrides */ nullptr, /*outputStrides*/ nullptr,
          /*profilingMetadata*/ nullptr);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, SwKernelOp swOp,
                       mlir::Value profilingOutput) {
    opState.addOperands(swOp->getOperands());

    auto& props = opState.getOrAddProperties<SwKernelOp::Properties>();

    auto dict = mlir::DictionaryAttr::get(builder.getContext(), swOp->getAttrs());
    VPUX_THROW_UNLESS(SwKernelOp::setPropertiesFromAttr(props, dict, nullptr).succeeded(),
                      "Cannot initialize SwKernelOp::Properties from attribute '{0}'!", dict);

    // Compute value for resultSegmentSizes attribute and add it to the properties
    int32_t resultsCount = swOp.getResults().size();
    int32_t dynamicOutputShapesCount = swOp.getDynamicOutputShapes().size();
    const int32_t profilingOutputCount = 1;
    props.setResultSegmentSizes({resultsCount, dynamicOutputShapesCount, profilingOutputCount});

    opState.addRegion();

    opState.addTypes(swOp.getResults().getTypes());
    if (!swOp.getDynamicOutputShapes().empty()) {
        opState.addTypes(swOp.getDynamicOutputShapes().getTypes());
    }
    opState.addTypes(profilingOutput.getType());
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::ValueRange dynamicInputShapes,
                       llvm::ArrayRef<int32_t> dynamicInputShapesMap, mlir::ValueRange dynamicOutputShapes,
                       llvm::ArrayRef<int32_t> dynamicOutputShapesMap, mlir::Value profilingOutput,
                       mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex, mlir::ArrayAttr outputStrides) {
    build(builder, opState, results.getTypes(), dynamicOutputShapes.getTypes(),
          (profilingOutput ? profilingOutput.getType() : nullptr), kernelFunction, inputs, dynamicInputShapes,
          DenseI32ArrayAttr::get(builder.getContext(), dynamicInputShapesMap), results, dynamicOutputShapes,
          DenseI32ArrayAttr::get(builder.getContext(), dynamicOutputShapesMap), profilingOutput, tileIndex,
          /* listIndex */ nullptr, /* inputStrides */ nullptr, outputStrides, /*profilingMetadata*/ nullptr);
}

void SwKernelOp::build(mlir::OpBuilder& builder, mlir::OperationState& opState, mlir::ValueRange inputs,
                       mlir::ValueRange results, mlir::ValueRange dynamicInputShapes,
                       llvm::ArrayRef<int32_t> dynamicInputShapesMap, mlir::ValueRange dynamicOutputShapes,
                       llvm::ArrayRef<int32_t> dynamicOutputShapesMap, mlir::Value profilingOutput,
                       mlir::SymbolRefAttr kernelFunction, mlir::IntegerAttr tileIndex, mlir::ArrayAttr inputStrides,
                       mlir::ArrayAttr outputStrides) {
    build(builder, opState, results.getTypes(), dynamicOutputShapes.getTypes(),
          (profilingOutput ? profilingOutput.getType() : nullptr), kernelFunction, inputs, dynamicInputShapes,
          DenseI32ArrayAttr::get(builder.getContext(), dynamicInputShapesMap), results, dynamicOutputShapes,
          DenseI32ArrayAttr::get(builder.getContext(), dynamicOutputShapesMap), profilingOutput, tileIndex,
          /* listIndex */ nullptr, inputStrides, outputStrides, /*profilingMetadata*/ nullptr);
}

mlir::LogicalResult SwKernelOp::verify() {
    const auto op = getOperation();
    if (VPUIP::isCacheHandlingOp(*this)) {
        if (!op->getOperands().empty()) {
            return errorAt(op, "SW Kernel Cache Op should have no operands");
        }
        if (!op->getResults().empty()) {
            return errorAt(op, "SW Kernel Cache Op should have no results");
        }
        auto kernelFunc =
                op->getParentOfType<mlir::ModuleOp>().lookupSymbol<mlir::func::FuncOp>(getKernelFunctionAttr());
        if (kernelFunc.getFunctionType().getNumInputs() != 0) {
            return errorAt(op, "SW Kernel Cache Op func should have no inputs");
        }

        if (kernelFunc.getFunctionType().getNumResults() != 0) {
            return errorAt(op, "SW Kernel Cache Op func should have no results");
        }
        return mlir::success();
    }

    auto swKernelRuns = getBody().getOps<VPUIP::SwKernelRun>();
    auto swKernelRunNum = std::distance(swKernelRuns.begin(), swKernelRuns.end());
    VPUX_THROW_UNLESS(swKernelRunNum > 0, "Got wrong number of VPUIP::SwKernelRun {0}", swKernelRunNum);

    auto hasBoundedBuffers = [](mlir::Operation::operand_range operandRange) {
        return llvm::any_of(operandRange, [](Value val) {
            return isBoundedBufferType(val);
        });
    };
    if (hasBoundedBuffers(getInputs())) {
        if (!getDynamicInputShapes().empty() || getDynamicInputShapesMap().has_value()) {
            return errorAt(op, "SW Kernel has inputs of BoundedBufferType but dynamicInputShapes or "
                               "dynamicInputShapesMap provided. Ambiguous dynamicShape information");
        }
    } else {
        if (getDynamicInputShapesMap().has_value()) {
            // When tiling is applied to an operation, we do not duplicate the values of
            // the getDynamicInputShapesMap; instead, we retain them as they are.
            if (getInputs().size() != getDynamicInputShapesMap().value().size() * swKernelRunNum) {
                return errorAt(
                        op,
                        "SW Kernel has inconsistent inputs and dynamicInputShapesMap. Inputs size [{0}] does not "
                        "match dynamicInputShapesMap size [{1}]",
                        getInputs().size(), getDynamicInputShapesMap().value().size());
            }
            auto dynamicShapesCount =
                    checked_cast<size_t>(llvm::count_if(getDynamicInputShapesMap().value(), [](int32_t dim) {
                        return dim != ABSENT_DIMS_FLAG;
                    }));
            if (getDynamicInputShapes().size() != dynamicShapesCount) {
                return errorAt(op,
                               "SW Kernel dynamicInputShapes size doesn't match with number of positive elements in "
                               "dynamicInputShapesMap");
            }
            for (const auto& i : getDynamicInputShapesMap().value_or(llvm::ArrayRef<int32_t>())) {
                if (i >= checked_cast<std::remove_reference<decltype(i)>::type>(getDynamicInputShapes().size())) {
                    return errorAt(
                            op, "SW Kernel dynamicInputShapesMap contains values which are out of dynamicInputShapes "
                                "operand range");
                }
            }
        }
    }

    if (hasBoundedBuffers(getOutputBuffs())) {
        if (!getDynamicOutputShapeBuffs().empty() || getDynamicOutputShapesMap().has_value()) {
            return errorAt(op, "SW Kernel has outputs of BoundedBufferType but dynamicOutputShapes or "
                               "dynamicOutputShapesMap provided. Ambiguous dynamicShape information");
        }
    } else {
        if (getDynamicOutputShapesMap().has_value()) {
            // When tiling is applied to an operation, we do not duplicate the values of
            // the getDynamicOutputShapesMap; instead, we retain them as they are.
            if (getOutputBuffs().size() != getDynamicOutputShapesMap().value().size() * swKernelRunNum) {
                return errorAt(
                        op,
                        "SW Kernel has inconsistent outputs and dynamicOutputShapesMap. Outputs size [{0}] does not "
                        "match dynamicOutputShapesMap size [{1}]",
                        getOutputBuffs().size(), getDynamicOutputShapesMap().value().size());
            }
            auto dynamicShapesCount =
                    checked_cast<size_t>(llvm::count_if(getDynamicOutputShapesMap().value(), [](int32_t dim) {
                        return dim != ABSENT_DIMS_FLAG;
                    }));
            if (getDynamicOutputShapeBuffs().size() != dynamicShapesCount) {
                return errorAt(op,
                               "SW Kernel dynamicOutputShapes size doesn't match with number of positive elements in "
                               "dynamicOutputShapesMap");
            }
            for (const auto& i : getDynamicOutputShapesMap().value_or(llvm::ArrayRef<int32_t>())) {
                if (i >= checked_cast<std::remove_reference<decltype(i)>::type>(getDynamicOutputShapeBuffs().size())) {
                    return errorAt(
                            op, "SW Kernel dynamicOutputShapesMap contains values which are out of dynamicOutputShapes "
                                "operand range");
                }
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult SwKernelOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                 mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                 mlir::OpaqueProperties props, mlir::RegionRange /*regions*/,
                                                 mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::SwKernelOpAdaptor swKernelOp(operands, attrs, props);
    if (mlir::failed(swKernelOp.verify(loc))) {
        return mlir::failure();
    }

    for (auto out : swKernelOp.getOutputBuffs()) {
        inferredTypes.push_back(out.getType());
    }

    for (auto outDims : swKernelOp.getDynamicOutputShapeBuffs()) {
        inferredTypes.push_back(outDims.getType());
    }

    if (swKernelOp.getProfilingData() != nullptr) {
        inferredTypes.push_back(swKernelOp.getProfilingData().getType());
    }

    return mlir::success();
}

// A SW-Kernel implementation may be split into multiple files:
//  -  to reduce code-size/stalls
//  -  to reduce general impact when adding a network specific optimization
// Returns {KernelEntry, KernelSource.cpp}
template <typename VPUOp>
std::pair<SmallString, SmallString> getKernelImpl(VPUOp op) = delete;

std::pair<SmallString, SmallString> getKernelImpl(VPU::ConvertOp op) {
    const auto arch = config::getArch(op.getOperation());
    if (arch != config::ArchKind::NPU37XX) {
        const auto iType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getElementType();
        const auto oType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType()).getElementType();

        if (iType.isF16() && oType.isF32()) {
            // FP16 -> FP32
            return std::pair<SmallString, SmallString>{"convert_p00", "convert_p00.cpp"};
        }

        auto isInputMaxByte = iType.isInteger(2) || iType.isInteger(4) || iType.isInteger(8);
        if (oType.isF16() && isInputMaxByte) {
            // UI8/SI8/UI4/SI4/UI2/SI2 -> FP16
            return std::pair<SmallString, SmallString>{"convert_p01", "convert_p01.cpp"};
        }

        if (arch != config::ArchKind::NPU40XX) {
            auto isInputFp8 = mlir::isa<mlir::Float8E4M3FNType, Float8E5M2Type>(iType);
            auto isOutputFp8 = mlir::isa<mlir::Float8E4M3FNType, Float8E5M2Type>(oType);
            if ((isInputFp8 && oType.isF16()) || (isOutputFp8 && iType.isF16())) {
                // BF8/HF8 <-> FP16
                return std::pair<SmallString, SmallString>{"convert_p01", "convert_p01.cpp"};
            }
        }
    }
    return std::pair<SmallString, SmallString>{"convert", "convert.cpp"};
}

std::pair<SmallString, SmallString> getKernelImpl(VPU::MVNOp op) {
    const auto arch = config::getArch(op.getOperation());
    if ((arch != config::ArchKind::NPU37XX) && (!vpux::IE::hasDynamicTensors(op))) {
        const auto type = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
        if (type.getDimsOrder() == DimsOrder::NCHW) {
            const auto shape = type.getShape();
            const auto WH = shape[Dims4D::Act::H] * shape[Dims4D::Act::W];
            if ((WH > 32) && (WH % 32 == 0) && (WH <= 256)) {
                const auto isF16 = type.getElementType().isF16();
                if ((!op.getAcrossChannels()) && op.getNormalizeVariance() && isF16) {
                    // NCHW, f16, W*H multiple of 32, <=256
                    return std::pair<SmallString, SmallString>{"mvn1_p00", "mvn1_p00.cpp"};
                }
            }
        }
    }
    return std::pair<SmallString, SmallString>{"mvn1", "mvn1.cpp"};
}

std::pair<SmallString, SmallString> getKernelImpl(VPU::RMSOp op) {
    const auto arch = config::getArch(op.getOperation());
    if ((arch != config::ArchKind::NPU37XX) && (!vpux::IE::hasDynamicTensors(op))) {
        const auto type = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
        const auto isF16 = type.getElementType().isF16();
        if (isF16 && (type.getDimsOrder() == DimsOrder::NCHW)) {
            const auto shape = type.getShape();
            const auto W = shape[Dims4D::Act::W];
            if (W == 64) {
                return std::pair<SmallString, SmallString>{"rms_norm_p00", "rms_norm_p00.cpp"};
            }
        }
    }
    return std::pair<SmallString, SmallString>{"rms_norm", "rms_norm.cpp"};
}

#define CASE_REDUCE(_OP_, _STR1_, _CTX_)                                                                       \
    .Case<_OP_>([&](_OP_ reduce) {                                                                             \
        const auto keepDims = static_cast<int64_t>(reduce.getKeepDimsAttr() != nullptr);                       \
        auto axesValue = parseIntArrayAttr<int64_t>(reduce.getAxesValue());                                    \
        const auto axesValueSize = static_cast<int64_t>(axesValue.size());                                     \
        VPUX_THROW_UNLESS(axesValueSize <= MAX_AXES_DIMS, "Axes size {0} bigger than max axis size {1}",       \
                          axesValueSize, MAX_AXES_DIMS);                                                       \
        for (auto& axis : axesValue) {                                                                         \
            axis = computeReverseMemDim(reduce.getInput(), axis);                                              \
        }                                                                                                      \
        const auto keepDimsAttr = getIntAttr(_CTX_, keepDims);                                                 \
        const auto axesValueAttr = getIntArrayAttr(_CTX_, axesValue);                                          \
        const auto axesValueSizeAttr = getIntAttr(_CTX_, axesValueSize);                                       \
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{keepDimsAttr, axesValueSizeAttr, axesValueAttr}, \
                                 {_STR1_}};                                                                    \
    })

VPUIP::KernelInfo SwKernelOp::getKernelInfo(mlir::Operation* origOp) {
    mlir::MLIRContext* ctx = origOp->getContext();

    return llvm::TypeSwitch<mlir::Operation*, VPUIP::KernelInfo>(origOp)
            .Case<VPU::GenericSwLayerOp>([&](VPU::GenericSwLayerOp genericSwLayerOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {genericSwLayerOp.getCallee().getLeafReference().getValue()},
                                         {"jit_generated"}};
            })
            .Case<VPU::ExternalKernelOp>([&](VPU::ExternalKernelOp externalKernelOp) {
                auto attrDict = externalKernelOp.getAttrDict();
                auto attrVec = SmallVector<mlir::Attribute>{};
                for (auto namedAttr : attrDict) {
                    attrVec.push_back(namedAttr.getValue());
                }
                return VPUIP::KernelInfo{attrVec, {externalKernelOp.getUniqueId().str()}};
            })
            .Case<VPU::ExpOp>([&](VPU::ExpOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_exp"}};
            })
            .Case<VPU::GatherOp>([&](VPU::GatherOp gather) {
                const auto axisParam = computeReverseMemDim(gather.getInput(), gather.getAxisValue().value());
                const auto axisParamAttr = getIntAttr(gather.getContext(), axisParam);
                const auto indicesShape = getShape(gather.getIndices());
                const auto indicesRankParamAttr =
                        getIntAttr(gather.getContext(), gather.getIndicesRank().value_or(indicesShape.size()));
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{axisParamAttr, gather.getBatchDimsAttr(), indicesRankParamAttr},
                        {"gather"}};
            })
            .Case<VPU::GatherElementsOp>([&](VPU::GatherElementsOp gatherElements) {
                const auto axisParam = computeReverseMemDim(gatherElements.getInput(), gatherElements.getAxis());
                const auto axisParamAttr = getIntAttr(gatherElements.getContext(), axisParam);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr}, {"gather_elements"}};
            })
            .Case<VPU::GatherNDOp>([&](VPU::GatherNDOp gatherND) {
                const auto storeOrigShapeInfoAttr =
                        packOriginalShapeAttrForGatherNDSwOp(gatherND.getOriginalShape().value_or(nullptr), ctx);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{gatherND.getBatchDimsAttr(), storeOrigShapeInfoAttr},
                        {"gatherND"}};
            })
            .Case<VPU::GatherTreeOp>([&](VPU::GatherTreeOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"gather_tree"}};
            })
            .Case<VPU::ScatterUpdateOp>([&](VPU::ScatterUpdateOp scatterUpdate) {
                const auto axisParam =
                        computeReverseMemDim(scatterUpdate.getInput(), scatterUpdate.getAxisValue().value());
                const auto axisParamAttr = getIntAttr(scatterUpdate.getContext(), axisParam);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr}, {"scatter_update"}};
            })
            .Case<VPU::ScatterElementsUpdateOp>([&](VPU::ScatterElementsUpdateOp scatterElementsUpdate) {
                const auto axisParam =
                        computeReverseMemDim(scatterElementsUpdate.getInput(), scatterElementsUpdate.getAxis());
                const auto axisParamAttr = getIntAttr(scatterElementsUpdate.getContext(), axisParam);

                const auto reduction = static_cast<int64_t>(scatterElementsUpdate.getReductionAttr().getValue());
                const auto reductionAttr = getIntAttr(ctx, reduction);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr, reductionAttr,
                                                                      scatterElementsUpdate.getUseInitValAttr()},
                                         {"scatter_elements_update"}};
            })
            .Case<VPU::TanhOp>([&](VPU::TanhOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_tanh"}};
            })
            .Case<VPU::HSwishOp>([&](VPU::HSwishOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_hswish"}};
            })
            .Case<VPU::HSigmoidOp>([&](VPU::HSigmoidOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_hsigmoid"}};
            })
            .Case<VPU::SigmoidOp>([&](VPU::SigmoidOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_sigmoid"}};
            })
            .Case<VPU::HardSigmoidOp>([&](VPU::HardSigmoidOp hardsigmoid) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{hardsigmoid.getAlphaValueAttr(), hardsigmoid.getBetaValueAttr()},
                        {"activation_hardsigmoid"}};
            })
            .Case<VPU::GridSampleOp>([&](VPU::GridSampleOp gridSample) {
                const auto alignCorners = static_cast<int64_t>(gridSample.getAlignCornersAttr() != nullptr);
                const auto alignCornersAttr = getIntAttr(ctx, alignCorners);

                int64_t mode = 0;
                if (gridSample.getModeAttr() != nullptr) {
                    mode = static_cast<int64_t>(gridSample.getModeAttr().getValue());
                }
                const auto modeIntAttr = getIntAttr(ctx, mode);

                int64_t paddingMode = 0;
                if (gridSample.getPaddingModeAttr() != nullptr) {
                    paddingMode = static_cast<int64_t>(gridSample.getPaddingModeAttr().getValue());
                }
                const auto paddingModeAttr = getIntAttr(ctx, paddingMode);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{alignCornersAttr, modeIntAttr, paddingModeAttr},
                                         {"grid_sample"}};
            })
            .Case<VPU::SoftMaxOp>([&](VPU::SoftMaxOp softmax) {
                // input tensor, to transform axis
                const auto axisParam = computeReverseMemDim(softmax.getInput(), softmax.getAxisInd());
                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                const int64_t padSize = 0;
                auto padSizeAttr = getIntAttr(ctx, padSize);
                if (softmax.getPadSize().has_value()) {
                    padSizeAttr = softmax.getPadSizeAttr();
                }
                const auto iType = mlir::cast<vpux::NDTypeInterface>(softmax.getInput().getType());
                if (iType.getElementType().isF32()) {
                    return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr, padSizeAttr}, {"softmaxFp32"}};
                } else {
                    return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr, padSizeAttr}, {"softmax"}};
                }
            })
            .Case<VPU::LogSoftmaxOp>([&](VPU::LogSoftmaxOp logSoftmax) {
                const auto axisParam = computeReverseMemDim(logSoftmax.getInput(), logSoftmax.getAxisInd());
                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                const int64_t padSize = 0;
                auto padSizeAttr = getIntAttr(ctx, padSize);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr, padSizeAttr}, {"log_softmax"}};
            })
            .Case<VPU::LoopSelectOp>([&](VPU::LoopSelectOp loopSelect) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{loopSelect.getDoConcatAttr(), loopSelect.getAxisAttr(),
                                                     loopSelect.getStrideAttr()},
                        {"loop_select"}};
            })
            .Case<VPU::InterpolateOp>([&](VPU::InterpolateOp interpolate) {
                const auto mode = static_cast<int64_t>(interpolate.getAttr().getMode().getValue());
                const auto coordMode = static_cast<int64_t>(interpolate.getAttr().getCoordMode().getValue());
                const auto nearestMode = static_cast<int64_t>(interpolate.getAttr().getNearestMode().getValue());
                const auto antialias = static_cast<int64_t>(interpolate.getAttr().getAntialias().getValue());
                const auto inOrder = DimsOrder::fromValue(interpolate.getInput());
                const auto initialInputDim = interpolate.getInitialInputDimsAttr().value();
                const auto initialInputDimsParam = permuteIntArrayAttr(inOrder, initialInputDim);
                const auto initialOutputDim = interpolate.getInitialOutputDimsAttr().value();
                const auto initialOutputDimsParam = permuteIntArrayAttr(inOrder, initialOutputDim);
                const auto tileFPList = parseFPArrayAttr<double>(interpolate.getTileOffsetAttrAttr());
                const auto cubeCoeffParam = interpolate.getAttr().getCubeCoeff().getValueAsDouble();
                const auto axisParam = parseIntArrayAttr<int64_t>(interpolate.getAxesAttrAttr());
                // Check the scaling dim size since the swkernel only support scaling at most two dims
                SmallVector<int64_t> scalingAxis;
                for (auto axis : axisParam) {
                    if (initialInputDim[checked_cast<unsigned>(axis)] !=
                        initialOutputDim[checked_cast<unsigned>(axis)]) {
                        scalingAxis.push_back(axis);
                    }
                }

                if (scalingAxis.size() == 1) {
                    if (scalingAxis[0] + 1 < static_cast<int64_t>(initialOutputDim.size())) {
                        scalingAxis.push_back(scalingAxis[0] + 1);
                    } else {
                        scalingAxis.push_back(scalingAxis[0] - 1);
                    }
                }

                VPUX_THROW_WHEN(scalingAxis.size() != 2,
                                "Unsupported number of axis at '{0}'. Expected 2, but got {1}.", interpolate->getLoc(),
                                scalingAxis.size());

                std::sort(scalingAxis.begin(), scalingAxis.end());
                VPUX_THROW_WHEN(scalingAxis[0] + 1 != scalingAxis[1],
                                "Only contiguous axes are supported by the Interpolate kernel");

                VPUX_THROW_WHEN(interpolate.getAttr().getAntialias().getValue() == true,
                                "Antialiasing is not supported.");

                const auto initialInputOffset =
                        interpolate.getInitialInputOffsetAttr().has_value()
                                ? permuteIntArrayAttr(inOrder, interpolate.getInitialInputOffsetAttr().value())
                                : SmallVector<int64_t>(inOrder.numDims(), 0);
                const auto initialOutputOffset =
                        interpolate.getInitialOutputOffsetAttr().has_value()
                                ? permuteIntArrayAttr(inOrder, interpolate.getInitialOutputOffsetAttr().value())
                                : SmallVector<int64_t>(inOrder.numDims(), 0);

                const auto modeAttr = getIntAttr(ctx, mode);
                const auto coordModeAttr = getIntAttr(ctx, coordMode);
                const auto nearestModeAttr = getIntAttr(ctx, nearestMode);
                const auto antialiasAttr = getIntAttr(ctx, antialias);
                const auto tileAttr = getFPArrayAttr(ctx, tileFPList);
                const auto initialInputDimsParamAttr = getIntArrayAttr(ctx, initialInputDimsParam);
                const auto initialOutputDimsParamAttr = getIntArrayAttr(ctx, initialOutputDimsParam);
                const auto cubeCoeffParamAttr = getFPAttr(ctx, cubeCoeffParam);
                const auto axisParamAttr = getIntArrayAttr(ctx, scalingAxis);
                const auto initialInputOffsetAttr = getIntArrayAttr(ctx, initialInputOffset);
                const auto initialOutputOffsetAttr = getIntArrayAttr(ctx, initialOutputOffset);

                // INT64_MAX added as a delimiter to find where MemRefData fields are ended in case of variadic number
                // of inputs
                const auto delimiterAttr = getIntAttr(ctx, INT64_MAX);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{delimiterAttr, modeAttr, coordModeAttr, nearestModeAttr,
                                                     antialiasAttr, tileAttr, initialInputDimsParamAttr,
                                                     initialOutputDimsParamAttr, axisParamAttr, cubeCoeffParamAttr,
                                                     initialInputOffsetAttr, initialOutputOffsetAttr},
                        {"interpolate"}};
            })
            .Case<VPU::ScatterNDUpdateOp>([&](VPU::ScatterNDUpdateOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"scatter_nd_update"}};
            })
            .Case<VPU::EluOp>([&](VPU::EluOp elu) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{elu.getXAttr()}, {"activation_elu"}};
            })
            .Case<VPU::ClampOp>([&](VPU::ClampOp clamp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{clamp.getMinAttr(), clamp.getMaxAttr()},
                                         {"activation_clamp"}};
            })
            .Case<VPU::FullyConnectedOp>([&](VPU::FullyConnectedOp op) {
                const auto inOrder = DimsOrder::fromValue(op.getInput());
                VPUX_THROW_UNLESS(inOrder == DimsOrder::NC, "Layout not supported, got {0}, expected NC", inOrder);
                const auto delimiterAttr = getIntAttr(ctx, INT64_MAX);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{delimiterAttr}, {"fully_connected"}};
            })
            .Case<VPU::MatMulOp>([&](VPU::MatMulOp op) {
                const auto transposeA = static_cast<int64_t>(op.getTransposeAAttr() != nullptr);
                const auto transposeB = static_cast<int64_t>(op.getTransposeBAttr() != nullptr);
                const auto transposeAAttr = getIntAttr(ctx, transposeA);
                const auto transposeBAttr = getIntAttr(ctx, transposeB);
                VPUX_THROW_UNLESS(op.getTransposeA() == false, "MatMul sw kernel does not support transpose");
                VPUX_THROW_UNLESS(op.getTransposeB() == false, "MatMul sw kernel does not support transpose");
                const auto inOrder1 = DimsOrder::fromValue(op.getInput1());
                const auto inOrder2 = DimsOrder::fromValue(op.getInput2());
                VPUX_THROW_UNLESS((inOrder1 == DimsOrder::NCHW) || (inOrder1 == DimsOrder::CHW),
                                  "MatMul sw kernel layout not supported, got {0}, expected NCHW or CHW", inOrder1);
                VPUX_THROW_UNLESS((inOrder2 == DimsOrder::NCHW) || (inOrder2 == DimsOrder::CHW),
                                  "MatMul sw kernel layout not supported, got {0}, expected NCHW or CHW", inOrder2);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{transposeAAttr, transposeBAttr}, {"matmul"}};
            })
            .Case<VPU::SqrtOp>([&](VPU::SqrtOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_sqrt"}};
            })
            .Case<VPU::ErfOp>([&](VPU::ErfOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_erf"}};
            })
            .Case<VPU::CeilingOp>([&](VPU::CeilingOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_ceil"}};
            })
            .Case<VPU::DivideOp>([&](VPU::DivideOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_div"}};
            })
            .Case<VPU::MultiplyOp>([&](VPU::MultiplyOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_mul"}};
            })
            .Case<VPU::AddOp>([&](VPU::AddOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_add"}};
            })
            .Case<VPU::SubtractOp>([&](VPU::SubtractOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_sub"}};
            })
            .Case<VPU::MinimumOp>([&](VPU::MinimumOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_min"}};
            })
            .Case<VPU::MaximumOp>([&](VPU::MaximumOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_max"}};
            })
            .Case<VPU::PowerOp>([&](VPU::PowerOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_power"}};
            })
            .Case<VPU::EqualOp>([&](VPU::EqualOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_equal"}};
            })
            .Case<VPU::FloorModOp>([&](VPU::FloorModOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_floor_mod"}};
            })
            .Case<VPU::ModOp>([&](VPU::ModOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_mod"}};
            })
            .Case<VPU::GreaterOp>([&](VPU::GreaterOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_greater"}};
            })
            .Case<VPU::GreaterEqualOp>([&](VPU::GreaterEqualOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_greater_equal"}};
            })
            .Case<VPU::LessOp>([&](VPU::LessOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_less"}};
            })
            .Case<VPU::LessEqualOp>([&](VPU::LessEqualOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_less_equal"}};
            })
            .Case<VPU::LogicalOrOp>([&](VPU::LogicalOrOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_logical_or"}};
            })
            .Case<VPU::LogicalXorOp>([&](VPU::LogicalXorOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_logical_xor"}};
            })
            .Case<VPU::LogicalNotOp>([&](VPU::LogicalNotOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_logical_not"}};
            })
            .Case<VPU::AndOp>([&](VPU::AndOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_and"}};
            })
            .Case<VPU::BitwiseAndOp>([&](VPU::BitwiseAndOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_bitwise_and"}};
            })
            .Case<VPU::BitwiseOrOp>([&](VPU::BitwiseOrOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_bitwise_or"}};
            })
            .Case<VPU::BitwiseXorOp>([&](VPU::BitwiseXorOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_bitwise_xor"}};
            })
            .Case<VPU::BitwiseNotOp>([&](VPU::BitwiseNotOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_bitwise_not"}};
            })
            .Case<VPU::NotEqualOp>([&](VPU::NotEqualOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_not_equal"}};
            })
            .Case<VPU::MishOp>([&](VPU::MishOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_mish"}};
            })
            .Case<VPU::MVNOp>([&](VPU::MVNOp mvn) {
                const auto iType = mlir::cast<vpux::NDTypeInterface>(mvn.getInput().getType());
                const auto oType = mlir::cast<vpux::NDTypeInterface>(mvn.getOutput().getType());
                const auto iOrder = iType.getDimsOrder();
                const auto supported = {DimsOrder::NCHW, DimsOrder::NCWH, DimsOrder::NHWC, DimsOrder::NWHC};
                VPUX_THROW_UNLESS(llvm::any_of(supported,
                                               [iOrder](DimsOrder order) {
                                                   return order == iOrder;
                                               }),
                                  "Unsupported order {0}", iOrder);

                if (iOrder == DimsOrder::NCHW || iOrder == DimsOrder::NCWH) {
                    const auto compact = StrideReqs::compact(iType.getRank());
                    VPUX_THROW_UNLESS(compact.checkStrides(iType), "Only compact input supported, got {0}", iType);
                    VPUX_THROW_UNLESS(compact.checkStrides(oType), "Only compact output supported, got {0}", oType);
                }

                auto [kernelEntry, kernelSrc] = getKernelImpl(mvn);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{mvn.getAcrossChannelsAttr(),
                                                                      mvn.getNormalizeVarianceAttr(), mvn.getEpsAttr()},
                                         kernelEntry,
                                         kernelSrc,
                                         {"mvn1"}};
                ;
            })
            .Case<VPU::MVN1SumOp>([&](VPU::MVN1SumOp mvnSum) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{mvnSum.getAcrossChannelsAttr(), mvnSum.getNormalizeVarianceAttr()},
                        {"mvn1_sum"}};
            })
            .Case<VPU::MVN1MeanVarOp>([&](VPU::MVN1MeanVarOp op) {
                int64_t groupC = 1;
                if (op.getInternalReshape().has_value()) {
                    const auto internalReshape = parseIntArrayAttr<int64_t>(op.getInternalReshape().value());
                    const auto inputShape = getShape(op.getInputs()[0]).raw();
                    groupC = inputShape[1] / internalReshape[1];
                }
                const auto groupChAttr = getIntAttr(ctx, groupC);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{op.getOrigShapeAttr(), op.getAcrossChannelsAttr(),
                                                     op.getNormalizeVarianceAttr(), groupChAttr, op.getEpsAttr()},
                        {"mvn1_mean_var"}};
            })
            .Case<VPU::MVN1NormalizeOp>([&](VPU::MVN1NormalizeOp op) {
                uint64_t internalMode = op.getHighPrecisionNormalize() ? 1 : 0;
                const auto internalModeAttr = getIntAttr(ctx, internalMode);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.getAcrossChannelsAttr(),
                                                                      op.getNormalizeVarianceAttr(), internalModeAttr},
                                         {"mvn1_norm"}};
            })
            .Case<VPU::MVN6Op>([&](VPU::MVN6Op mvn) {
                mlir::MLIRContext* ctx = mvn.getContext();
                // Convert 'axes' to physical reversed (innermost first) equivalent
                const auto axesVal = parseIntArrayAttr<int64_t>(mvn.getAxes());
                SmallVector<int64_t> memAxes;
                for (const auto axis : axesVal) {
                    memAxes.push_back(computeReverseMemDim(mvn.getInput(), axis));
                }
                const auto numAxes = memAxes.size();
                const auto epsMode = static_cast<int64_t>(mvn.getEpsModeAttr().getValue());
                const auto memAxesAttr = getIntArrayAttr(ctx, memAxes);
                const auto numAxesAttr = getIntAttr(ctx, numAxes);
                const auto epsModeAttr = getIntAttr(ctx, epsMode);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{optionalIoAttr(mvn), mvn.getNormalizeVarianceAttr(), epsModeAttr,
                                                     mvn.getEpsAttr(), numAxesAttr, memAxesAttr},
                        {"mvn6"}};
            })
            .Case<VPU::MemPermuteOp>([&](VPU::MemPermuteOp permute) {
                auto memPermArr = reversePermutation(permute.getMemPerm());
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{getIntArrayAttr(ctx, memPermArr)}, {"reorder"}};
            })
            .Case<VPU::LRNOp>([&](VPU::LRNOp LRN) {
                const auto axesVal = parseIntArrayAttr<int64_t>(LRN.getAxes());

                SmallVector<int64_t> memAxes;
                for (const auto axis : axesVal) {
                    memAxes.push_back(computeReverseMemDim(LRN.getInput(), axis));
                }
                const auto memAxesAttr = getIntArrayAttr(ctx, memAxes);

                const auto numAxes = memAxes.size();
                const auto numAxesAttr = getIntAttr(ctx, numAxes);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{LRN.getAlphaAttr(), LRN.getBetaAttr(), LRN.getBiasAttr(),
                                                     LRN.getSizeAttr(), numAxesAttr, memAxesAttr},
                        {"lrn"}};
            })
            .Case<VPU::ConvertOp>([&](VPU::ConvertOp op) {
                auto [kernelEntry, kernelSrc] = getKernelImpl(op);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, kernelEntry, kernelSrc, {"convert"}};
            })
            .Case<VPU::GroupNormalizationOp>([&](VPU::GroupNormalizationOp groupNormOp) {
                const auto epsilon = static_cast<float>(groupNormOp.getEpsilonAttr().getValueAsDouble());
                const auto numGroups = static_cast<int64_t>(groupNormOp.getNumGroupsAttr().getInt());

                const auto epsilonAttr = getFPAttr(ctx, epsilon);
                const auto numGroupsAttr = getIntAttr(ctx, numGroups);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{epsilonAttr, numGroupsAttr},
                                         {"group_normalization"}};
            })
            .Case<VPU::RegionYoloOp>([&](VPU::RegionYoloOp regionYolo) {
                // Add maskSize as input params
                const auto maskSize = static_cast<int64_t>(regionYolo.getMaskAttr().getValue().size());
                const auto maskSizeParamAttr = getIntAttr(ctx, maskSize);
                // Supplement for maskAttrInput to meet array setSize(9) defined in kernel
                const auto maskVec = parseIntArrayAttr<int64_t>(regionYolo.getMaskAttr());
                assert(maskSize <= REGION_YOLO_MAX_MASK_SIZE);
                int64_t maskList[REGION_YOLO_MAX_MASK_SIZE];
                for (int i = 0; i < REGION_YOLO_MAX_MASK_SIZE; i++) {
                    if (i >= maskSize) {
                        maskList[i] = 0;
                    } else {
                        maskList[i] = maskVec[i];
                    }
                }
                const auto maskArrayRef = ArrayRef<int64_t>(maskList, REGION_YOLO_MAX_MASK_SIZE);
                const auto maskAttrInput = getIntArrayAttr(ctx, maskArrayRef);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{regionYolo.getCoordsAttr(), regionYolo.getClassesAttr(),
                                                     regionYolo.getNumRegionsAttr(), regionYolo.getDoSoftmaxAttr(),
                                                     maskSizeParamAttr, maskAttrInput, regionYolo.getAxisAttr(),
                                                     regionYolo.getEndAxisAttr()},
                        {"region_yolo"}};
            })
            .Case<VPU::TopKOp>([&](VPU::TopKOp topk) {
                const auto axisParam = computeReverseMemDim(topk.getInput(), topk.getAxis());
                const auto mode = static_cast<int64_t>(topk.getModeAttr().getValue());
                const auto sortMode = static_cast<int64_t>(topk.getSortAttr().getValue());
                const auto kValue = static_cast<int64_t>(topk.getKValue().value());

                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                const auto modeIntAttr = getIntAttr(ctx, mode);
                const auto sortModeAttr = getIntAttr(ctx, sortMode);
                const auto kValueAttr = getIntAttr(ctx, kValue);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{axisParamAttr, modeIntAttr, sortModeAttr, kValueAttr},
                        {"topk"}};
            })
            .Case<VPU::ExtractImagePatchesOp>([&](VPU::ExtractImagePatchesOp op) {
                auto sizes = parseIntArrayAttr<int64_t>(op.getSizesAttr());
                auto strides = parseIntArrayAttr<int64_t>(op.getStridesAttr());
                auto rates = parseIntArrayAttr<int64_t>(op.getRatesAttr());
                const auto autoPad = static_cast<int32_t>(op.getAutoPadAttr().getValue());

                const auto iType = mlir::cast<vpux::NDTypeInterface>(op.getData().getType());
                const auto iOrder = iType.getDimsOrder();
                const auto supported = {DimsOrder::NCHW};
                VPUX_THROW_UNLESS(llvm::any_of(supported,
                                               [iOrder](DimsOrder order) {
                                                   return order == iOrder;
                                               }),
                                  "Unsupported order {0}", iOrder);

                const auto kernelSizeAttr = getIntArrayAttr(ctx, sizes);
                const auto stridesAttr = getIntArrayAttr(ctx, strides);
                const auto ratesAttr = getIntArrayAttr(ctx, rates);
                const auto autoPadAttr = getIntAttr(ctx, autoPad);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{kernelSizeAttr, stridesAttr, ratesAttr, autoPadAttr},
                        {"extract_image_patches"}};
            })
            .Case<VPU::PReluOp>([&](VPU::PReluOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"prelu_fp16"}};
            })
            .Case<VPU::PadOp>([&](VPU::PadOp pad) {
                mlir::MLIRContext* ctx = origOp->getContext();
                const auto inOrder = DimsOrder::fromValue(pad.getInput());
                const auto padBegin = permuteIntArrayAttr(inOrder, pad.getPadsBeginAttr().value());
                const auto padEnd = permuteIntArrayAttr(inOrder, pad.getPadsEndAttr().value());
                const auto padMode = static_cast<int64_t>(pad.getModeAttr().getValue());

                const auto padBeginAttr = getIntArrayAttr(ctx, padBegin);
                const auto padEndAttr = getIntArrayAttr(ctx, padEnd);
                const auto padModeAttr = getIntAttr(ctx, padMode);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{padBeginAttr, padEndAttr, pad.getPadValueAttrAttr(), padModeAttr},
                        {"pad"}};
            })
            .Case<VPU::AvgPoolOp>([&](VPU::AvgPoolOp op) {
                constexpr size_t MAX_ATTR_SZ = 3;  // base on filter description
                auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
                auto strides = parseIntArrayAttr<int64_t>(op.getStridesAttr());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBeginAttr());
                auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEndAttr());
                const auto excludePads = static_cast<int64_t>(op.getExcludePadsAttr() != nullptr);
                // In order to support any format, pad with identity up to the max attribute size (3).
                // Filter definition have a strict dependency between input tensor rank, params size and layout, but for
                // hw usage we can cover and situation when this are not respected.
                for (auto i = kernelSize.size(); i < MAX_ATTR_SZ; i++) {
                    kernelSize.insert(kernelSize.begin(), 1);
                    strides.insert(strides.begin(), 1);
                    padsBegin.insert(padsBegin.begin(), 0);
                    padsEnd.insert(padsEnd.begin(), 0);
                }
                const auto kernelSizeAttr = getIntArrayAttr(ctx, kernelSize);
                const auto stridesAttr = getIntArrayAttr(ctx, strides);
                const auto padsBeginAttr = getIntArrayAttr(ctx, padsBegin);
                const auto padsEndAttr = getIntArrayAttr(ctx, padsEnd);
                const auto excludePadsAttr = getIntAttr(ctx, excludePads);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{kernelSizeAttr, stridesAttr, padsBeginAttr,
                                                                      padsEndAttr, excludePadsAttr},
                                         {"avg_pool"}};
            })
            .Case<VPU::AvgPool16Op>([&](VPU::AvgPool16Op op) {
                constexpr size_t MAX_ATTR_SZ = 3;  // base on filter description
                auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
                auto strides = parseIntArrayAttr<int64_t>(op.getStridesAttr());
                auto dilations = parseIntArrayAttr<int64_t>(op.getDilationsAttr());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBeginAttr());
                auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEndAttr());
                const auto excludePads = static_cast<int64_t>(op.getExcludePadsAttr() != nullptr);
                // In order to support any format, pad with identity up to the max attribute size (3).
                // Filter definition have a strict dependency between input tensor rank, params size and layout, but for
                // hw usage we can cover and situation when this are not respected.
                for (auto i = kernelSize.size(); i < MAX_ATTR_SZ; i++) {
                    kernelSize.insert(kernelSize.begin(), 1);
                    strides.insert(strides.begin(), 1);
                    dilations.insert(dilations.begin(), 1);
                    padsBegin.insert(padsBegin.begin(), 0);
                    padsEnd.insert(padsEnd.begin(), 0);
                }
                const auto kernelSizeAttr = getIntArrayAttr(ctx, kernelSize);
                const auto stridesAttr = getIntArrayAttr(ctx, strides);
                const auto dilationsAttr = getIntArrayAttr(ctx, dilations);
                const auto padsBeginAttr = getIntArrayAttr(ctx, padsBegin);
                const auto padsEndAttr = getIntArrayAttr(ctx, padsEnd);
                const auto excludePadsAttr = getIntAttr(ctx, excludePads);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{kernelSizeAttr, stridesAttr, dilationsAttr,
                                                                      padsBeginAttr, padsEndAttr, excludePadsAttr},
                                         {"avg_pool16"}};
            })
            .Case<VPU::AdaptiveAvgPoolOp>([&](VPU::AdaptiveAvgPoolOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"adaptive_pool"}};
            })
            .Case<VPU::AdaptiveMaxPoolOp>([&](VPU::AdaptiveMaxPoolOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"adaptive_max_pool"}};
            })
            .Case<VPU::FakeQuantizeOp>([&](VPU::FakeQuantizeOp op) {
                const auto iType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
                const auto oType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
                VPUX_THROW_UNLESS(iType.getElementType().isF16() && oType.getElementType().isF16(),
                                  "Only supports FP16 in/out");

                VPUX_THROW_UNLESS(op.getLowFpType().has_value() || op.getLevels().has_value(),
                                  "LowFpType or levels attribute should has value.");

                const auto levelsAttr =
                        op.getLevels().has_value() ? getIntAttr(ctx, op.getLevels().value()) : getIntAttr(ctx, -1);

                int64_t defaultLowFpTypeValue = -1;
                auto lowFpType = op.getLowFpType();
                if (lowFpType.has_value()) {
                    if (mlir::isa<mlir::Float8E4M3FNType>(*lowFpType)) {  // HF8
                        defaultLowFpTypeValue = T_HF8;
                    }
                    if (mlir::isa<mlir::Float8E5M2Type>(*lowFpType)) {  // BF8
                        defaultLowFpTypeValue = T_BF8;
                    }
                }
                auto LowFpTypeAttr = getIntAttr(ctx, defaultLowFpTypeValue);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{levelsAttr, LowFpTypeAttr}, {"fake_quantize"}};
            })
            .Case<VPU::FakeConvertOp>([&](VPU::FakeConvertOp op) {
                const auto dstTypeAttr = op.getDstTypeAttr();
                const auto delimiterAttr = getIntAttr(ctx, INT64_MAX);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{delimiterAttr, dstTypeAttr}, {"fake_convert"}};
            })
            .Case<VPU::QuantizeOp>([&](VPU::QuantizeOp op) {
                const auto iType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
                mlir::ArrayAttr paramsAttr;
                getQuantParamsAttr(op.getOutput(), iType.getElementType(), paramsAttr);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{paramsAttr}, {"quantize"}};
            })
            .Case<VPU::DequantizeOp>([&](VPU::DequantizeOp op) {
                const auto oType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
                mlir::ArrayAttr paramsAttr;
                getQuantParamsAttr(op.getInput(), oType.getElementType(), paramsAttr);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{paramsAttr}, {"dequantize"}};
            })
            .Case<VPU::DynamicQuantizeOp>([&](VPU::DynamicQuantizeOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"dynamic_quantize"}};
            })
            .Case<VPU::DynamicDequantizeOp>([&](VPU::DynamicDequantizeOp) {
                const auto delimiterAttr = getIntAttr(ctx, INT64_MAX);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{delimiterAttr}, {"dynamic_dequantize"}};
            })
            .Case<VPU::DepthToSpaceOp>([&](VPU::DepthToSpaceOp depth_to_space) {
                const auto mode = static_cast<int64_t>(depth_to_space.getModeAttr().getValue());
                const auto modeAttr = getIntAttr(ctx, mode);
                SmallVector<mlir::Attribute> paramAttr = {depth_to_space.getBlockSizeAttr(), modeAttr};
                if (depth_to_space.getPaddedChannels().has_value()) {
                    auto paddedIC = depth_to_space.getPaddedChannels().value().getInput();
                    auto paddedOC = depth_to_space.getPaddedChannels().value().getOutput();

                    if (paddedIC != nullptr) {
                        paramAttr.push_back(paddedIC);
                    }

                    if (paddedOC != nullptr) {
                        paramAttr.push_back(paddedOC);
                    }
                }
                return VPUIP::KernelInfo{paramAttr, {"depth_to_space"}};
            })
            .Case<VPU::SpaceToDepthOp>([&](VPU::SpaceToDepthOp space_to_depth) {
                const auto mode = static_cast<int64_t>(space_to_depth.getModeAttr().getValue());
                const auto modeAttr = getIntAttr(ctx, mode);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{space_to_depth.getBlockSizeAttr(), modeAttr},
                                         {"space_to_depth"}};
            })
            .Case<VPU::SpaceToBatch>([&](VPU::SpaceToBatch space_to_batch) {
                const auto blockShape = parseIntArrayAttr<int64_t>(space_to_batch.getBlockShapeValueAttr());
                const auto padsBegin = parseIntArrayAttr<int64_t>(space_to_batch.getPadsBeginValueAttr());
                const auto padsEnd = parseIntArrayAttr<int64_t>(space_to_batch.getPadsEndValueAttr());

                const auto inputDim = static_cast<int64_t>(space_to_batch.getBlockShapeValueAttr().getValue().size());

                int64_t blockShapeList[SPACETOBATCH_MAX_INPUT_DIM];
                int64_t padsBeginList[SPACETOBATCH_MAX_INPUT_DIM];
                int64_t padsEndList[SPACETOBATCH_MAX_INPUT_DIM];

                for (int i = 0; i < inputDim; i++) {
                    if (i >= inputDim) {
                        blockShapeList[i] = -1;
                        padsBeginList[i] = -1;
                        padsEndList[i] = -1;
                    } else {
                        blockShapeList[i] = blockShape[i];
                        padsBeginList[i] = padsBegin[i];
                        padsEndList[i] = padsEnd[i];
                    }
                }

                const auto blockShapeArryRef = ArrayRef<int64_t>(blockShapeList, SPACETOBATCH_MAX_INPUT_DIM);
                const auto padsBeginArryRef = ArrayRef<int64_t>(padsBeginList, SPACETOBATCH_MAX_INPUT_DIM);
                const auto padsEndArryRef = ArrayRef<int64_t>(padsEndList, SPACETOBATCH_MAX_INPUT_DIM);
                const auto blockShapeAttr = getIntArrayAttr(ctx, blockShapeArryRef);
                const auto padsBeginAttr = getIntArrayAttr(ctx, padsBeginArryRef);
                const auto padsEndAttr = getIntArrayAttr(ctx, padsEndArryRef);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{blockShapeAttr, padsBeginAttr, padsEndAttr},
                                         {"space_to_batch"}};
            })
            .Case<VPU::BatchToSpace>([&](VPU::BatchToSpace space_to_batch) {
                const auto blockShape = parseIntArrayAttr<int64_t>(space_to_batch.getBlockShapeValueAttr());
                const auto cropsBegin = parseIntArrayAttr<int64_t>(space_to_batch.getCropsBeginValueAttr());
                const auto cropsEnd = parseIntArrayAttr<int64_t>(space_to_batch.getCropsEndValueAttr());

                const auto inputDim = static_cast<int64_t>(space_to_batch.getBlockShapeValueAttr().getValue().size());

                std::array<int64_t, BATCHTOSPACE_MAX_INPUT_DIM> blockShapeList, cropsBeginList, cropsEndList;

                for (int i = 0; i < inputDim; i++) {
                    if (i >= inputDim) {
                        blockShapeList[i] = -1;
                        cropsBeginList[i] = -1;
                        cropsEndList[i] = -1;
                    } else {
                        blockShapeList[i] = blockShape[i];
                        cropsBeginList[i] = cropsBegin[i];
                        cropsEndList[i] = cropsEnd[i];
                    }
                }

                const auto blockShapeAttr = getIntArrayAttr(ctx, ArrayRef(blockShapeList));
                const auto cropsBeginAttr = getIntArrayAttr(ctx, ArrayRef(cropsBeginList));
                const auto cropsEndAttr = getIntArrayAttr(ctx, ArrayRef(cropsEndList));

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{blockShapeAttr, cropsBeginAttr, cropsEndAttr},
                                         {"batch_to_space"}};
            })
            .Case<VPU::SeluOp>([&](VPU::SeluOp selu) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{selu.getAlphaValueAttr(), selu.getLambdaValueAttr()},
                        {"activation_selu"}};
            })
            .Case<VPU::LeakyReluOp>([&](VPU::LeakyReluOp leakyRelu) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{leakyRelu.getNegativeSlopeAttr()},
                                         {"leaky_relu_fp16"}};
            })
            .Case<VPU::SwishOp>([&](VPU::SwishOp swish) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{swish.getBetaValueAttr()}, {"activation_swish"}};
            })
            .Case<VPU::ReLUOp>([&](VPU::ReLUOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_relu"}};
            })
            .Case<VPU::NegativeOp>([&](VPU::NegativeOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_negative"}};
            })
            .Case<VPU::StridedSliceOp>([&](VPU::StridedSliceOp stridedSlice) {
                mlir::MLIRContext* ctx = origOp->getContext();
                const auto emptyArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0, 0, 0});

                int64_t isBeginsConst = stridedSlice.getBeginsAttr().has_value() ? 1 : 0;
                VPUX_THROW_UNLESS(isSupportedElemType(isBeginsConst, stridedSlice.getBegins()),
                                  "strided_slice kernel supports only i32 begins.");
                const auto isBeginsConstAttr = getIntAttr(ctx, isBeginsConst);

                const auto stridedSliceBegins = stridedSlice.getBeginsAttr().has_value()
                                                        ? padIntArrayAttr(stridedSlice.getBeginsAttr().value())
                                                        : padIntArrayAttr(emptyArrayAttr);

                int64_t isEndsConst = stridedSlice.getEndsAttr().has_value() ? 1 : 0;
                VPUX_THROW_UNLESS(isSupportedElemType(isEndsConst, stridedSlice.getEnds()),
                                  "strided_slice kernel supports only i32 ends.");
                const auto isEndsConstAttr = getIntAttr(ctx, isEndsConst);
                const auto stridedSliceEnds = stridedSlice.getEndsAttr().has_value()
                                                      ? padIntArrayAttr(stridedSlice.getEndsAttr().value())
                                                      : padIntArrayAttr(emptyArrayAttr);

                int64_t isStridesConst = stridedSlice.getStridesAttr().has_value() ? 1 : 0;
                VPUX_THROW_UNLESS(isSupportedElemType(isStridesConst, stridedSlice.getStrides()),
                                  "strided_slice kernel supports only i32 strides.");
                const auto isStridesConstAttr = getIntAttr(ctx, isStridesConst);
                const auto stridedSliceStrides = stridedSlice.getStridesAttr().has_value()
                                                         ? padIntArrayAttr(stridedSlice.getStridesAttr().value())
                                                         : padIntArrayAttr(emptyArrayAttr);

                const auto stridedSliceBeginsAttr = getIntArrayAttr(ctx, stridedSliceBegins);
                const auto stridedSliceEndsAttr = getIntArrayAttr(ctx, stridedSliceEnds);
                const auto stridedSliceStridesAttr = getIntArrayAttr(ctx, stridedSliceStrides);

                // INT64_MAX added as a delimiter to find where MemRefData fields are ended in case of
                // variadic number of inputs for StridedSlice
                const auto delimiterAttr = getIntAttr(ctx, INT64_MAX);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{delimiterAttr, stridedSliceBeginsAttr, stridedSliceEndsAttr,
                                                     stridedSliceStridesAttr, isBeginsConstAttr, isEndsConstAttr,
                                                     isStridesConstAttr},
                        {"strided_slice"}};
            })
            .Case<VPU::ReverseSequenceOp>([&](VPU::ReverseSequenceOp ReverseSequence) {
                const auto batchAxis = computeReverseMemDim(ReverseSequence.getData(), ReverseSequence.getBatchAxis());
                const auto seqAxis = computeReverseMemDim(ReverseSequence.getData(), ReverseSequence.getSeqAxis());
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{getIntAttr(ctx, batchAxis), getIntAttr(ctx, seqAxis)},
                        {"reverse_sequence"}};
            })
            .Case<VPU::YuvToRgbOp>([&](VPU::YuvToRgbOp yuvToRgb) {
                mlir::MLIRContext* ctx = origOp->getContext();
                const auto inFmt = yuvToRgb.getInFmtAttr().getValue();
                const auto outFmt = static_cast<int64_t>(yuvToRgb.getOutFmtAttr().getValue()) - 2;
                const auto outFmtIntAttr = getIntAttr(ctx, outFmt);
                auto singlePlane = (yuvToRgb.getInput2() == nullptr);

                if (inFmt == vpux::IE::ColorFmt::NV12) {
                    if (singlePlane) {
                        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{outFmtIntAttr},
                                                 {"nv12_to_rgb_single_plane"}};
                    } else {
                        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{outFmtIntAttr}, {"nv12_to_rgb"}};
                    }
                } else {
                    if (singlePlane) {
                        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{outFmtIntAttr},
                                                 {"i420_to_rgb_single_plane"}};
                    } else {
                        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{outFmtIntAttr}, {"i420_to_rgb"}};
                    }
                }
            })
            .Case<VPU::RandomUniformOp>([&](VPU::RandomUniformOp rand) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{rand.getGlobalSeedAttr(), rand.getOpSeedAttr()},
                                         {"random_uniform"}};
            })
            .Case<VPU::ROIPoolingOp>([&](VPU::ROIPoolingOp roi) {
                const auto iType = mlir::cast<vpux::NDTypeInterface>(roi.getInput().getType());
                VPUX_THROW_UNLESS(iType.getRank() == 4, "Supporting only 4D input, got {0}", iType.getRank());

                const auto method = static_cast<int64_t>(roi.getMethodAttr().getValue());
                const auto methodAttr = getIntAttr(ctx, method);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{roi.getOutputSizeAttr(), roi.getSpatialScaleAttr(), methodAttr},
                        {"roi_pooling"}};
            })
            .Case<VPU::ROIAlignOp>([&](VPU::ROIAlignOp roiAlign) {
                const auto iType = mlir::cast<vpux::NDTypeInterface>(roiAlign.getInput().getType());
                VPUX_THROW_UNLESS(iType.getRank() == 4, "Supporting only 4D input, got {0}", iType.getRank());

                const auto pooledH = roiAlign.getPooledHAttr();
                const auto pooledW = roiAlign.getPooledWAttr();
                const auto samplingRatio = roiAlign.getSamplingRatioAttr();
                const auto spatialScale = roiAlign.getSpatialScaleAttr();

                const auto poolingMode = static_cast<int64_t>(roiAlign.getPoolingModeAttr().getValue());
                const auto poolingModeAttr = getIntAttr(ctx, poolingMode);

                const auto alignedMode = static_cast<int64_t>(roiAlign.getAlignedModeAttr().getValue());
                const auto alignedModeAttr = getIntAttr(ctx, alignedMode);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{pooledH, pooledW, samplingRatio, spatialScale,
                                                                      poolingModeAttr, alignedModeAttr},
                                         {"roi_align"}};
            })
            .Case<VPU::RollOp>([&](VPU::RollOp rollOp) {
                const auto axes = rollOp.getAxes();
                SmallVector<int64_t> axesValues;
                if (axes != nullptr) {
                    auto axesConst = axes.getDefiningOp<Const::DeclareOp>();
                    if (axesConst != nullptr) {
                        const auto axesContent = axesConst.getContent();
                        const auto axesVec = to_small_vector(axesContent.getValues<int64_t>());
                        axesValues.append(axesVec.begin(), axesVec.end());
                    }
                }
                const auto axesValAttr = getIntArrayAttr(ctx, axesValues);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axesValAttr}, {"roll"}};
            })
            .Case<VPU::OneHotOp>([&](VPU::OneHotOp oneHot) {
                int64_t axis = oneHot.getAxis();
                const auto shape = getShape(oneHot.getInput());
                auto nDims = checked_cast<int64_t>(shape.size());
                const int64_t actualAxis = (axis < 0) ? -axis - 1 : nDims - axis;

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{getIntAttr(ctx, actualAxis), oneHot.getDepthAttr(),
                                                     oneHot.getOnValueAttr(), oneHot.getOffValueAttr()},
                        {"onehot"}};
            })
            .Case<VPU::ReorgYoloOp>([&](VPU::ReorgYoloOp reorgYolo) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{reorgYolo.getStrideAttr()}, {"reorg_yolo"}};
            })
            .Case<VPU::ProposalOp>([&](VPU::ProposalOp proposal) {
                const auto baseSizeAttr = proposal.getProposalAttrsAttr().getBaseSize();
                const auto preNmsTopNAttr = proposal.getProposalAttrsAttr().getPreNmsTopN();
                const auto postNmsTopNAttr = proposal.getProposalAttrsAttr().getPostNmsTopN();
                const auto nmsThreshAttr = proposal.getProposalAttrsAttr().getNmsThresh();
                const auto featStrideAttr = proposal.getProposalAttrsAttr().getFeatStride();
                const auto minSizeAttr = proposal.getProposalAttrsAttr().getMinSize();

                const auto ratioSize = static_cast<int>(proposal.getProposalAttrsAttr().getRatio().size());
                const auto ratio = parseFPArrayAttr<double>(proposal.getProposalAttrsAttr().getRatio());
                float ratioList[PROPOSAL_MAX_RATIO];
                for (int i = 0; i < PROPOSAL_MAX_RATIO; i++) {
                    if (i >= ratioSize) {
                        ratioList[i] = -1;
                    } else {
                        ratioList[i] = checked_cast<float>(ratio[i]);
                    }
                }
                const auto ratioRef = ArrayRef<float>(ratioList, PROPOSAL_MAX_RATIO);
                const auto ratioAttr = getFPArrayAttr(ctx, ratioRef);

                const auto scaleSize = static_cast<int>(proposal.getProposalAttrsAttr().getScale().size());
                const auto scale = parseFPArrayAttr<double>(proposal.getProposalAttrsAttr().getScale());
                float scaleList[PROPOSAL_MAX_SCALE];
                for (int i = 0; i < PROPOSAL_MAX_SCALE; i++) {
                    if (i >= scaleSize) {
                        scaleList[i] = -1;
                    } else {
                        scaleList[i] = checked_cast<float>(scale[i]);
                    }
                }
                const auto scaleRef = ArrayRef<float>(scaleList, PROPOSAL_MAX_SCALE);
                const auto scaleAttr = getFPArrayAttr(ctx, scaleRef);

                const bool clipBeforeNms = proposal.getProposalAttrsAttr().getClipBeforeNms().getValue();
                const int64_t clipBeforeNmsInt = clipBeforeNms != 0;
                const auto clipBeforeNmsAttr = getIntAttr(ctx, clipBeforeNmsInt);

                const bool clipAfterNms = proposal.getProposalAttrsAttr().getClipAfterNms().getValue();
                const int64_t clipAfterNmsInt = clipAfterNms != 0;
                const auto clipAfterNmsAttr = getIntAttr(ctx, clipAfterNmsInt);

                const bool normalize = proposal.getProposalAttrsAttr().getNormalize().getValue();
                const int64_t normalizeInt = normalize != 0;
                const auto normalizeAttr = getIntAttr(ctx, normalizeInt);

                const auto boxSizeScaleAttr = proposal.getProposalAttrsAttr().getBoxSizeScale();
                const auto boxCoordinateScaleAttr = proposal.getProposalAttrsAttr().getBoxCoordinateScale();

                const auto framework = proposal.getProposalAttrsAttr().getFramework().getValue().str();
                int64_t frameworkInt = framework == "tensorflow";
                const auto frameworkAttr = getIntAttr(ctx, frameworkInt);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{baseSizeAttr, preNmsTopNAttr, postNmsTopNAttr, nmsThreshAttr,
                                                     featStrideAttr, minSizeAttr, ratioAttr, scaleAttr,
                                                     clipBeforeNmsAttr, clipAfterNmsAttr, normalizeAttr,
                                                     boxSizeScaleAttr, boxCoordinateScaleAttr, frameworkAttr},
                        {"proposal"}};
            })
            .Case<VPU::ReverseOp>([&](VPU::ReverseOp reverse) {
                int64_t mode = static_cast<int64_t>(reverse.getMode());
                const auto reverseModeAttr = getIntAttr(ctx, mode);

                auto sizeAxis = static_cast<int64_t>(reverse.getAxisValueAttr().getValue().size());
                auto sizeAxisAttr = getIntAttr(ctx, sizeAxis);

                const auto axesVal = parseIntArrayAttr<int64_t>(reverse.getAxisValue());
                const auto axesValAttr = getIntArrayAttr(ctx, axesVal);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{sizeAxisAttr, reverseModeAttr, axesValAttr},
                                         {"reverse"}};
            })
            .Case<VPU::PSROIPoolingOp>([&](VPU::PSROIPoolingOp psroi) {
                const auto iType = mlir::cast<vpux::NDTypeInterface>(psroi.getInput().getType());
                VPUX_THROW_UNLESS(iType.getRank() == 4, "Supporting only 4D input, got {0}", iType.getRank());
                VPUX_THROW_UNLESS(psroi.getMode() == IE::PSROIPoolingMode::AVERAGE,
                                  "Supporting only average psroi mode, got {0}", psroi.getMode());

                const auto spatialBinsX = static_cast<int64_t>(psroi.getSpatialBinsXAttr() != nullptr);
                const auto spatialBinsY = static_cast<int64_t>(psroi.getSpatialBinsYAttr() != nullptr);
                const auto mode = static_cast<int64_t>(psroi.getModeAttr() != nullptr);

                const auto spatialBinsXAttr = getIntAttr(ctx, spatialBinsX);
                const auto spatialBinsYAttr = getIntAttr(ctx, spatialBinsY);
                const auto modeAttr = getIntAttr(ctx, mode);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{psroi.getOutputDimAttr(), psroi.getSpatialScaleAttr(),
                                                     psroi.getGroupSizeAttr(), spatialBinsXAttr, spatialBinsYAttr,
                                                     modeAttr},
                        {"ps_roipooling"}};
            })
            .Case<VPU::NonMaxSuppressionOp>([&](VPU::NonMaxSuppressionOp nms) {
                const auto boxEncoding = static_cast<int64_t>(nms.getBoxEncodingAttr().getValue());
                const auto boxEncodingAttr = getIntAttr(ctx, boxEncoding);
                const auto sortResultDescending = static_cast<int64_t>(nms.getSortResultDescendingAttr() != nullptr);
                const auto sortResultDescendingAttr = getIntAttr(ctx, sortResultDescending);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{nms.getMaxOutputBoxesPerClassValueAttr(),
                                                     nms.getIouThresholdValueAttr(), nms.getScoreThresholdValueAttr(),
                                                     nms.getSoftNmsSigmaValueAttr(), boxEncodingAttr,
                                                     sortResultDescendingAttr},
                        {"nms"}};
            })
            .Case<VPU::CTCGreedyDecoderOp>([&](VPU::CTCGreedyDecoderOp op) {
                const auto mergeRepeated = static_cast<int64_t>(op.getMergeRepeatedAttr() != nullptr);
                const auto mergeRepeatedAttr = getIntAttr(ctx, mergeRepeated);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{mergeRepeatedAttr}, {"ctc_greedy_decoder"}};
            }) CASE_REDUCE(VPU::ReduceL1Op, "reduce_l1", ctx) CASE_REDUCE(VPU::ReduceSumOp, "reduce_sum", ctx)
                    CASE_REDUCE(VPU::ReduceMeanOp, "reduce_mean", ctx)
                            CASE_REDUCE(VPU::ReduceLogicalAndOp, "reduce_and", ctx)
                                    CASE_REDUCE(VPU::ReduceLogicalOrOp, "reduce_or", ctx)
                                            CASE_REDUCE(VPU::ReduceMaxOp, "reduce_max", ctx)
                                                    CASE_REDUCE(VPU::ReduceMinOp, "reduce_min", ctx)
                                                            CASE_REDUCE(VPU::ReduceProdOp, "reduce_prod", ctx)
                                                                    CASE_REDUCE(VPU::ReduceL2Op, "reduce_l2", ctx)
                                                                            CASE_REDUCE(VPU::ReduceMeanSquareOp,
                                                                                        "reduce_mean_square", ctx)
            .Case<VPU::SinOp>([&](VPU::SinOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_sin"}};
            })
            .Case<VPU::CosOp>([&](VPU::CosOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_cos"}};
            })
            .Case<VPU::SinhOp>([&](VPU::SinhOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_sinh"}};
            })
            .Case<VPU::CoshOp>([&](VPU::CoshOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_cosh"}};
            })
            .Case<VPU::FloorOp>([&](VPU::FloorOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_floor"}};
            })
            .Case<VPU::SignOp>([&](VPU::SignOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_sign"}};
            })
            .Case<VPU::TileOp>([&](VPU::TileOp tileOp) {
                auto repeats_size = static_cast<int64_t>(tileOp.getRepeatsValuesAttr().getValue().size());
                auto repeats_sizeAttr = getIntAttr(ctx, repeats_size);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{repeats_sizeAttr, tileOp.getRepeatsValuesAttr()},
                                         {"tile"}};
            })
            .Case<VPU::TanOp>([&](VPU::TanOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_tan"}};
            })
            .Case<VPU::AsinOp>([&](VPU::AsinOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_asin"}};
            })
            .Case<VPU::AcosOp>([&](VPU::AcosOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_acos"}};
            })
            .Case<VPU::AtanOp>([&](VPU::AtanOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_atan"}};
            })
            .Case<VPU::AsinhOp>([&](VPU::AsinhOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_asinh"}};
            })
            .Case<VPU::AcoshOp>([&](VPU::AcoshOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_acosh"}};
            })
            .Case<VPU::AtanhOp>([&](VPU::AtanhOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_atanh"}};
            })
            .Case<VPU::ExperimentalDetectronROIFeatureExtractorOp>(
                    [&](VPU::ExperimentalDetectronROIFeatureExtractorOp exp) {
                        // INT64_MAX added as a delimiter to find where MemRefData fields are ended in case of variadic
                        // number of inputs
                        const auto delimiterAttr = getIntAttr(ctx, INT64_MAX);
                        const auto outputSize = exp.getAttr().getOutputSize().getInt();
                        const auto samplingRatio = exp.getAttr().getSamplingRatio().getInt();
                        const auto pyramidScales = parseIntArrayAttr<int64_t>(exp.getAttr().getPyramidScales());
                        const auto aligned = static_cast<int64_t>(exp.getAttr().getAligned().getValue());

                        const auto alignedAttr = getIntAttr(ctx, aligned);
                        const auto outputSizeAttr = getIntAttr(ctx, outputSize);
                        const auto samplingRatioAttr = getIntAttr(ctx, samplingRatio);
                        const auto pyramidScalesAttr = getIntArrayAttr(ctx, pyramidScales);

                        return VPUIP::KernelInfo{
                                SmallVector<mlir::Attribute>{delimiterAttr, outputSizeAttr, samplingRatioAttr,
                                                             alignedAttr, pyramidScalesAttr},
                                {"experimental_detectron_roi_feature_extractor"}};
                    })
            .Case<VPU::DetectionOutputNormalizeOp>([&](VPU::DetectionOutputNormalizeOp op) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.getInputWidthAttr(), op.getInputHeightAttr()},
                                         {"detection_output_normalize"}};
            })
            .Case<VPU::DetectionOutputDecodeBoxesOp>([&](VPU::DetectionOutputDecodeBoxesOp op) {
                const auto codeType = [&]() -> int64_t {
                    enum CodeType : int64_t { CENTER_SIZE, CORNER, CORNER_SIZE };
                    switch (op.getCodeType()) {
                    case IE::DetectionOutputCodeType::CENTER_SIZE:
                        return CodeType::CENTER_SIZE;
                    case IE::DetectionOutputCodeType::CORNER:
                        return CodeType::CORNER;
                    case IE::DetectionOutputCodeType::CORNER_SIZE:
                        return CodeType::CORNER_SIZE;
                    }

                    const auto codeTypeString = stringifyDetectionOutputCodeType(op.getCodeType());
                    VPUX_THROW("Unsupported DetectionOutput codeType: {0}", codeTypeString);
                }();

                const auto clipBeforeNms = static_cast<int64_t>(op.getClipBeforeNms());
                const auto clipBeforeNmsAttr = getIntAttr(ctx, clipBeforeNms);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{getIntAttr(ctx, codeType), clipBeforeNmsAttr},
                                         {"detection_output_decode_boxes"}};
            })
            .Case<VPU::DetectionOutputSortOp>([&](VPU::DetectionOutputSortOp op) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{op.getConfidenceThresholdAttr(), op.getTopKAttr()},
                        {"detection_output_sort"}};
            })
            .Case<VPU::DetectionOutputNmsCaffeOp>([&](VPU::DetectionOutputNmsCaffeOp op) {
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{op.getNmsThresholdAttr(), op.getBackgroundIdAttr()},
                        {"detection_output_nms_caffe"}};
            })
            .Case<VPU::DetectionOutputCollectResultsOp>([&](VPU::DetectionOutputCollectResultsOp op) {
                const auto clipAfterNms = static_cast<int64_t>(op.getClipAfterNms());
                const auto clipAfterNmsAttr = getIntAttr(ctx, clipAfterNms);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{op.getKeepTopKAttr(), clipAfterNmsAttr},
                                         {"detection_output_collect_results"}};
            })
            .Case<VPU::PermuteQuantizeOp>([&](VPU::PermuteQuantizeOp op) {
                // permutation params convert to shv order
                auto memPermArr = reversePermutation(op.getMemPerm());
                //  kernel implementation allow optimized speed configuration to be specify
                enum PermuteQuantizeOptMode : int64_t {
                    // fields meaning: InOrder_OutOrder_NoChanels
                    PQ_NONE = 0,
                    PQ_NCHW_NHWC_C1 = 1,
                    PQ_NCHW_NHWC_C3 = 2,
                    PQ_NCHW_NHWC_C4 = 3,
                    PQ_NCHW_NHWC_C3EXP4 = 4,
                    PQ_NCHW_NHWC_C4EXP4 = 5,
                    PQ_NCHW_NHWC_C1EXP4 = 6
                };
                int64_t optMode = PermuteQuantizeOptMode::PQ_NONE;
                const auto iType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
                const auto oType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
                const auto inOrder = DimsOrder::fromValue(op.getInput());
                const auto outOrder = DimsOrder::fromValue(op.getOutput());
                if ((inOrder == DimsOrder::NCHW) && (outOrder == DimsOrder::NHWC) &&
                    (1 == iType.getShape()[Dims4D::Act::N])) {
                    if (1 == iType.getShape()[Dims4D::Act::C]) {
                        optMode = PermuteQuantizeOptMode::PQ_NCHW_NHWC_C1;
                    } else if (3 == iType.getShape()[Dims4D::Act::C]) {
                        optMode = PermuteQuantizeOptMode::PQ_NCHW_NHWC_C3;
                    } else if (4 == iType.getShape()[Dims4D::Act::C]) {
                        optMode = PermuteQuantizeOptMode::PQ_NCHW_NHWC_C4;
                    }
                }
                // Extract quantize scale and zero
                auto quantParams = mlir::cast<mlir::quant::UniformQuantizedType>(oType.getElementType());
                double scale = quantParams.getScale();
                int64_t zero = quantParams.getZeroPoint();

                // Custom kernels speed up for scenario of expand to 4.
                // Reduce kernel rump-up by enabling just higher resolutions that kernel parallelism allows.
                // Haven't implement trailing elements management. If resolution is not multiple of 16 will be managed
                // by default mode.
                auto szwh = oType.getShape()[Dims4D::Act::H] * oType.getShape()[Dims4D::Act::W];
                if ((PermuteQuantizeOptMode::PQ_NCHW_NHWC_C3 == optMode) && (4 == oType.getShape()[Dims4D::Act::C]) &&
                    (0 == (szwh % 16)) && (szwh >= 48)) {
                    optMode = PQ_NCHW_NHWC_C3EXP4;
                }
                if ((PermuteQuantizeOptMode::PQ_NCHW_NHWC_C4 == optMode) && (4 == oType.getShape()[Dims4D::Act::C]) &&
                    (0 == (szwh % 8)) && (szwh >= 48)) {
                    optMode = PQ_NCHW_NHWC_C4EXP4;
                }
                if ((PermuteQuantizeOptMode::PQ_NCHW_NHWC_C1 == optMode) && (4 == oType.getShape()[Dims4D::Act::C]) &&
                    (0 == (szwh % 16)) && (szwh >= 64)) {
                    optMode = PQ_NCHW_NHWC_C1EXP4;
                }
                if (iType.getElementType().isF32()) {
                    return VPUIP::KernelInfo{
                            SmallVector<mlir::Attribute>{getIntAttr(ctx, optMode), getFPAttr(ctx, scale),
                                                         getIntAttr(ctx, zero), getIntArrayAttr(ctx, memPermArr)},
                            {"permute_quantize_fp32"}};
                } else {
                    return VPUIP::KernelInfo{
                            SmallVector<mlir::Attribute>{getIntAttr(ctx, optMode), getFPAttr(ctx, scale),
                                                         getIntAttr(ctx, zero), getIntArrayAttr(ctx, memPermArr)},
                            {"permute_quantize"}};
                }
            })
            .Case<VPU::RoundOp>([&](VPU::RoundOp round) {
                const auto mode = static_cast<int64_t>(round.getModeAttr().getValue());
                const auto modeIntAttr = getIntAttr(ctx, mode);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{modeIntAttr}, {"round_fp16"}};
            })
            .Case<VPU::LogOp>([&](VPU::LogOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_log"}};
            })
            .Case<VPU::NormalizeL2Op>([&](VPU::NormalizeL2Op normalizeL2) {
                const auto epsMode = static_cast<int64_t>(normalizeL2.getEpsModeAttr().getValue());
                const auto epsModeAttr = getIntAttr(ctx, epsMode);
                auto axes = parseIntArrayAttr<int32_t>(normalizeL2.getAxesValueAttr());
                auto noAxes = axes.size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int32_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{normalizeL2.getEpsAttr(), epsModeAttr, noAxesAttr, axesAttr},
                        {"normalize_l2"}};
            })
            .Case<VPU::CumSumOp>([&](VPU::CumSumOp cumSum) {
                const auto axisParam = computeReverseMemDim(cumSum.getInput(), cumSum.getAxisValue().value());
                const auto exclusive = static_cast<int64_t>(cumSum.getExclusiveAttr() != nullptr);
                const auto reverse = static_cast<int64_t>(cumSum.getReverseAttr() != nullptr);

                const auto axisParamAttr = getIntAttr(ctx, axisParam);
                const auto reverseAttr = getIntAttr(ctx, reverse);
                const auto exclusiveAttr = getIntAttr(ctx, exclusive);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{axisParamAttr, exclusiveAttr, reverseAttr},
                                         {"cum_sum"}};
            })
            .Case<VPU::SelectOp>([&](VPU::SelectOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_select"}};
            })
            .Case<VPU::EmbeddingBagOffsetsSumOp>([&](VPU::EmbeddingBagOffsetsSumOp op) {
                const auto delimiterAttr = getIntAttr(ctx, INT64_MAX);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{delimiterAttr, op.getDefaultIndexValueAttr()},
                                         {"embedding_bag_offsets_sum"}};
            })
            .Case<VPU::SoftPlusOp>([&](VPU::SoftPlusOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_softplus"}};
            })
            .Case<VPU::EmbeddingBagPackedSumOp>([&](VPU::EmbeddingBagPackedSumOp) {
                const auto delimiterAttr = getIntAttr(ctx, INT64_MAX);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{delimiterAttr}, {"embedding_bag_packed_sum"}};
            })
            .Case<VPU::EyeOp>([&](VPU::EyeOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eye"}};
            })
            .Case<VPU::GRUSequenceOp>([&](VPU::GRUSequenceOp gru) {
                const auto mode = gru.getDirection();
                VPUX_THROW_UNLESS(
                        mode == IE::RNNSequenceDirection::FORWARD || mode == IE::RNNSequenceDirection::REVERSE,
                        "GRUSequence supports FORWARD and REVERSE");
                mlir::IntegerAttr directionModeAttr =
                        (mode == IE::RNNSequenceDirection::FORWARD) ? getIntAttr(ctx, 0) : getIntAttr(ctx, 1);
                const auto shouldLinearBeforeReset =
                        static_cast<int64_t>(gru.getShouldLinearBeforeResetAttr() != nullptr);
                const auto shouldLinearBeforeResetAttr = getIntAttr(ctx, shouldLinearBeforeReset);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{gru.getHiddenSizeAttr(), directionModeAttr, gru.getSeqLengthAttr(),
                                                     shouldLinearBeforeResetAttr, gru.getClipAttr()},
                        {"gru_sequence"}};
            })
            .Case<VPU::GRUSequenceFirstPartOp>([&](VPU::GRUSequenceFirstPartOp gru) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{gru.getHiddenSizeAttr(), gru.getSeqLengthAttr(),
                                                                      gru.getClipAttr()},
                                         {"gru_sequence_first_part"}};
            })
            .Case<VPU::GRUSequenceLastPartOp>([&](VPU::GRUSequenceLastPartOp gru) {
                const auto mode = static_cast<int64_t>(gru.getDirection() == IE::RNNSequenceDirection::FORWARD ? 0 : 1);
                const auto directionModeAttr = getIntAttr(ctx, mode);
                const auto shouldLinearBeforeReset =
                        static_cast<int64_t>(gru.getShouldLinearBeforeResetAttr() != nullptr);
                const auto shouldLinearBeforeResetAttr = getIntAttr(ctx, shouldLinearBeforeReset);

                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{gru.getHiddenSizeAttr(), directionModeAttr, gru.getSeqLengthAttr(),
                                                     shouldLinearBeforeResetAttr, gru.getClipAttr()},
                        {"gru_sequence_last_part"}};
            })
            .Case<VPU::GRUGatesOp>([&](VPU::GRUGatesOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"gru_gates"}};
            })
            .Case<VPU::LSTMCellOp>([&](VPU::LSTMCellOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"lstm_cell"}};
            })
            .Case<VPU::LSTMGatesOp>([&](VPU::LSTMGatesOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"lstm_gates"}};
            })
            .Case<VPU::LSTMSequenceOp>([&](VPU::LSTMSequenceOp LSTMSequence) {
                const auto direction =
                        static_cast<std::underlying_type_t<IE::RNNSequenceDirection>>(LSTMSequence.getDirection());
                const auto RNNForward = getIntAttr(ctx, direction);
                auto useDpu = false;
                if (LSTMSequence.getUseDpuAttr()) {
                    useDpu = LSTMSequence.getUseDpuAttr().getValue();
                }
                if (useDpu) {
                    return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{RNNForward}, {"lstm_dpu"}};
                }
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{RNNForward}, {"lstm_sequence"}};
            })
            .Case<VPU::CTCGreedyDecoderSeqLenOp>([&](VPU::CTCGreedyDecoderSeqLenOp op) {
                const auto mergeRepeated = static_cast<int64_t>(op.getMergeRepeatedAttr() != nullptr);
                const auto mergeRepeatedAttr = getIntAttr(ctx, mergeRepeated);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{mergeRepeatedAttr},
                                         {"ctc_greedy_decoder_seq_len"}};
            })
            .Case<VPU::EmbeddingSegmentsSumOp>([&](VPU::EmbeddingSegmentsSumOp op) {
                const auto delimiterAttr = getIntAttr(ctx, INT64_MAX);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{delimiterAttr, op.getNumSegmentsValueAttr(),
                                                                      op.getDefaultIndexValueAttr()},
                                         {"embedding_segments_sum"}};
            })
            .Case<VPU::SquaredDifferenceOp>([&](VPU::SquaredDifferenceOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"eltwise_squared_difference"}};
            })
            .Case<VPU::MaxPoolOp>([&](VPU::MaxPoolOp op) {
                constexpr size_t MAX_ATTR_SZ = 3;  // base on filter description
                auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
                auto strides = parseIntArrayAttr<int64_t>(op.getStridesAttr());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBeginAttr());
                auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEndAttr());
                // In order to support any format, pad with identity up to the max attribute size (3).
                // Filter definition have a strict dependency between input tensor rank, params size and layout, but for
                // hw usage we can cover and situation when this are not respected.
                for (auto i = kernelSize.size(); i < MAX_ATTR_SZ; i++) {
                    kernelSize.insert(kernelSize.begin(), 1);
                    strides.insert(strides.begin(), 1);
                    padsBegin.insert(padsBegin.begin(), 0);
                    padsEnd.insert(padsEnd.begin(), 0);
                }
                const auto kernelSizeAttr = getIntArrayAttr(ctx, kernelSize);
                const auto stridesAttr = getIntArrayAttr(ctx, strides);
                const auto padsBeginAttr = getIntArrayAttr(ctx, padsBegin);
                const auto padsEndAttr = getIntArrayAttr(ctx, padsEnd);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{kernelSizeAttr, stridesAttr, padsBeginAttr, padsEndAttr},
                        {"max_pool"}};
            })
            .Case<VPU::MaxPool8Op>([&](VPU::MaxPool8Op op) {
                constexpr size_t MAX_ATTR_SZ = 3;
                constexpr size_t MAX_DIMS_SZ = 5;
                auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
                auto strides = parseIntArrayAttr<int64_t>(op.getStridesAttr());
                auto dilations = parseIntArrayAttr<int64_t>(op.getDilationsAttr());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBeginAttr());
                auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEndAttr());

                auto numToInsert = MAX_ATTR_SZ - kernelSize.size();
                kernelSize.insert(kernelSize.begin(), numToInsert, 1);
                strides.insert(strides.begin(), numToInsert, 1);
                dilations.insert(dilations.begin(), numToInsert, 1);
                padsBegin.insert(padsBegin.begin(), numToInsert, 0);
                padsEnd.insert(padsEnd.begin(), numToInsert, 0);

                auto getOrDefault = [](const std::optional<mlir::ArrayAttr>& attr,
                                       const SmallVector<int64_t>& defaultValue) {
                    return attr.has_value() ? parseIntArrayAttr<int64_t>(attr.value()) : defaultValue;
                };
                auto iShape = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getShape();
                auto oShape = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType()).getShape();
                auto initialInputDims =
                        getOrDefault(op.getInitialInputDimsAttr(), SmallVector<int64_t>(iShape.begin(), iShape.end()));
                auto initialOutputDims =
                        getOrDefault(op.getInitialOutputDimsAttr(), SmallVector<int64_t>(oShape.begin(), oShape.end()));
                auto initialInputOffset =
                        getOrDefault(op.getInitialInputOffsetAttr(), SmallVector<int64_t>(MAX_DIMS_SZ, 0));
                auto initialOutputOffset =
                        getOrDefault(op.getInitialOutputOffsetAttr(), SmallVector<int64_t>(MAX_DIMS_SZ, 0));

                auto axis = op.getAxis() + MAX_DIMS_SZ - initialInputDims.size();
                initialInputDims.insert(initialInputDims.begin(), MAX_DIMS_SZ - initialInputDims.size(), 1);
                initialOutputDims.insert(initialOutputDims.begin(), MAX_DIMS_SZ - initialOutputDims.size(), 1);
                initialInputOffset.insert(initialInputOffset.begin(), MAX_DIMS_SZ - initialInputOffset.size(), 0);
                initialOutputOffset.insert(initialOutputOffset.begin(), MAX_DIMS_SZ - initialOutputOffset.size(), 0);

                const auto kernelSizeAttr = getIntArrayAttr(ctx, kernelSize);
                const auto stridesAttr = getIntArrayAttr(ctx, strides);
                const auto dilationsAttr = getIntArrayAttr(ctx, dilations);
                const auto padsBeginAttr = getIntArrayAttr(ctx, padsBegin);
                const auto padsEndAttr = getIntArrayAttr(ctx, padsEnd);
                const auto axisParamAttr = getIntAttr(ctx, axis);

                const auto initialInputDimsAttr = getIntArrayAttr(ctx, initialInputDims);
                const auto initialOutputDimsAttr = getIntArrayAttr(ctx, initialOutputDims);
                const auto initialInputOffsetAttr = getIntArrayAttr(ctx, initialInputOffset);
                const auto initialOutputOffsetAttr = getIntArrayAttr(ctx, initialOutputOffset);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{kernelSizeAttr, stridesAttr, dilationsAttr,
                                                                      padsBeginAttr, padsEndAttr, axisParamAttr,
                                                                      initialInputDimsAttr, initialOutputDimsAttr,
                                                                      initialInputOffsetAttr, initialOutputOffsetAttr},
                                         {"max_pool8"}};
            })
            .Case<VPU::AbsOp>([&](VPU::AbsOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_abs"}};
            })
            .Case<VPU::GeluOp>([&](VPU::GeluOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_gelu"}};
            })
            .Case<VPU::BucketizeOp>([&](VPU::BucketizeOp bucketize) {
                const auto dataType = mlir::cast<vpux::NDTypeInterface>(bucketize.getData().getType());
                const auto bucketsType = mlir::cast<vpux::NDTypeInterface>(bucketize.getBuckets().getType());
                VPUX_THROW_UNLESS(dataType.getElementType().isF16() && bucketsType.getElementType().isF16(),
                                  "Only supports FP16 for Input1 & Input2");
                const auto with_right_bound = static_cast<int64_t>(bucketize.getWithRightBound());
                const auto with_right_boundAttr = getIntAttr(ctx, with_right_bound);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{with_right_boundAttr}, {"bucketize"}};
            })
            .Case<VPU::ConvolutionOp>([&](VPU::ConvolutionOp op) {
                auto group = getIntAttr(ctx, checked_cast<int32_t>(1));
                auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEnd());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBegin());
                auto strides = parseIntArrayAttr<int64_t>(op.getStrides());
                auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
                const auto padsBeginAttr = getIntArrayAttr(ctx, padsBegin);
                const auto padsEndAttr = getIntArrayAttr(ctx, padsEnd);
                const auto stridesAttr = getIntArrayAttr(ctx, strides);
                const auto dilationsAttr = getIntArrayAttr(ctx, dilations);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{stridesAttr, padsBeginAttr, padsEndAttr, dilationsAttr, group},
                        {"convolution"}};
            })
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp op) {
                auto group = static_cast<int64_t>(op.getGroups().value());
                auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEnd());
                auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBegin());
                auto strides = parseIntArrayAttr<int64_t>(op.getStrides());
                auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
                const auto padsBeginAttr = getIntArrayAttr(ctx, padsBegin);
                const auto padsEndAttr = getIntArrayAttr(ctx, padsEnd);
                const auto stridesAttr = getIntArrayAttr(ctx, strides);
                const auto dilationsAttr = getIntArrayAttr(ctx, dilations);
                const auto groupAttr = getIntAttr(ctx, group);
                return VPUIP::KernelInfo{
                        SmallVector<mlir::Attribute>{stridesAttr, padsBeginAttr, padsEndAttr, dilationsAttr, groupAttr},
                        {"convolution"}};
            })

            .Case<VPU::DFTOp>([&](VPU::DFTOp op) {
                auto axes = getAxesArrayRevertAndOrderAware(op.getInput(), op.getAxesAttr());
                auto noAxes = op.getAxesAttr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"dft"}};
            })
            .Case<VPU::RDFTUncutOp>([&](VPU::RDFTUncutOp op) {
                auto axes = getAxesArrayRevertAndOrderAware(op.getInput(), op.getAxesAttr());
                auto noAxes = op.getAxesAttr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"rdft"}};
            })
            .Case<VPU::IDFTOp>([&](VPU::IDFTOp op) {
                auto axes = getAxesArrayRevertAndOrderAware(op.getInput(), op.getAxesAttr());
                auto noAxes = op.getAxesAttr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"idft"}};
            })
            .Case<VPU::IRDFTLastAxisOp>([&](VPU::IRDFTLastAxisOp op) {
                auto axes = getAxesArrayRevertAndOrderAware(op.getInput(), op.getAxesAttr());
                auto noAxes = op.getAxesAttr().size();
                const auto noAxesAttr = getIntAttr(ctx, static_cast<int64_t>(noAxes));
                const auto axesAttr = getIntArrayAttr(ctx, axes);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{noAxesAttr, axesAttr}, {"irdft"}};
            })
            .Case<VPU::ConditionalCopyOp>([&](VPU::ConditionalCopyOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"conditional_copy"}};
            })
            .Case<VPU::AccumulateOp>([](VPU::AccumulateOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"accumulate"}};
            })
            .Case<VPU::NonZeroOp>([&](VPU::NonZeroOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"non_zero"}};
            })
            .Case<VPU::RangeOp>([&](VPU::RangeOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"range"}};
            })
            .Case<VPU::ShapeOfOp>([&](VPU::ShapeOfOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"shape_of"}};
            })
            .Case<VPU::PermuteCastOp>([&](VPU::PermuteCastOp op) {
                const auto reverseMap = reversePermutation(op.getMemPerm());
                const auto permuteAttr = getIntArrayAttr(ctx, reverseMap);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{permuteAttr}, {"permute_cast"}};
            })
            .Case<VPU::DynamicReshapeOp>([&](VPU::DynamicReshapeOp op) {
                const auto onlySetShapeValue = static_cast<int64_t>(op.getOnlySetShapeAttr() != nullptr);
                const auto onlySetShapeAttr = getIntAttr(ctx, onlySetShapeValue);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{onlySetShapeAttr},
                                         {"dynamic_reshape"},
                                         {"dynamic_reshape.cpp"}};
            })
            .Case<VPU::DynamicTileOp>([&](VPU::DynamicTileOp tileOp) {
                auto repeatsSize = static_cast<int64_t>(getShape(tileOp.getOutput()).size());
                auto repeatsSizeAttr = getIntAttr(ctx, repeatsSize);
                SmallVector<int64_t> repeatsValue(repeatsSize, 1);
                auto repeatsValueAttr = getIntArrayAttr(ctx, repeatsValue);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{repeatsSizeAttr, repeatsValueAttr},
                                         {"dynamic_tile"}};
            })
            .Case<VPU::ConcatOp>([&](VPU::ConcatOp op) {
                VPUX_THROW_UNLESS(op.getStaticOffsets().has_value(), "VPU.Concat does not provide offsets.");

                const auto staticOffsetsAttr = op.getStaticOffsets().value();
                VPUX_THROW_WHEN(staticOffsetsAttr.size() != 2,
                                "VPU.Concat must have only 2 values of static_offsets attribute.");

                const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInputs().front().getType());
                VPUX_THROW_UNLESS(inType.getElementType().isF16() || inType.getElementType().isUnsignedInteger(8) ||
                                          inType.getElementType().isInteger(32),
                                  "Only supports FP16, U8, SI32 type");

                const auto offsetsRange = staticOffsetsAttr.getAsRange<mlir::ArrayAttr>();
                const mlir::ArrayAttr logicalOffsetsLhs = *offsetsRange.begin();
                const mlir::ArrayAttr logicalOffsetsRhs = *(offsetsRange.begin() + 1);
                const auto order = DimsOrder::fromValue(op.getOutput());
                auto memOffsetsLhs = permuteIntArrayAttr(order, logicalOffsetsLhs);
                auto memOffsetsRhs = permuteIntArrayAttr(order, logicalOffsetsRhs);
                memOffsetsLhs.resize(std::max(memOffsetsLhs.size(), static_cast<size_t>(4)), 0);
                memOffsetsRhs.resize(std::max(memOffsetsRhs.size(), static_cast<size_t>(4)), 0);
                const auto memOffsetsLhsAttr = getIntArrayAttr(ctx, memOffsetsLhs);
                const auto memOffsetsRhsAttr = getIntArrayAttr(ctx, memOffsetsRhs);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{memOffsetsLhsAttr, memOffsetsRhsAttr},
                                         {"concat"}};
            })
            .Case<VPU::RMSOp>([&](VPU::RMSOp op) {
                const auto iType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
                VPUX_THROW_UNLESS(iType.getRank() <= 4, "Supporting only 3D and 4D input, got {0}", iType.getRank());
                const auto epsilonAttr = op.getEpsAttr();
                auto [kernelEntry, kernelSrc] = getKernelImpl(op);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{epsilonAttr},
                                         kernelEntry,
                                         kernelSrc,
                                         {"rms_norm"}};
            })
            .Case<VPU::InverseOp>([&](VPU::InverseOp inverse) {
                const auto adjointAttr = getIntAttr(ctx, static_cast<int64_t>(inverse.getAdjoint()));
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{adjointAttr}, {"inverse"}};
            })
            .Case<VPU::DeformableConvolutionOp>([&](VPU::DeformableConvolutionOp op) {
                const auto bilinearInterpolatepad = static_cast<int64_t>(op.getBilinearInterpolatePadAttr() != nullptr);
                const auto bilinearInterpolatepadAttr = getIntAttr(ctx, bilinearInterpolatepad);

                const auto inOrder = DimsOrder::fromValue(op.getInput());

                const auto initialOutputOffset =
                        op.getInitialOutputOffsetAttr().has_value()
                                ? permuteIntArrayAttr(inOrder, op.getInitialOutputOffsetAttr().value())
                                : SmallVector<int64_t>(inOrder.numDims(), 0);

                const auto initialOutputOffsetAttr = getIntArrayAttr(ctx, initialOutputOffset);

                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{
                                                 op.getStridesAttr(), op.getPadsBeginAttr(), op.getPadsEndAttr(),
                                                 op.getDilationsAttr(), op.getGroupAttr(), op.getDeformableGroupAttr(),
                                                 bilinearInterpolatepadAttr, initialOutputOffsetAttr},
                                         {"deformable_convolution"}};
            })
            .Case<VPU::DynamicExpandOp>([&](VPU::DynamicExpandOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"dynamic_expand"}, {"dynamic_expand.cpp"}};
            })
            .Case<VPU::PopulateWeightTableOp>([&](VPU::PopulateWeightTableOp offPop) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{offPop.getBaseAttr(), offPop.getStepAttr()},
                                         {"populate_weight_table"},
                                         {"populate_weight_table.cpp"}};
            })
            .Case<VPU::RoPEOp>([&](VPU::RoPEOp op) {
                const auto iType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
                VPUX_THROW_UNLESS(iType.getRank() <= 4, "Supporting only 3D and 4D input, got {0}", iType.getRank());

                mlir::MLIRContext* ctx = origOp->getContext();
                const auto isInterleaved = static_cast<int64_t>(op.getIsInterleavedAttr() != nullptr);
                const auto isInterleavedAttr = getIntAttr(ctx, isInterleaved);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{isInterleavedAttr},
                                         {isInterleaved ? "rope_ilv" : "rope"},
                                         {isInterleaved ? "rope_ilv.cpp" : "rope.cpp"}};
            })
            .Case<VPU::DynamicDataMaskOp>([&](VPU::DynamicDataMaskOp) {
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{},
                                         {"dynamic_data_mask"},
                                         {"dynamic_data_mask.cpp"}};
            })
            .Case<VPU::SDPAOp>([&](VPU::SDPAOp op) {
                const auto iType = mlir::cast<vpux::NDTypeInterface>(op.getInputQ().getType());
                VPUX_THROW_UNLESS(iType.getRank() == 4, "Supporting only 4D input, got {0}", iType.getRank());

                const auto useMaskAttr = getIntAttr(ctx, op.getInputMask() == nullptr ? 0 : 1);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{optionalIoAttr(op), useMaskAttr},
                                         {"sdpa"},
                                         {"sdpa.cpp"}};
            })
            .Case<VPU::FlashSDPAOp>([&](VPU::FlashSDPAOp op) {
                int64_t flags = ((1 << 0) * op.getIsHead()) |                      //
                                ((1 << 1) * op.getIsTail()) |                      //
                                ((1 << 2) * (op.getAttentionMask() != nullptr)) |  //
                                ((1 << 3) * (op.getScale() != nullptr));           //

                int64_t sourceSeqLenPadSize = op.getSourceSeqLenPadSize();
                auto attrs =
                        getIntArrayAttr(op->getContext(), SmallVector<int64_t>{INT64_MAX, flags, sourceSeqLenPadSize});
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{attrs}, {"flash_sdpa"}, {"flash_sdpa.cpp"}};
            })
            .Case<VPU::SDPAExtendedOp>([&](VPU::SDPAExtendedOp op) {
                const auto iType = mlir::cast<vpux::NDTypeInterface>(op.getInputQ().getType());
                VPUX_THROW_UNLESS(iType.getRank() == 4, "Supporting only 4D input, got {0}", iType.getRank());
                int64_t padSize = 0;
                if (op.getPadSizeSAttr() != nullptr) {
                    if (op.getPadSizeS().has_value()) {
                        padSize = op.getPadSizeS().value();
                    }
                }
                const auto padSizeAttr = getIntAttr(ctx, padSize);
                return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{optionalIoAttr(op), padSizeAttr},
                                         {"sdpa_extended"}};
            })
            .Default([](mlir::Operation* unknownOp) -> VPUIP::KernelInfo {
                VPUX_THROW("Operation '{0}' is not supported by the act-shaves", unknownOp->getName());
            });
}

#undef CASE_REDUCE

VPUIP::KernelInfo SwKernelOp::getDummyKernelInfo() {
    return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"dummy"}};
}

size_t SwKernelOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto swKernelOp = mlir::cast<VPUIP::SwKernelOp>(this->getOperation());
    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();

    // TODO: Expose API to get arch from cost model
    const auto arch = config::getArch(module);
    return checked_cast<size_t>(calculateShaveActCycles(swKernelOp, costModel, arch));
}

}  // namespace VPUIP
}  // namespace vpux
