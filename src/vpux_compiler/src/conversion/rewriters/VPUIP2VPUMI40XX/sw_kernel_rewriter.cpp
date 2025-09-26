//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/sw_kernel_rewriter.hpp"
#include <kernels/inc/common_types.h>
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/conversion/passes/VPUIP2VPUMI40XX/buffer_conversion.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/bounded_buffer.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/utils/llvm_to_binary.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace vpux::vpuip2vpumi40xx {

namespace {

// blockArgs: inputs, outputs
// operands : inputs, input_dims, outputs, output_dims
// get {inputs, isDynamic} or {outputs, isDynamic}
auto getOperandValAndIsDynamic(VPUIP::SwKernelOp& swKernelOp, int32_t operandId, int32_t outputId, bool isInput) {
    const auto& shapesMap = isInput ? swKernelOp.getDynamicInputShapesMap() : swKernelOp.getDynamicOutputShapesMap();
    const auto isDynamic = shapesMap && shapesMap.value()[outputId] != ABSENT_DIMS_FLAG;
    const auto& operandVal = swKernelOp->getOpOperand(operandId).get();
    return std::make_tuple(operandVal, isDynamic);
}

auto extractKernelBuffer(VPUIP::SwKernelOp& swKernelOp, int32_t inDimsSize, int32_t blockId) {
    const auto insSize = static_cast<int32_t>(swKernelOp.getInputs().size());

    if (blockId < insSize) {
        const auto operandId = blockId;
        return getOperandValAndIsDynamic(swKernelOp, operandId, operandId, true);
    } else {
        const auto operandId = blockId + inDimsSize;
        const auto outputId = blockId - insSize;
        return getOperandValAndIsDynamic(swKernelOp, operandId, outputId, false);
    }
}

auto computeTileMaskForBroadcast(mlir::Value outputBuff) {
    std::optional<uint32_t> tileMaskForBroadcast;
    const auto distributedOutputBuff = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputBuff.getType());

    if (distributedOutputBuff) {
        const auto distributedTensorAttr = distributedOutputBuff.getDistribution();
        if (distributedTensorAttr && distributedTensorAttr.getMode().getValue() == VPU::DistributionMode::DUPLICATED) {
            const auto numClusters = static_cast<size_t>(distributedTensorAttr.getNumClusters().getInt());
            auto definingOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(outputBuff.getDefiningOp());

            VPUX_THROW_WHEN(!definingOp.getSectionIndex().has_value(), "Distributed buffer without section index: {0}",
                            definingOp);

            const auto clusters = parseIntArrayAttr<uint32_t>(definingOp.getSectionIndex().value());
            VPUX_THROW_WHEN(
                    clusters.size() != numClusters,
                    "Size of distributed buffer section index ({0}) different than distribution num_clusters ({1})",
                    clusters.size(), numClusters);

            tileMaskForBroadcast = VPUMI40XX::generateTileMask(clusters);
        }
    }

    return tileMaskForBroadcast;
}

template <class T>
void appendValueToVector(SmallVector<uint8_t>& vec, const T& anyValue) {
    ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&anyValue), sizeof(anyValue));
    vec.insert(vec.end(), valueAsArray.begin(), valueAsArray.end());
}

sw_params::DataType getDataTypeFromMlirType(mlir::Type type) {
    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
        auto typeWidth = floatType.getWidth();
        switch (typeWidth) {
        case 64:
            return sw_params::DataType::NN_FP64;
        case 32:
            return sw_params::DataType::NN_FP32;
        case 16:
            if (type.isBF16()) {
                return sw_params::DataType::NN_BF16;
            }
            return sw_params::DataType::NN_FP16;
        case 8:
            if (mlir::isa<mlir::Float8E4M3FNType>(floatType)) {
                return sw_params::DataType::NN_HF8;
            } else if (mlir::isa<mlir::Float8E5M2Type>(floatType)) {
                return sw_params::DataType::NN_BF8;
            }
            break;
        }
    } else if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(type)) {
        if (integerType.isSigned()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return sw_params::DataType::NN_I64;
            case 32:
                return sw_params::DataType::NN_I32;
            case 16:
                return sw_params::DataType::NN_I16;
            case 8:
                return sw_params::DataType::NN_I8;
            case 4:
                return sw_params::DataType::NN_I4;
            case 2:
                return sw_params::DataType::NN_I2;
            case 1:
                return sw_params::DataType::NN_BIN;
            }
        } else if (integerType.isUnsigned()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return sw_params::DataType::NN_U64;
            case 32:
                return sw_params::DataType::NN_U32;
            case 16:
                return sw_params::DataType::NN_U16;
            case 8:
                return sw_params::DataType::NN_U8;
            case 4:
                return sw_params::DataType::NN_U4;
            case 2:
                return sw_params::DataType::NN_U2;
            case 1:
                return sw_params::DataType::NN_BIN;
            }
        } else if (integerType.isSignless()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return sw_params::DataType::NN_I64;
            case 32:
                return sw_params::DataType::NN_I32;
            case 16:
                return sw_params::DataType::NN_I16;
            case 8:
                return sw_params::DataType::NN_I8;
            case 4:
                return sw_params::DataType::NN_I4;
            case 2:
                return sw_params::DataType::NN_I2;
            case 1:
                return sw_params::DataType::NN_BIN;
            }
        }
    } else if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type)) {
        const auto isSigned = qType.isSigned();
        auto bitWidth = qType.getStorageTypeIntegralWidth();
        auto isQuantileType =
                mlir::isa<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType>(qType);
        auto isFloatStorage = mlir::isa<mlir::FloatType>(qType.getStorageType());
        switch (bitWidth) {
        case 16:
            if (!isQuantileType && !isFloatStorage) {
                return isSigned ? sw_params::DataType::NN_I16 : sw_params::DataType::NN_U16;
            }
            break;
        case 8:
            if (!isQuantileType && !isFloatStorage) {
                return isSigned ? sw_params::DataType::NN_I8 : sw_params::DataType::NN_U8;
            }
            break;
        case 4:
            if (!isQuantileType && !isFloatStorage) {
                return isSigned ? sw_params::DataType::NN_I4 : sw_params::DataType::NN_U4;
            }
            if (isNF4SpecQuantized(qType)) {
                return sw_params::DataType::NN_NF4;
            }
            break;
        case 2:
            if (!isQuantileType && !isFloatStorage) {
                return isSigned ? sw_params::DataType::NN_I2 : sw_params::DataType::NN_U2;
            }
            break;
        }
    }
    VPUX_THROW("Conversion to sw_params::DataType failed for {0}", type);
    return sw_params::DataType::NN_UNDEFINED;
}

sw_params::Location getSwParamsLocationFromMemKind(VPU::MemoryKind memKind) {
    static const EnumMap<VPU::MemoryKind, sw_params::Location> memKindMapping = {
            {VPU::MemoryKind::DDR, sw_params::Location::DDR},
            {VPU::MemoryKind::CMX_NN, sw_params::Location::NN_CMX},
            {VPU::MemoryKind::CSRAM, sw_params::Location::NONE},
            {VPU::MemoryKind::Register, sw_params::Location::NONE},
    };
    return memKindMapping.at(memKind);
}

void addTensorArgToVector(SmallVector<uint8_t>& vec, std::optional<uint32_t> tileMaskForBroadcast, mlir::Value value,
                          bool isDynamic) {
    sw_params::MemRefData memrefData{};

    const auto shape = getShape(value);
    memrefData.numDims = checked_cast<uint32_t>(shape.size());

    // order
    const auto inOrder = DimsOrder::fromValue(value);
    const auto memShape = inOrder.toMemoryOrder(shape);
    memrefData.dimsOrder = inOrder.invertedCode();

    auto type = value.getType();
    auto ndType = mlir::cast<vpux::NDTypeInterface>(type);

    auto ndTypeMemSpace = ndType.getMemSpace();
    auto tileIndex = static_cast<uint32_t>(ndTypeMemSpace == nullptr ? 0 : ndType.getMemSpace().getIndex().value_or(0));
    auto tileMask = VPUMI40XX::generateTileMask({tileIndex});
    if (tileMaskForBroadcast.has_value()) {
        tileMask |= tileMaskForBroadcast.value();
    }

    memrefData.dataAddr = tileMask;

    memrefData.dataType = getDataTypeFromMlirType(ndType.getElementType());
    memrefData.location = getSwParamsLocationFromMemKind(ndType.getMemoryKind());
    memrefData.isStatic = !isDynamic;

    appendValueToVector(vec, memrefData);
}

void addBasicAttrToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr) {
    if (auto val = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attr)) {
        appendValueToVector(vec, val.getValue().getSExtValue());
    } else if (auto val = mlir::dyn_cast_or_null<mlir::FloatAttr>(attr)) {
        appendValueToVector(vec, static_cast<float>(val.getValue().convertToDouble()));
    } else if (auto val = mlir::dyn_cast_or_null<mlir::TypeAttr>(attr)) {
        appendValueToVector(vec, getDataTypeFromMlirType(val.getValue()));
    } else {
        VPUX_THROW("Act Shave Invocation: cannot store attribute {0}", attr);
    }
}

void addAttrsToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr) {
    if (auto arr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
        auto vals = arr.getValue();
        for (auto val : vals) {
            addBasicAttrToVector(vec, val);
        }
    } else {
        addBasicAttrToVector(vec, attr);
    }
}

// This function is used to serialize MemRef arguments in their LLVM-style representation
// More info on this representation can be found at:
// https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types
void addLLVMMemrefArgToVector(SmallVector<uint8_t>& vec, mlir::Value value,
                              std::optional<uint32_t> tileMaskForBroadcast) {
    auto type = value.getType();
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(type);
    VPUX_THROW_UNLESS(ndType, "Sw Kernel argument is not an ND Type");

    const auto shape = ndType.getShape();
    auto rankMemref = shape.size();

    auto ndTypeMemSpace = ndType.getMemSpace();
    auto tileIndex = static_cast<uint32_t>(ndTypeMemSpace == nullptr ? 0 : ndType.getMemSpace().getIndex().value_or(0));
    auto tileMask = VPUMI40XX::generateTileMask({tileIndex});
    if (tileMaskForBroadcast.has_value()) {
        tileMask |= tileMaskForBroadcast.value();
    }

    // E#149920: Implement helper factory class for creating serial LLVM-style MemRef representations
    uint32_t allocatedPointer = 0;
    uint32_t alignedPointer = tileMask;  // Currently use only alignedPointer.
    int32_t offset = 0;

    auto sizeVec = std::vector<int32_t>(rankMemref);
    auto strideVec = std::vector<int32_t>(rankMemref);

    for (auto size : sizeVec | indexed) {
        vpux::Dim aDim(size.index());
        size.value() = shape[aDim];
    }

    const auto elemTypeSize = vpux::getElemTypeSize(ndType.getElementType());
    const auto dimsOrder = ndType.getDimsOrder();
    const auto memShape = dimsOrder.toMemoryOrder(Shape(ndType.getShape()));
    const auto memStrides = StrideReqs::compact(dimsOrder.numDims()).calcStrides(elemTypeSize, memShape);
    const auto computedStrides = dimsOrder.toLogicalOrder(memStrides);
    VPUX_THROW_WHEN(computedStrides.size() > strideVec.size(), "Number of computed strides is larger than expected");

    for (auto stride : computedStrides | indexed) {
        strideVec[stride.index()] = stride.value().count() / elemTypeSize.count();
    }

    // We can't represent in C/C++ a struct with compile-time-unknown-size arrays (arrays size and stride) with
    // contiguous memory.
    //   Therefore we just serialize the data directly in the vector.
    appendValueToVector(vec, allocatedPointer);

    appendValueToVector(vec, alignedPointer);

    appendValueToVector(vec, offset);

    ArrayRef<uint8_t> sizeByteArray(reinterpret_cast<const uint8_t*>(sizeVec.data()),
                                    sizeof(sizeVec[0]) * sizeVec.size());
    vec.insert(vec.end(), sizeByteArray.begin(), sizeByteArray.end());

    ArrayRef<uint8_t> strideByteArray(reinterpret_cast<const uint8_t*>(strideVec.data()),
                                      sizeof(strideVec[0]) * strideVec.size());
    vec.insert(vec.end(), strideByteArray.begin(), strideByteArray.end());
}

SmallVector<uint8_t> createKernelParams(VPUIP::SwKernelOp swKernelOp, bool isJitCompiled = false) {
    SmallVector<uint8_t> paramsVector;

    const auto insSize = swKernelOp.getInputs().size();
    const auto outsSize = swKernelOp.getOutputBuffs().size();
    const auto dynInputShapesSize = swKernelOp.getDynamicInputShapes().size();

    const auto kernelOpArgsCount = insSize + outsSize;

    for (auto&& kernelRun : swKernelOp.getBody().getOps<VPUIP::SwKernelRun>()) {
        for (auto&& operand : kernelRun.getArgs()) {
            auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(operand);
            if (blockArg) {
                auto blockId = blockArg.getArgNumber();
                VPUX_THROW_UNLESS(blockId < kernelOpArgsCount,
                                  "Index '{0}' of argument of Kernel.Run operation is out of range {1}'", blockId,
                                  kernelOpArgsCount);

                auto blockArgType = blockArg.getType();
                auto blockArgNdTypeIf = mlir::cast<vpux::NDTypeInterface>(blockArgType);
                auto ioType = blockId < insSize ? swKernelOp.getInputs()[blockId].getType()
                                                : swKernelOp.getOutputBuffs()[blockId - insSize].getType();
                auto ioNdTypeIf = mlir::cast<vpux::NDTypeInterface>(ioType);
                VPUX_THROW_UNLESS(blockArgNdTypeIf != nullptr || ioNdTypeIf != nullptr,
                                  "createKernelParams: sw kernel I/O does not implement NDTypeInterface");
                if (!vpux::areTypesCompatible(blockArgType, ioType, vpux::IE::TypeComparisonMode::STRICT_EQUAL, true,
                                              true)) {
                    VPUX_THROW("createKernelParams: types of sw kernel I/O do not match, op: {0}, loc: {1}",
                               swKernelOp->getName(), swKernelOp.getLoc());
                }
                VPUX_THROW_UNLESS(blockArgNdTypeIf.getShape() == ioNdTypeIf.getShape(),
                                  "createKernelParams: shapes of I/O do not match, op: {0}", swKernelOp->getName(),
                                  ", loc: ", swKernelOp.getLoc());

                const auto [buffer, isDynamic] = extractKernelBuffer(swKernelOp, dynInputShapesSize, blockId);
                auto tileMaskForOutputBroadcast =
                        blockId < insSize ? std::nullopt
                                          : computeTileMaskForBroadcast(swKernelOp.getOutputBuffs()[blockId - insSize]);

                if (isJitCompiled) {
                    addLLVMMemrefArgToVector(paramsVector, buffer, tileMaskForOutputBroadcast);
                } else {
                    addTensorArgToVector(paramsVector, tileMaskForOutputBroadcast, buffer, isDynamic);
                }

            } else {
                VPUX_THROW("Only block arguments are supported");
            }
        }
        if (kernelRun.getAttrs().has_value()) {
            const mlir::ArrayAttr arrayAttrs = kernelRun.getAttrs().value();
            const auto& attrs = arrayAttrs.getValue();
            for (const auto& attr : attrs) {
                addAttrsToVector(paramsVector, attr);
            }
        }
    }
    return paramsVector;
}

template <class RangeT>
bool hasDuplicatedMode(RangeT&& outputTypes) {
    auto isDuplicated = [](auto type) {
        auto distributed = mlir::dyn_cast<VPUIP::DistributedBufferType>(type);
        if (!distributed) {
            return false;
        }
        const auto distribution = distributed.getDistribution();
        return distribution && distribution.getMode().getValue() == VPU::DistributionMode::DUPLICATED;
    };

    return llvm::any_of(outputTypes, isDuplicated);
}

void createComputeOpSwKernel(VPUIP::SwKernelOp swKernelOp, mlir::OpBuilder builder, mlir::StringAttr swKernelELF,
                             VPURegMapped::IndexType indexType, bool isJitCompiled = false) {
    auto ctx = builder.getContext();

    auto swKernelTextOp = builder.create<VPUMI40XX::DeclareKernelTextOp>(swKernelOp.getLoc(), indexType, swKernelELF);
    auto swKernelArgsOp = builder.create<VPUMI40XX::DeclareKernelArgsOp>(swKernelOp.getLoc(), indexType, swKernelELF);
    auto swKernelEntryOp = builder.create<VPUMI40XX::DeclareKernelEntryOp>(swKernelOp.getLoc(), indexType, swKernelELF);
    auto parentTaskOp = swKernelOp->getParentOfType<VPURT::TaskOp>();

    auto kernelRangeOp = builder.create<VPUMI40XX::ActKernelRangeOp>(
            swKernelOp.getLoc(), indexType,
            nullptr,  // taskLocation
            nullptr,  // previousTask
            swKernelTextOp, swKernelArgsOp, swKernelEntryOp,
            mlir::SymbolRefAttr::get(ctx, VPU::stringifyActShaveTaskType(VPU::ActShaveTaskType::COMPUTE)),
            parentTaskOp.getWlmPageAttr());

    const auto extractDynShapes = [](mlir::OperandRange dynShapes, ArrayRef<int32_t> dynShapesMap) {
        if (dynShapesMap.empty()) {
            return SmallVector<SmallVector<mlir::Value>>();
        }
        auto dynShapesBuffers = SmallVector<SmallVector<mlir::Value>>(dynShapesMap.size());

        const auto getDynShapeValueByIndex = [&](const auto index) {
            return (index == ABSENT_DIMS_FLAG) ? SmallVector<mlir::Value>{}
                                               : SmallVector<mlir::Value>{dynShapes[index]};
        };
        llvm::transform(dynShapesMap, dynShapesBuffers.begin(), getDynShapeValueByIndex);

        return dynShapesBuffers;
    };

    // mlir::ValueRange does not own data.
    // Therefore, the array to store dynamic shapes must be allocated first.
    // Then SmallVector<mlir::Value> must be casted to mlir::ValueRange.
    const auto toValueRange = [](ArrayRef<mlir::Value> range) -> mlir::ValueRange {
        return range;
    };

    auto dynInputShapesMapOpt = swKernelOp.getDynamicInputShapesMap();
    auto dynInputShapesMap = dynInputShapesMapOpt.value_or(ArrayRef<int32_t>{});
    auto dynInputShapes = extractDynShapes(swKernelOp.getDynamicInputShapes(), dynInputShapesMap);
    auto dynInputShapesRange = SmallVector<mlir::ValueRange>();
    llvm::transform(dynInputShapes, std::back_inserter(dynInputShapesRange), toValueRange);

    auto dynOutputShapesMapOpt = swKernelOp.getDynamicOutputShapesMap();
    auto dynOutputShapesMap = dynOutputShapesMapOpt.value_or(ArrayRef<int32_t>{});
    auto dynOutputShapes = extractDynShapes(swKernelOp.getDynamicOutputShapeBuffs(), dynOutputShapesMap);
    auto dynOutputShapesRange = SmallVector<mlir::ValueRange>();
    llvm::transform(dynOutputShapes, std::back_inserter(dynOutputShapesRange), toValueRange);

    auto isOutputBroadcasted = hasDuplicatedMode(swKernelOp.getResultTypes()) ? mlir::UnitAttr::get(ctx) : nullptr;
    // should it be checked in SWKernelOp::verify?
    assert(!isOutputBroadcasted || swKernelOp.getOutputBuffs().size() == 1);

    auto actKernalParamResults = swKernelOp.getOutputBuffs().size() > 1
                                         ? swKernelOp.getOutputBuffs()
                                         : convertOrUnrollBuffer(builder, swKernelOp.getOutputBuffs()[0]);

    auto paramsVector = createKernelParams(swKernelOp, isJitCompiled);
    auto paramsData = getIntArrayAttr(ctx, paramsVector);
    auto kernelParamsOp = builder.create<VPUMI40XX::KernelParamsOp>(
            swKernelOp.getLoc(), indexType, swKernelOp.getInputs(), actKernalParamResults, dynInputShapesRange,
            dynOutputShapesRange, swKernelELF, paramsData, isOutputBroadcasted,
            isJitCompiled ? mlir::UnitAttr::get(ctx) : nullptr);

    builder.create<VPUMI40XX::ActKernelInvocationOp>(swKernelOp.getLoc(), indexType,
                                                     nullptr,             // taskLocation
                                                     nullptr,             // previousTask
                                                     mlir::ValueRange(),  // waitBarriers
                                                     mlir::ValueRange(),  // updateBarriers
                                                     kernelRangeOp.getResult(), kernelParamsOp.getResult(),
                                                     swKernelOp.getProfilingData(), indexType.getTileIdx(),
                                                     0,  // start_after
                                                     0,  // clean_after
                                                     swKernelOp.getProfilingMetadataAttr(),
                                                     nullptr,                       // enqueueBarrier
                                                     parentTaskOp.getWlmPageAttr()  // wlmPageAttr
    );
}

std::string stringifyCacheSWKernelType(VPU::ActShaveTaskType type) {
    switch (type) {
    case VPU::ActShaveTaskType::CACHE_FLUSH:
        return "cache_op_flush";
    case VPU::ActShaveTaskType::CACHE_INVALIDATE:
        return "cache_op_invalidate";
    case VPU::ActShaveTaskType::CACHE_FLUSH_INVALIDATE:
        return "cache_op_flush_invalidate";
    case VPU::ActShaveTaskType::CACHE_PREFETCH:
        return "cache_op_prefetch";
    default:
        VPUX_THROW("Unrecognized Kernel Task Type");
    }
}

void createCacheOpSwKernel(VPUIP::SwKernelOp swKernelOp, mlir::OpBuilder builder, mlir::func::FuncOp swKernelFuncOp,
                           VPURegMapped::IndexType indexType) {
    auto swKernelTaskType = swKernelFuncOp->getAttrOfType<mlir::SymbolRefAttr>("VPU.task_type");
    auto swKernelTaskTypeLeaf = swKernelTaskType.getLeafReference();

    auto maybeTaskType = VPU::symbolizeActShaveTaskType(swKernelTaskTypeLeaf.strref());
    VPUX_THROW_UNLESS(maybeTaskType.has_value(), "Operation '{0}' has invalid VPU.task_type attribute '{1}'",
                      swKernelOp.getKernelFunction(), swKernelTaskTypeLeaf);
    auto taskType = maybeTaskType.value();

    auto swKernelType = stringifyCacheSWKernelType(taskType);

    mlir::Value swKernelTextOp = nullptr;
    mlir::Value swKernelArgsOp = nullptr;
    mlir::Value swKernelEntryOp = nullptr;
    if (taskType == VPU::ActShaveTaskType::CACHE_PREFETCH) {
        const auto swKernelELF = mlir::cast<mlir::StringAttr>(swKernelOp->getAttr("kernelElfName"));
        swKernelTextOp = builder.create<VPUMI40XX::DeclareKernelTextOp>(swKernelOp.getLoc(), indexType, swKernelELF);
        swKernelArgsOp = builder.create<VPUMI40XX::DeclareKernelArgsOp>(swKernelOp.getLoc(), indexType, swKernelELF);
        swKernelEntryOp = builder.create<VPUMI40XX::DeclareKernelEntryOp>(swKernelOp.getLoc(), indexType, swKernelELF);
    }

    auto ctx = swKernelOp.getContext();
    auto parentTaskOp = swKernelOp->getParentOfType<VPURT::TaskOp>();

    auto kernelRangeOp = builder.create<VPUMI40XX::ActKernelRangeOp>(
            swKernelOp.getLoc(), indexType,
            nullptr,  // taskLocation
            nullptr,  // previousTask
            swKernelTextOp, swKernelArgsOp, swKernelEntryOp,
            mlir::SymbolRefAttr::get(ctx, swKernelTaskTypeLeaf.strref()), parentTaskOp.getWlmPageAttr());

    auto kernelParamsData = getIntArrayAttr(ctx, SmallVector<uint8_t>{0xFF});

    auto kernelParamsOp =
            builder.create<VPUMI40XX::KernelParamsOp>(swKernelOp.getLoc(), indexType,
                                                      mlir::ValueRange(),                // inputs
                                                      mlir::ValueRange(),                // results
                                                      SmallVector<mlir::ValueRange>(0),  // dynInputShapes
                                                      SmallVector<mlir::ValueRange>(0),  // dynOutputShapes
                                                      mlir::StringAttr::get(ctx, swKernelType), kernelParamsData);

    builder.create<VPUMI40XX::ActKernelInvocationOp>(swKernelOp.getLoc(), indexType,
                                                     nullptr,             // taskLocation
                                                     nullptr,             // previousTask
                                                     mlir::ValueRange(),  // waitBarriers
                                                     mlir::ValueRange(),  // updateBarriers
                                                     kernelRangeOp.getResult(), kernelParamsOp.getResult(),
                                                     nullptr,  // profiling_data
                                                     indexType.getTileIdx(),
                                                     0,                             // start_after
                                                     0,                             // clean_after
                                                     nullptr,                       // profilingMetadata
                                                     nullptr,                       // enqueueBarrier
                                                     parentTaskOp.getWlmPageAttr()  // wlmPageAttr
    );
}

}  // namespace

mlir::LogicalResult SWKernelRewriter::matchAndRewrite(VPUIP::SwKernelOp origOp, OpAdaptor,
                                                      mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto moduleOp = origOp->getParentOfType<mlir::ModuleOp>();

    const auto tileIndex = origOp.getTileIndex().value_or(0);
    const auto listIndex = origOp.getListIndex().value_or(0);
    const auto index = VPURegMapped::IndexType::get(ctx, tileIndex, listIndex, 0);
    auto kernelFuncSym = origOp.getKernelFunction();

    if (auto swKernelFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(kernelFuncSym)) {
        auto swKernelTaskType = swKernelFuncOp->getAttrOfType<mlir::SymbolRefAttr>("VPU.task_type");

        if (VPUIP::isCacheOpTaskType(swKernelTaskType)) {
            createCacheOpSwKernel(origOp, rewriter, swKernelFuncOp, index);
        } else {
            auto swKernelELF = swKernelFuncOp->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
            createComputeOpSwKernel(origOp, rewriter, swKernelELF, index);
        }
        // For ShaveCodeGen we expect a LLVMFuncOp at this point
    } else if (auto jitCompiledSwKernelFuncOp = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(kernelFuncSym)) {
        auto llvmFuncOpName = jitCompiledSwKernelFuncOp.getNameAttr();

        llvm::LLVMContext llvmContext;
        auto llvmModule = vpux::translateToLLVMIR(moduleOp, kernelFuncSym, llvmContext);

        createComputeOpSwKernel(origOp, rewriter, llvmFuncOpName, index, true);

        auto log = vpux::Logger("translate-to-LLVMIR", vpux::Logger::global().level());
        vpux::lowerLLVMToBinary(moduleOp, std::move(llvmModule), kernelFuncSym, log);
        ShaveBinaryResources::loadElfData(moduleOp);
    }

    rewriter.eraseOp(origOp);
    return mlir::success();
}

}  // namespace vpux::vpuip2vpumi40xx
