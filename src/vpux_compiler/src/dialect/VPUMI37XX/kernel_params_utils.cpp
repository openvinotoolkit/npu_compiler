//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI37XX/kernel_params_utils.hpp"
#include "vpux/compiler/core/bounded_buffer.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

namespace vpux {
namespace VPUMI37XX {

sw_params::DataType KernelParamsSerializer::getDataTypeFromMlirType(mlir::Type type) {
    if (auto floatType = type.dyn_cast<mlir::FloatType>()) {
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
            return sw_params::DataType::NN_FP8;
        }
    } else if (auto integerType = type.dyn_cast<mlir::IntegerType>()) {
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
    } else if (auto quantizeType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto isSigned = quantizeType.isSigned();
        auto bitWidth = quantizeType.getStorageTypeIntegralWidth();
        switch (bitWidth) {
        case 8:
            return isSigned ? sw_params::DataType::NN_I8 : sw_params::DataType::NN_U8;
        case 4:
            return isSigned ? sw_params::DataType::NN_I4 : sw_params::DataType::NN_U4;
        }
    }
    return sw_params::DataType::NN_UNDEFINED;
}

sw_params::Location KernelParamsSerializer::getSwParamsLocationFromMemKind(VPU::MemoryKind memKind) {
    static const EnumMap<VPU::MemoryKind, sw_params::Location> memKindMapping = {
            {VPU::MemoryKind::DDR, sw_params::Location::DDR},
            {VPU::MemoryKind::CMX_NN, sw_params::Location::NN_CMX},
            {VPU::MemoryKind::CSRAM, sw_params::Location::NONE},
            {VPU::MemoryKind::Register, sw_params::Location::NONE},
    };
    return memKindMapping.at(memKind);
}

void KernelParamsSerializer::addBasicAttrToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr) {
    if (auto val = attr.dyn_cast_or_null<mlir::IntegerAttr>()) {
        appendValueToVector(vec, val.getValue().getSExtValue());
    } else if (auto val = attr.dyn_cast_or_null<mlir::FloatAttr>()) {
        appendValueToVector(vec, static_cast<float>(val.getValue().convertToDouble()));
    } else {
        const auto typedAttr = attr.dyn_cast_or_null<mlir::TypedAttr>();
        const auto type = typedAttr != nullptr ? typedAttr.getType() : nullptr;
        VPUX_THROW("Act Shave Invocation: arg type is unknown or unsupported {0}", type);
    }
}

void KernelParamsSerializer::addAttrsToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr) {
    if (auto arr = attr.dyn_cast_or_null<mlir::ArrayAttr>()) {
        auto vals = arr.getValue();
        for (auto val : vals) {
            addBasicAttrToVector(vec, val);
        }
    } else {
        addBasicAttrToVector(vec, attr);
    }
}

void KernelParamsSerializer::addLLVMMemrefArgToVector(SmallVector<uint8_t>& vec, mlir::Value value) {
    const auto shape = getShape(value);
    int64_t rankMemref = checked_cast<uint32_t>(shape.size());

    // We can't have a struct in C++ that has an array field with variable
    // length.
    // The implementation of the struct for MemRef is documented in MLIR at:
    //        https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types
    uint32_t allocatedPointer =
            0;  // Both allocatedPointer and alignedPointer will be relocated (they will be solved by the linker).
    uint32_t alignedPointer = 0;  // (They are linked before execution, in IMDemo.)
                                  // We use only alignedPointer.
    int32_t offset = 0;

    auto sizeVec = std::vector<int32_t>(rankMemref);
    auto strideVec = std::vector<int32_t>(rankMemref);

    for (std::size_t i = 0; i < shape.size(); i++) {
        vpux::Dim aDim(i);
        sizeVec[i] = shape[aDim];
    }

    strideVec[shape.size() - 1] = 1;
    for (int i = (int)shape.size() - 2; i >= 0; i--) {
        strideVec[i] = strideVec[i + 1] * sizeVec[i + 1];
        /*
        The stride between the 2 array elements:
            a[1][1][0]
            a[1][2][0]
          is stride[2] * size[2].
        The stride between the 2 array elements:
            a[1][0][0]
            a[2][0][0]
          is size[1] * stride[2] * size[2].
        */
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

void KernelParamsSerializer::addTensorArgToVector(SmallVector<uint8_t>& vec, mlir::Value value, bool isDynamic) {
    // What happened with assigning to memrefData.dimsAddr, memrefData.stridesAddr?
    //     Answer: They will get serialized with appendValueToVector().
    sw_params::MemRefData memrefData{};

    const auto shape = getShape(value);
    memrefData.numDims = checked_cast<uint32_t>(shape.size());

    // order
    const auto inOrder = DimsOrder::fromValue(value);
    const auto memShape = inOrder.toMemoryOrder(shape);
    memrefData.dimsOrder = inOrder.invertedCode();

    auto type = value.getType();
    auto ndType = type.cast<vpux::NDTypeInterface>();

    memrefData.dataType = getDataTypeFromMlirType(ndType.getElementType());
    memrefData.location = getSwParamsLocationFromMemKind(ndType.getMemoryKind());
    memrefData.isStatic = !isDynamic;

    appendValueToVector(vec, memrefData);
}

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

SmallVector<uint8_t> KernelParamsSerializer::createKernelParams(VPUIP::SwKernelOp swKernelOp) {
    SmallVector<uint8_t> paramsVector;

    const auto insSize = swKernelOp.getInputs().size();
    const auto outsSize = swKernelOp.getResults().size();
    const auto dynInputShapesSize = swKernelOp.getDynamicInputShapes().size();

    const auto kernelOpArgsCount = insSize + outsSize;

    mlir::Operation* firstInnerOp = &(swKernelOp.getBody().front().front());
    auto firstInnerOpPackMemrefs = mlir::dyn_cast<vpux::IERT::PackMemrefsOp>(firstInnerOp);
    if (firstInnerOpPackMemrefs != nullptr) {
        for (auto&& operand : firstInnerOpPackMemrefs.getOperands()) {
            auto blockArg = operand.dyn_cast_or_null<mlir::BlockArgument>();
            if (blockArg) {
                auto blockId = blockArg.getArgNumber();
                VPUX_THROW_UNLESS(blockId < kernelOpArgsCount,
                                  "Index '{0}' of argument of Kernel.Run operation is out of range {1}'", blockId,
                                  kernelOpArgsCount);

                auto blockArgType = blockArg.getType();
                auto blockArgNdTypeIf = blockArgType.cast<vpux::NDTypeInterface>();
                auto ioType = blockId < insSize ? swKernelOp.getInputs()[blockId].getType()
                                                : swKernelOp.getOutputBuffs()[blockId - insSize].getType();
                auto ioNdTypeIf = ioType.cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(blockArgNdTypeIf != nullptr || ioNdTypeIf != nullptr,
                                  "createKernelParams: sw kernel I/O does not implement NDTypeInterface");
                // The types can differ w.r.t. the address space (e.g., memref<1x1000xf16> and memref<1x1000xf16, @DDR>)
                VPUX_THROW_UNLESS(blockArgNdTypeIf.getShape() == ioNdTypeIf.getShape(),
                                  "createKernelParams: shapes of I/O do not match");
                VPUX_THROW_UNLESS(blockArgNdTypeIf.getElementType() == ioNdTypeIf.getElementType(),
                                  "createKernelParams: the element types of I/O do not match");

                const auto operandVal = swKernelOp->getOpOperand(blockId).get();
                // We serialize the data pointed to by the pointer passed to the SHAVE sw kernel.
                // We make this serialization in the compiler since we want
                //   execution on LeonRT to be as efficient as possible.
                addLLVMMemrefArgToVector(paramsVector, operandVal);
            } else {
                VPUX_THROW("Only block arguments are supported");
            }
        }
    } else {
        for (auto&& kernelRun : swKernelOp.getBody().getOps<VPUIP::SwKernelRun>()) {
            for (auto&& operand : kernelRun.getArgs()) {
                auto blockArg = operand.dyn_cast_or_null<mlir::BlockArgument>();
                if (blockArg) {
                    auto blockId = blockArg.getArgNumber();
                    VPUX_THROW_UNLESS(blockId < kernelOpArgsCount,
                                      "Index '{0}' of argument of Kernel.Run operation is out of range {1}'", blockId,
                                      kernelOpArgsCount);

                    auto blockArgType = blockArg.getType();
                    auto blockArgNdTypeIf = blockArgType.cast<vpux::NDTypeInterface>();
                    auto ioType = blockId < insSize ? swKernelOp.getInputs()[blockId].getType()
                                                    : swKernelOp.getOutputBuffs()[blockId - insSize].getType();
                    auto ioNdTypeIf = ioType.cast<vpux::NDTypeInterface>();
                    VPUX_THROW_UNLESS(blockArgNdTypeIf != nullptr || ioNdTypeIf != nullptr,
                                      "createKernelParams: sw kernel I/O does not implement NDTypeInterface");
                    VPUX_THROW_UNLESS(vpux::areTypesCompatible(blockArgType, ioType,
                                                               vpux::IE::TypeComparisonMode::STRICT_EQUAL, true, true),
                                      "createKernelParams: types of sw kernel I/O do not match");
                    VPUX_THROW_UNLESS(blockArgNdTypeIf.getShape() == ioNdTypeIf.getShape(),
                                      "createKernelParams: shapes of I/O do not match");

                    const auto [buffer, isDynamic] = extractKernelBuffer(swKernelOp, dynInputShapesSize, blockId);
                    addTensorArgToVector(paramsVector, buffer, isDynamic);
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
    }

    return paramsVector;
}

}  // namespace VPUMI37XX
}  // namespace vpux
