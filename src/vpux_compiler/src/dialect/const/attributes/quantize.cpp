//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/stable_hash.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <climits>

using namespace vpux;

//
// QuantizeAttr::verify
//

mlir::LogicalResult vpux::Const::QuantizeAttr::verify(FuncRef<mlir::InFlightDiagnostic()>,
                                                      mlir::quant::QuantizedType qType) {
    VPUX_THROW_WHEN(!mlir::isa<mlir::IntegerType>(qType.getStorageType()),
                    "Const.Quantize supports only integer storage type");
    return mlir::success(qType != nullptr);
}

//
// QuantizeAttr::print
//

void vpux::Const::QuantizeAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    if (const auto targetTypeVal = getTargetType()) {
        printer.printType(targetTypeVal);
    }
    printer << ">";
}

//
// QuantizeAttr::parse
//

mlir::Attribute vpux::Const::QuantizeAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::quant::QuantizedType elemType;
    parser.parseOptionalType(elemType);

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::QuantizeAttr>(parser.getContext(), elemType);
}

//
// QuantizeAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::QuantizeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto quantType = getTargetType();
    VPUX_THROW_WHEN(quantType == nullptr, "Can't quantize to empty type");
    return input.changeElemType(quantType);
}

bool vpux::Const::QuantizeAttr::inferOutputSplat(bool, vpux::NDTypeInterface) const {
    // Splat depends on output type, for example quantize of splat cst to per-axis type no longer splat
    // But it's static method, so no access to output type. Assuming output is always not splat
    return false;
}

namespace {
template <class StorageType>
Const::Content allocateTempBuffer(mlir::MLIRContext* ctx, vpux::NDTypeInterface outputType, const bool isSplat) {
    constexpr bool isSigned = std::is_signed<StorageType>::value;
    size_t bitWidth = sizeof(StorageType) * CHAR_BIT;
    const auto storageType =
            mlir::IntegerType::get(ctx, bitWidth, isSigned ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned);
    return Const::Content::allocTempBuffer(outputType, storageType, isSplat);
}

using QuantizeFn = std::function<int64_t(double)>;

QuantizeFn createQuantizeFn(double scale, int64_t zeroPoint, mlir::quant::QuantizedType qType) {
    const auto qMin = qType.getStorageTypeMin();
    const auto qMax = qType.getStorageTypeMax();
    const auto inMin = dequantize(qMin, scale, zeroPoint);
    const auto inMax = dequantize(qMax, scale, zeroPoint);
    const auto numLevels = qMax - qMin + 1;
    // fakeQuantize can be used for quantization, we just need to infer it parameters
    // it also helps to handle border cases, when value lies out the quantization range
    // For example <i8:0.003:-21> qType can store values in [-0.321; 0.444] range
    // If dequantized const has values outside that range we want to encode it by -128 or 127(i8 type min/max)
    // For other values quantization is equivalent for FakeQuantization from [-0.321;0.444] to [-128.0; 127.0] range
    return [=](double x) {
        const auto fqVal = fakeQuantize(x, inMin, inMax, qMin, qMax, numLevels);
        return static_cast<int64_t>(fqVal);
    };
}

template <class StorageType>
Const::Content transformImpl(mlir::quant::QuantizedType qElemType, mlir::Type outType, mlir::MLIRContext* ctx,
                             vpux::Const::Content& input) {
    // Splat value cannot be used to store weights for per-axis quantization.
    // Applying different scales to the same splat input value yields non-splat results.
    bool isSplat = input.isSplat() && !mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(qElemType);
    auto output = allocateTempBuffer<StorageType>(ctx, outType, isSplat);
    auto qVals = output.template getTempBuf<StorageType>();

    if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(qElemType)) {
        const auto scale = uniformType.getScale();
        const auto zeroPoint = uniformType.getZeroPoint();
        const auto quantizer = createQuantizeFn(scale, zeroPoint, qElemType);

        // qVals.size is 1 when the input is splat, while realVals.size can be greater than 1
        // realVals must contain same element at every index when the input is splat.
        // Use qVals.size to terminate the loop early in this scenario.
        input.read([&](auto realVals) {
            for (size_t i = 0; i < qVals.size(); ++i) {
                const auto realVal = checked_cast<float>(realVals[i]);
                qVals[i] = static_cast<StorageType>(quantizer(realVal));
            }
        });
    } else if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(qElemType)) {
        const auto realVals = input.getValues<float>();

        const auto scales = uniformType.getScales();
        const auto zeroPoints = uniformType.getZeroPoints();
        const auto axis = Dim(uniformType.getQuantizedDimension());

        const auto dimsOrder = input.getType().getDimsOrder();
        const auto memAxis = dimsOrder.toMemDim(axis);
        const auto memShape = dimsOrder.toMemoryOrder(input.getType().getShape());

        // Get the volume of the dimensions less significant than the quantized axis,
        // which use the same dequantization parameters.
        // (QuantAxisDim, ..., InnerMostDim]
        int64_t innerSize = 1;
        for (size_t i = memAxis.ind() + 1; i < memShape.size(); ++i) {
            innerSize *= memShape[MemDim(i)];
        }
        VPUX_THROW_WHEN(innerSize == 0, "Inner size is zero");

        // Get the size of the quantization dimension.
        const int64_t quantAxisSize = memShape[memAxis];
        VPUX_THROW_WHEN(quantAxisSize == 0, "Quantized axis size is zero");

        const int64_t quantAxisTotalSize = quantAxisSize * innerSize;  // = [QuantAxisDim, ..., InnerMostDim]
        // Get the volume of the dimensions more significant than the quantized axis.
        // [OuterMostDim, ..., QuantAxisDim)
        const int64_t outerSize = memShape.totalSize() / quantAxisTotalSize;
        VPUX_THROW_WHEN(outerSize == 0, "Outer size is zero");

        VPUX_THROW_UNLESS(scales.size() == checked_cast<size_t>(quantAxisSize),
                          "Wrong scales size '{0}', expected '{1}'", scales.size(), quantAxisSize);
        VPUX_THROW_UNLESS(zeroPoints.size() == checked_cast<size_t>(quantAxisSize),
                          "Wrong zeroPoints size '{0}', expected '{1}'", zeroPoints.size(), quantAxisSize);

        SmallVector<QuantizeFn> quantizers;
        for (int64_t i = 0; i < quantAxisSize; ++i) {
            quantizers.push_back(createQuantizeFn(scales[i], zeroPoints[i], qElemType));
        }

        // Outermost loop goes through the volume of the outer dimensions.
        // Middle loop goes through the quantized axis. Scale/ZP are updated based on this index.
        // Innermost loop goes through the volume of the innermost dimensions, which share the same quantization
        // parameters.
        loop_3d(LoopExecPolicy::Parallel, ctx, outerSize, quantAxisSize, innerSize,
                [&](int64_t outerInd, int64_t quantAxisInd, int64_t innerInd) {
                    const auto quantizer = quantizers[quantAxisInd];
                    const auto idx = outerInd * quantAxisTotalSize + quantAxisInd * innerSize + innerInd;
                    qVals[idx] = static_cast<StorageType>(quantizer(realVals[idx]));
                });
    } else {
        VPUX_THROW("Unsupported Quantized Type '{0}'", qElemType);
    }
    return output;
}

}  // namespace

//
// QuantizeAttr::transform
//

Const::Content vpux::Const::QuantizeAttr::transform(vpux::Const::Content& input) const {
    const auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(getTargetType());
    VPUX_THROW_UNLESS(qElemType != nullptr, "Got non quantized type '{0}' in 'QuantizeAttr'");
    const auto storageType = qElemType.getStorageType();
    mlir::Type outType = inferOutputType(input.getType());
    auto ctx = getContext();

    // Note: sub-byte values (e.g. i4) are "unpacked" into 8-bits
    if (vpux::isSubByteType(storageType)) {
        if (storageType.isUnsignedInteger()) {
            return transformImpl<uint8_t>(qElemType, outType, ctx, input);
        } else if (storageType.isSignedInteger()) {
            return transformImpl<int8_t>(qElemType, outType, ctx, input);
        }
        VPUX_THROW("Unexpected storage type");
    } else if (storageType == getSInt8Type(ctx)) {
        return transformImpl<int8_t>(qElemType, outType, ctx, input);
    } else if (storageType == getUInt8Type(ctx)) {
        return transformImpl<uint8_t>(qElemType, outType, ctx, input);
    } else if (storageType == getSInt16Type(ctx)) {
        return transformImpl<int16_t>(qElemType, outType, ctx, input);
    } else if (storageType == getUInt16Type(ctx)) {
        return transformImpl<uint16_t>(qElemType, outType, ctx, input);
    } else if (storageType == getSInt32Type(ctx)) {
        return transformImpl<int32_t>(qElemType, outType, ctx, input);
    } else if (storageType == getUInt32Type(ctx)) {
        return transformImpl<uint32_t>(qElemType, outType, ctx, input);
    } else if (storageType == getSInt64Type(ctx)) {
        return transformImpl<int64_t>(qElemType, outType, ctx, input);
    } else if (storageType == getUInt64Type(ctx)) {
        return transformImpl<uint64_t>(qElemType, outType, ctx, input);
    }
    VPUX_THROW("Unsupported {0} storage type", storageType);
}

//
// QuantizeAttr::getStableHashValue
//

llvm::hash_code vpux::Const::QuantizeAttr::getStableHashValue() const {
    const auto type = getTargetType();
    return llvm::hash_combine(getMnemonic(), getStableHash(type));
}
