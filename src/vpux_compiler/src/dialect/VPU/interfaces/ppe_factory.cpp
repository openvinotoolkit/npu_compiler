//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/ppe_factory.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

using namespace vpux::VPU;

mlir::Type IPpeAdapterFpPreluAlpha::adaptTypeForPreluAlphaScaling(PPEAttr orig, mlir::Type elemType) const {
    if (!hasQuantScalingThroughPreluAlpha(orig) || !mlir::isa<mlir::quant::QuantizedType>(elemType)) {
        return elemType;
    }

    if (const auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)) {
        return mlir::quant::UniformQuantizedType::get(qType.getFlags(), qType.getStorageType(),
                                                      qType.getExpressedType(), /*scale=*/1.0, qType.getZeroPoint(),
                                                      qType.getStorageTypeMin(), qType.getStorageTypeMax());
    }

    VPUX_THROW("Applying quantization scale (from type: {0}) through pReluAlpha is not supported", elemType);
}
