//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace type {

//===----------------------------------------------------------------------===//
// NF4Type
//===----------------------------------------------------------------------===//

ArrayRef<double> NF4Type::getSpecQuantiles() {
    static constexpr std::array quantiles = {-1.0,
                                             -0.6961928009986877,
                                             -0.5250730514526367,
                                             -0.39491748809814453,
                                             -0.28444138169288635,
                                             -0.18477343022823334,
                                             -0.09105003625154495,
                                             0.0,
                                             0.07958029955625534,
                                             0.16093020141124725,
                                             0.24611230194568634,
                                             0.33791524171829224,
                                             0.44070982933044434,
                                             0.5626170039176941,
                                             0.7229568362236023,
                                             1.0};
    return quantiles;
}

mlir::quant::QuantileType NF4Type::get(mlir::MLIRContext* ctx, mlir::Type storageType, mlir::Type quantileType,
                                       ArrayRef<double> quantiles) {
    if (!quantiles.empty()) {
        VPUX_THROW_UNLESS(quantiles.size() == 16, "quantiles array size of nf4 type needs to be equal to 16");
        return mlir::quant::QuantileType::get(ctx, storageType, quantileType, quantiles);
    }
    return mlir::quant::QuantileType::get(ctx, storageType, quantileType, NF4Type::getSpecQuantiles());
}

bool NF4Type::classof(mlir::Type type) {
    const auto qType = mlir::dyn_cast<mlir::quant::QuantileType>(type);
    return qType && qType.getStorageWidth() == 4 && qType.getQuantiles() == NF4Type::getSpecQuantiles();
}

}  // namespace type
}  // namespace vpux
