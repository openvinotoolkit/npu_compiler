//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux {
namespace VPU {

class ICostModelUtilsInterface : public mlir::DialectInterface::Base<ICostModelUtilsInterface> {
public:
    // required by MLIR's internal type-id infrastructure:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ICostModelUtilsInterface)

    ICostModelUtilsInterface(mlir::Dialect* dialect): Base(dialect) {
    }

    // indicate whether the cost model supports NCEOps with int4 weights.
    virtual bool isNCEWithInt4WeightsSupported() const = 0;

    // indicate whether the cost model cache is supported.
    virtual bool isNNCacheStatisticsSupported() const = 0;

    // indicate whether the cost model supports NCEOps with multi-dim pipeline tiling.
    virtual bool isMultiDimPipelineTilingSupported() const = 0;
};

}  // namespace VPU
}  // namespace vpux
