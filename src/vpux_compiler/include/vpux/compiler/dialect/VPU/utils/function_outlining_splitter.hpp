//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/function_outlining_splitter.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

// pure_vertical_fusion_region is true when the outlining region contains only vertical fusion subgraph
static constexpr StringLiteral PureVerticalFusionRegionAttrName{"pure_vertical_fusion_region"};

//
// FunctionOutlinerVerticalFusion
//

class FunctionOutlinerVerticalFusion final : public IFunctionOutliner {
public:
    FunctionOutlinerVerticalFusion(size_t numInstanceThreshold, size_t verticalFusionTileThreshold, Logger log);
    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp mainFunction) override;

private:
    size_t _numInstanceThreshold;
    size_t _verticalFusionTileThreshold;
    Logger _log;
};

}  // namespace VPU
}  // namespace vpux
