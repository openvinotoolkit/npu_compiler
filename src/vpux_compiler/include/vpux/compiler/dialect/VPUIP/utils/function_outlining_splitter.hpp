//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPUIP {

//
// FunctionOutlinerAsyncRegion
//
// This class is responsible for outlining async operations into sub functions.
// It defines the algorithm of how to split the main function into smaller functions
// based on the AsyncDepsInfo and the minimum number of operations in a sub function.
class FunctionOutlinerAsyncRegion final : public IFunctionOutliner {
public:
    FunctionOutlinerAsyncRegion(size_t minOpsInBlock, AsyncDepsInfo& depsInfo, Logger log);
    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp mainFunction) override;

private:
    size_t _minOpsInBlock;
    AsyncDepsInfo& _depsInfo;
    Logger _log;
};

}  // namespace VPUIP
}  // namespace vpux
