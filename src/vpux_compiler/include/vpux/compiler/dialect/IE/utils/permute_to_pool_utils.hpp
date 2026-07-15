//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/IR/AffineMap.h>
#include <mlir/Support/LLVM.h>

namespace vpux {

constexpr int64_t PERMUTE_TO_POOLING_THRESHOLD = 32 * 16 * 224;

DimsOrder getNHWCOutputLayout(const DimsOrder& memPermute);

SmallVector<std::pair<Shape, DimsOrder>> calculateConversions(ShapeRef originInputShape, const int64_t alignedChannel,
                                                              const DimsOrder& targetOrder);

bool isLegalConvertToPool(NDTypeInterface inputType, NDTypeInterface outputType, mlir::Operation* parentOp,
                          mlir::AffineMap memPermMap, mlir::MLIRContext* ctx, int64_t numClusters,
                          llvm::StringRef debugName, config::ArchKind arch, const Logger& log);

}  // namespace vpux
