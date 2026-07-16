//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <optional>

namespace vpux {

constexpr StringLiteral multiClusterStrategy = "multiClusterStrategy";  // only be used for manual strategy utils
constexpr StringLiteral tilingStrategy = "tilingStrategy";
constexpr StringLiteral tilingScenario = "tilingScenario";
constexpr StringLiteral strategyCost = "strategyCost";
constexpr StringLiteral defaultNoValue = "NONE";
constexpr StringLiteral verticalFusion = "verticalFusion";  // only be used for manual strategy utils
constexpr StringLiteral verticalFusionHash = "verticalFusionHash";
constexpr StringLiteral verticalFusionScenario = "VFScenario";
constexpr StringLiteral layerTypeName = "layerType";
constexpr StringLiteral updatedVFTiling = "updatedVFTiling";
constexpr StringLiteral outputPipelining = "outputPipelining";
constexpr StringLiteral outputPipeliningMinFragmentation = "outputPipeliningMinFragmentation";

void collectAllComputeOps(mlir::func::FuncOp func, llvm::MapVector<mlir::Location, mlir::Operation*>& operations,
                          llvm::MapVector<mlir::Location, mlir::Operation*>& outputPipeliningOps,
                          bool updateStrategyForOutputPipelining);

/// Loop attributes are special attributes that are used by loop scheduling. Loop scheduling relies on information about
/// how operations were tiled, vertical-fused and so on. These attributes are attached to the operations in the passes
/// in question and are then propagated along the pipeline to the FeasibleAllocation pass.
/// Identifies the tiling region
constexpr StringLiteral TILING_LOOP_INDEX_ATTR_NAME = "tiling_loop_index";
/// Identifies the vertical fusion region
constexpr StringLiteral VF_LOOP_INDEX_ATTR_NAME = "vf_loop_index";
/// Identifies the tile index within vertical fusion region
constexpr StringLiteral VF_LOOP_TILE_INDEX_ATTR_NAME = "vf_loop_tile_index";

/// This class is a proxy for mlir::IntegerAttr with 64-bit integer type. It integrates nicely with MLIR's typesystem
/// and to the user it looks like any other ordinary attribute. Example usage:
/// ```cpp
///     auto i64Attr = I64Attr::get(context, 42);
///     bool isI64Attr = mlir::isa<I64Attr>(i64Attr);
/// ```
class I64Attr : public mlir::Attribute {
public:
    using mlir::Attribute::Attribute;
    static constexpr size_t BIT_WIDTH = 64;

    static I64Attr get(mlir::MLIRContext* context, int64_t value) {
        return mlir::cast<I64Attr>(mlir::IntegerAttr::get(mlir::IntegerType::get(context, BIT_WIDTH), value));
    }

    /// This function is used by MLIR's isa/cast/dyn_cast functionality. We basically pretend to be an instance of
    /// I64Attr if the attribute is an mlir::IntegerAttr with 64-bit integer type.
    static bool classof(mlir::Attribute attr) {
        const auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
        return intAttr != nullptr && intAttr.getType().isInteger(BIT_WIDTH);
    }
};

using TilingLoopIndexAttr = I64Attr;
using VFLoopIndexAttr = I64Attr;
using VFLoopTileIndexAttr = I64Attr;

struct LoopAttributes {
    LoopAttributes(TilingLoopIndexAttr tilingLoopIndex, VFLoopIndexAttr vfLoopIndex,
                   VFLoopTileIndexAttr vfLoopTileIndex)
            : tilingLoopIndex(tilingLoopIndex), vfLoopIndex(vfLoopIndex), vfLoopTileIndex(vfLoopTileIndex) {
    }

    TilingLoopIndexAttr tilingLoopIndex;
    VFLoopIndexAttr vfLoopIndex;
    VFLoopTileIndexAttr vfLoopTileIndex;
};

LoopAttributes getLoopAttributes(mlir::Operation* op);
/// Copies loop attributes from srcOp to dstOp.
void copyLoopAttributes(mlir::Operation* srcOp, mlir::Operation* dstOp);

}  // namespace vpux
