//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
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
/// Identifies the tile coordinate within a tiling region
constexpr StringLiteral LOOP_TILE_COORD = "loop_tile_coord";
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
using TileCoordinateAttr = mlir::DictionaryAttr;
using VFLoopIndexAttr = I64Attr;
using VFLoopTileIndexAttr = I64Attr;

/// Serializes tiled logical dimensions and their zero-based tile ordinals.
/// `axes` is in ascending logical Dim order, independent of physical layout and unroll order.
/// For a 4D [N, C, H, W] activation with tilingStrategy = [1, 4, 2, 1], `axes = [1, 2]` identifies
/// the C and H axes. `indices = [2, 1]` identifies the third C tile and second H tile (zero-based).
/// Channel-first unrolling emits [0, 0], [0, 1], [1, 0], ...;
/// spatial-first unrolling emits [0, 0], [1, 0], [2, 0], ...; both retain `axes = [1, 2]`.
/// For a 5D grouped activation [G, N, C, H, W] with tilingStrategy = [2, 1, 4, 1, 3],
/// `axes = [0, 2, 4], indices = [1, 2, 1]` identifies the second G tile, third C tile, and second W tile.
TileCoordinateAttr getTileCoordinateAttr(mlir::MLIRContext* context, const TileCoordinate& coordinate);

struct LoopAttributes {
    LoopAttributes(TilingLoopIndexAttr tilingLoopIndex, TileCoordinateAttr loopTileCoord, VFLoopIndexAttr vfLoopIndex,
                   VFLoopTileIndexAttr vfLoopTileIndex)
            : tilingLoopIndex(tilingLoopIndex),
              loopTileCoord(loopTileCoord),
              vfLoopIndex(vfLoopIndex),
              vfLoopTileIndex(vfLoopTileIndex) {
    }

    TilingLoopIndexAttr tilingLoopIndex;
    TileCoordinateAttr loopTileCoord;
    VFLoopIndexAttr vfLoopIndex;
    VFLoopTileIndexAttr vfLoopTileIndex;
};

LoopAttributes getLoopAttributes(mlir::Operation* op);
/// Copies loop attributes from srcOp to dstOp.
void copyLoopAttributes(mlir::Operation* srcOp, mlir::Operation* dstOp);
/// Thread-safely returns a module-wide unused tiling loop index.
int64_t getNextTilingLoopIndex(mlir::ModuleOp module);
/// Thread-safely returns a module-wide unused VF loop index.
int64_t getNextVFLoopIndex(mlir::ModuleOp module);

}  // namespace vpux
