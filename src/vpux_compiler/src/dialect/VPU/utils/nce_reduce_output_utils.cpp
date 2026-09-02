//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_reduce_output_utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

SmallVector<VPU::ReduceOutputKind> VPU::getReduceOutputKinds(mlir::Operation* op) {
    SmallVector<ReduceOutputKind> kinds;
    kinds.push_back(ReduceOutputKind::None);  // result 0 = main activation output

    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    if (!nceOp) {
        return kinds;
    }

    if (nceOp.getReduceXyMax()) {
        kinds.push_back(ReduceOutputKind::MaxXY);
    }
    if (nceOp.getReduceXyMin()) {
        kinds.push_back(ReduceOutputKind::MinXY);
    }
    if (nceOp.getReduceTensorMinMax()) {
        kinds.push_back(ReduceOutputKind::TensorMinMax);
    }

    return kinds;
}

bool VPU::hasReduceOutputs(mlir::Operation* op) {
    // getReduceOutputKinds always prepends None for the main output, so a size > 1
    // means at least one extra reduce result is active.
    return getReduceOutputKinds(op).size() > 1;
}

std::optional<Dim> VPU::getReducedDim(mlir::Operation* op) {
    // Only ops with active reduce outputs have a meaningful reduced axis.
    // An op may carry axes_value for other reasons (e.g. unfused reduce consumer);
    // without active reduce results the attribute does not represent a tiling constraint.
    if (!hasReduceOutputs(op)) {
        return std::nullopt;
    }

    const auto axesAttr = mlir::dyn_cast_if_present<mlir::ArrayAttr>(op->getAttr("axes_value"));
    if (!axesAttr) {
        return std::nullopt;
    }

    const auto axes = parseIntArrayAttr<int64_t>(axesAttr);
    if (axes.size() != 1) {
        // Per-tensor reduction (all dims) or unsupported multi-axis: no single reduced dim.
        return std::nullopt;
    }

    const auto rank =
            static_cast<int64_t>(mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getShape().size());
    const auto axis = axes[0] < 0 ? axes[0] + rank : axes[0];
    VPUX_THROW_WHEN(axis < 0 || axis >= rank,
                    "axes_value ({0}) is out of bounds for op '{1}' with output rank {2} at '{3}'", axes[0],
                    op->getName(), rank, op->getLoc());
    return Dim(axis);
}

OutputTiling VPU::getReduceOutputTiling(mlir::Operation* op, const TileInfo& mainTile) {
    const auto kinds = getReduceOutputKinds(op);
    if (kinds.size() <= 1) {
        return {};
    }

    const auto reducedDimOpt = getReducedDim(op);
    const auto rank = mainTile.shape.size();

    VPUX_THROW_WHEN(reducedDimOpt.has_value() && static_cast<size_t>(reducedDimOpt->ind()) >= rank,
                    "axes_value ({0}) is out of bounds for op '{1}' with output rank {2} at '{3}'",
                    reducedDimOpt->ind(), op->getName(), rank, op->getLoc());

    OutputTiling result;

    for (size_t i = 1; i < kinds.size(); ++i) {
        if (kinds[i] == ReduceOutputKind::TensorMinMax) {
            // Per-tensor reduction: all dimensions collapse to 1.
            TileInfo tile(rank);
            for (size_t d = 0; d < rank; ++d) {
                tile.shape[Dim(d)] = 1;
                tile.offsets[Dim(d)] = 0;
                tile.axis[Dim(d)] = 1;
            }
            result.push_back(tile);
        } else {
            // MaxXY / MinXY: follow the main tile but clamp the reduced dim to 1/0.
            TileInfo tile = mainTile;
            if (reducedDimOpt.has_value()) {
                tile.shape[*reducedDimOpt] = 1;
                tile.offsets[*reducedDimOpt] = 0;
                result.push_back(tile);
            } else {
                VPUX_THROW("Op '{0}' at '{1}' has active {2} reduce output but no single reduced dim (missing/invalid "
                           "axes_value)",
                           op->getName(), op->getLoc(), kinds[i] == ReduceOutputKind::MaxXY ? "MaxXY" : "MinXY");
            }
        }
    }

    return result;
}

TileInfo VPU::getMainTileFromReduceOutputTiling(mlir::Operation* op,
                                                const std::pair<mlir::OpResult, TileInfo>& reduceTile) {
    const auto& result = reduceTile.first;
    if (!result) {
        return TileInfo(ShapeRef());
    }

    VPUX_THROW_WHEN(result.getOwner() != op, "Value is not produced by the expected op");

    const auto kinds = VPU::getReduceOutputKinds(op);
    if (kinds.size() <= 1) {
        // no reduce output to infer main tile from
        return TileInfo(ShapeRef());
    }

    const auto resultIdx = result.getResultNumber();
    VPUX_THROW_WHEN(resultIdx >= kinds.size(), "Invalid reduce result index {0} for op {1}", resultIdx, op->getName());

    if (kinds[resultIdx] == VPU::ReduceOutputKind::None) {
        return reduceTile.second;
    }

    if (kinds[resultIdx] == VPU::ReduceOutputKind::TensorMinMax) {
        // cannot derive main tile from per-tensor reduction, as all dims collapse to 1
        return TileInfo(ShapeRef());
    }

    const auto reducedDimOpt = VPU::getReducedDim(op);
    if (!reducedDimOpt.has_value()) {
        // cannot derive main tile from MaxXY / MinXY reduction if no single reduced dim is present
        return TileInfo(ShapeRef());
    }

    auto outputTile = reduceTile.second;
    outputTile.shape[*reducedDimOpt] = getBoundedShape(op->getResult(0))[*reducedDimOpt];
    outputTile.offsets[*reducedDimOpt] = 0;
    return outputTile;
}
