//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <memory>
#include <utility>

namespace vpux {
namespace VPU {

//
// WorkloadSizeConstraint
//

struct WorkloadSizeConstraint {
    template <typename T>
    WorkloadSizeConstraint(T t) noexcept: self{std::make_unique<Model<T>>(std::move(t))} {
    }

    // Checks whether the operation needs further workload splitting. `inputShapeOverride`, when
    // non-empty, is used in place of operand(0)'s own (possibly still-dynamic) IR shape — callers
    // that already know the concrete per-tile shape (e.g. SCF flows using dynamic/bounded tensor
    // types) must pass it explicitly here instead of relying on the op's IR type, which may still
    // report `mlir::ShapedType::kDynamic` for dims that are only resolved outside the IR.
    bool doesDWOperationNeedWorkloadSplit(mlir::Operation* op, ShapeRef inputShapeOverride = ShapeRef()) const;

    // Checks if the output tiles results exceed the maximum amount of workloads per invariant
    bool checkDWOperationWorkloadLimit(mlir::Operation* op, const OutputTiling& tiles) const;

    // Returns an array of supported workloads taking into account the already filtered
    // supported channels
    SmallVector<int64_t> getChannelsSupportedBySmallSpatialComputeDwOp(ArrayRef<int64_t> workloadsChannels) const;

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual bool doesDWOperationNeedWorkloadSplit(mlir::Operation* op, ShapeRef inputShapeOverride) const = 0;
        virtual bool checkDWOperationWorkloadLimit(mlir::Operation* op, const OutputTiling& tiles) const = 0;
        virtual SmallVector<int64_t> getChannelsSupportedBySmallSpatialComputeDwOp(
                ArrayRef<int64_t> workloadsChannels) const = 0;
    };

    template <typename T>
    struct Model : Concept {
        Model(T s) noexcept: self{std::move(s)} {
        }
        bool doesDWOperationNeedWorkloadSplit(mlir::Operation* op, ShapeRef inputShapeOverride) const override {
            return self.doesDWOperationNeedWorkloadSplit(op, inputShapeOverride);
        }
        bool checkDWOperationWorkloadLimit(mlir::Operation* op, const OutputTiling& tiles) const override {
            return self.checkDWOperationWorkloadLimit(op, tiles);
        }
        SmallVector<int64_t> getChannelsSupportedBySmallSpatialComputeDwOp(
                ArrayRef<int64_t> workloadsChannels) const override {
            return self.getChannelsSupportedBySmallSpatialComputeDwOp(workloadsChannels);
        }
        T self;
    };

    std::unique_ptr<Concept> self;
};

}  // namespace VPU
}  // namespace vpux
