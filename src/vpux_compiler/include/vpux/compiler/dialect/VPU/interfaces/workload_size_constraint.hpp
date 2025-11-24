//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Types.h>
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

    // Checks whether the operation needs further workload splitting
    bool doesDWOperationNeedWorkloadSplit(mlir::Operation* op) const;
    // Returns an array of supported workloads taking into account the already filtered
    // supported channels
    SmallVector<int64_t> getChannelsSupportedBySmallSpatialComputeDwOp(ArrayRef<int64_t> workloadsChannels) const;

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual bool doesDWOperationNeedWorkloadSplit(mlir::Operation* op) const = 0;
        virtual SmallVector<int64_t> getChannelsSupportedBySmallSpatialComputeDwOp(
                ArrayRef<int64_t> workloadsChannels) const = 0;
    };

    template <typename T>
    struct Model : Concept {
        Model(T s) noexcept: self{std::move(s)} {
        }
        bool doesDWOperationNeedWorkloadSplit(mlir::Operation* op) const override {
            return self.doesDWOperationNeedWorkloadSplit(op);
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
