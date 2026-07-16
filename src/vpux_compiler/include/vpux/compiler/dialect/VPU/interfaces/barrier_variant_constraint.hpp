//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include <memory>
#include "vpux/utils/core/helper_macros.hpp"

namespace vpux {
namespace VPU {

//
// PerBarrierSlotConstraint
//

struct PerBarrierSlotConstraint {
    template <typename T>
    PerBarrierSlotConstraint(T t) noexcept: self{std::make_unique<Model<T>>(std::move(t))} {
    }

    size_t getPerBarrierMaxSlotCount() const;

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual size_t getPerBarrierMaxSlotCount() const = 0;
    };

    template <typename T>
    struct Model : Concept {
        Model(T s) noexcept: self{std::move(s)} {
        }
        virtual size_t getPerBarrierMaxSlotCount() const override {
            return self.getPerBarrierMaxSlotCount();
        }
        T self;
    };

    std::unique_ptr<Concept> self;
};

}  // namespace VPU
}  // namespace vpux
