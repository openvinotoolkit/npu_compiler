//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Types.h>

#include <memory>
#include <utility>

namespace vpux {
namespace VPU {

//
// SparsityConstraint
//

struct SparsityConstraint {
    template <typename T>
    SparsityConstraint(T t) noexcept: self{std::make_unique<Model<T>>(std::move(t))} {
    }

    // Checks whether the given channel size can be configured to be the storage element size
    bool areChannelsFitForSESize(int64_t channels) const;
    bool areChannelsFitForSESize(mlir::Type inputType, int64_t channels) const;

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual bool areChannelsFitForSESize(int64_t channels) const = 0;
        virtual bool areChannelsFitForSESize(mlir::Type inputType, int64_t channels) const = 0;
    };

    template <typename T>
    struct Model : Concept {
        Model(T s) noexcept: self{std::move(s)} {
        }
        bool areChannelsFitForSESize(int64_t channels) const override {
            return self.areChannelsFitForSESize(channels);
        }
        bool areChannelsFitForSESize(mlir::Type inputType, int64_t channels) const override {
            return self.areChannelsFitForSESize(inputType, channels);
        }
        T self;
    };

    std::unique_ptr<Concept> self;
};

}  // namespace VPU
}  // namespace vpux
