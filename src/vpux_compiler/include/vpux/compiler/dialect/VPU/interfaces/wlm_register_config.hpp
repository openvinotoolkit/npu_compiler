//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include <memory>
#include "vpux/utils/core/helper_macros.hpp"

namespace vpux {
namespace VPU {

//
// RegisterConfig
//

struct RegisterConfig {
    template <typename T>
    RegisterConfig(T t): self{std::make_unique<Model<T>>(std::move(t))} {
    }

    llvm::SmallVector<uint32_t> getSHVRegisterAddrs() const;
    llvm::SmallVector<uint32_t> getDPURegisterAddrs() const;
    uint32_t getNCEBarrierFifoAddr() const;

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual llvm::SmallVector<uint32_t> getSHVRegisterAddrs() const = 0;
        virtual llvm::SmallVector<uint32_t> getDPURegisterAddrs() const = 0;
        virtual uint32_t getNCEBarrierFifoAddr() const = 0;
    };

    template <typename T>
    struct Model : Concept {
        Model(T s): self{std::move(s)} {
        }
        llvm::SmallVector<uint32_t> getSHVRegisterAddrs() const override {
            return self.getSHVRegisterAddrs();
        }
        llvm::SmallVector<uint32_t> getDPURegisterAddrs() const override {
            return self.getDPURegisterAddrs();
        }
        uint32_t getNCEBarrierFifoAddr() const override {
            return self.getNCEBarrierFifoAddr();
        }
        T self;
    };

    std::unique_ptr<Concept> self;
};

}  // namespace VPU
}  // namespace vpux
