//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/ppe_factory.hpp"

#include <mutex>

namespace vpux::VPU {
/* @brief
 * Static class for encapsulating PPE-related objects.
 */
class PpeVersionConfig {
private:
    static std::unique_ptr<IPpeFactory>& _getFactory();

    static std::mutex& _getPpeFactoryMutex() {
        static std::mutex mtx;
        return mtx;
    }

public:
    template <typename ConcreteFactoryT>
    static void setFactory() {
        // Note: Multi-threaded compilation scenarios can concurrently call setFactory in the same process.
        // A mutex prevents data races, but switching between factory types can lead to undefined behavior, since the
        // factory object is shared for all compilation threads. In other words, compiling models for platforms with
        // different PPE factories on separate threads, but part of the same process, is not supported with the current
        // PPE factory design
        std::lock_guard lock(_getPpeFactoryMutex());
        const auto alreadyInit = dynamic_cast<ConcreteFactoryT*>(_getFactory().get()) != nullptr;
        if (!alreadyInit) {
            _getFactory() = std::make_unique<ConcreteFactoryT>();
            Logger::global().info("Changed PpeFactory instance");
        }
    }

    static const IPpeFactory& getFactory();

    template <typename DstT, std::enable_if_t<std::is_pointer_v<DstT>, bool> = true>
    static auto getFactoryAs() {
        using ConstDstPtrT = std::add_pointer_t<std::add_const_t<std::remove_pointer_t<DstT>>>;
        return dynamic_cast<const ConstDstPtrT>(&getFactory());
    }

    template <typename DstT, std::enable_if_t<!std::is_pointer_v<DstT>, bool> = true>
    static const DstT& getFactoryAs() {
        const auto* casted = dynamic_cast<const DstT*>(&getFactory());
        VPUX_THROW_WHEN(casted == nullptr, "Failed to cast the default PpeFactory instance to the required type");
        return *casted;
    }

    static PPEAttr retrievePPEAttribute(mlir::Operation* operation) {
        return getFactory().retrievePPEAttribute(operation);
    }
};

}  // namespace vpux::VPU
