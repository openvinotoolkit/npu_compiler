//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

namespace vpux::config {
enum class Platform : uint64_t;
}

namespace vpux {

//
// IPassesRegistry
//

class IPassesRegistry {
public:
    virtual void registerPasses() = 0;

    virtual ~IPassesRegistry() = default;
};

class EmptyPassesRegistry final : public IPassesRegistry {
public:
    void registerPasses() override;
};

//
// createPassesRegistry
//

std::unique_ptr<IPassesRegistry> createPassesRegistry(config::Platform platform);

}  // namespace vpux
