//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/init/passes_register.hpp"

namespace vpux {

//
// PassesRegistry60XX
//

class PassesRegistry60XX final : public IPassesRegistry {
public:
    void registerPasses() override;
};

}  // namespace vpux
