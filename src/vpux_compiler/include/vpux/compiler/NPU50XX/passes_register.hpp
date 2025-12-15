//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/passes_register.hpp"

namespace vpux {

//
// PassesRegistry50XX
//

class PassesRegistry50XX final : public IPassesRegistry {
public:
    void registerPasses() override;
};

}  // namespace vpux
