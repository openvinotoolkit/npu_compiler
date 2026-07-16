//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/bitc/bitc.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace vpux::bitc {

class Encoder {
public:
    Encoder();
    bool encode(const BitCompactorConfig& config, const std::vector<uint8_t>& in, std::vector<uint8_t>& out);
    ~Encoder();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vpux::bitc
