//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include "bitc.hpp"

namespace vpux {
namespace bitc {
class Decoder {
public:
    Decoder(std::vector<uint8_t>&& bits, BitCompactorConfig&& config);
    bool decode(std::vector<uint8_t>& out);
    ~Decoder();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace bitc
}  // namespace vpux
