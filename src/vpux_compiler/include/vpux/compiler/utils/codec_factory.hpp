//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Support/LogicalResult.h>

#include <vector>

namespace vpux {

class ICodec {
public:
    enum CompressionAlgorithm { BITCOMPACTOR_CODEC };
    enum class CompressionMode { UINT8, FP16 };
    enum class CompressionPath {
        BTC,
    };
    virtual bool supportsFP16compression() const;
    virtual mlir::FailureOr<std::vector<uint8_t>> compress(std::vector<uint8_t>& data,
                                                           CompressionMode mode = CompressionMode::UINT8,
                                                           CompressionPath compPath = CompressionPath::BTC,
                                                           const Logger& _log = vpux::Logger::global()) const = 0;
    virtual ~ICodec() {};

    static std::string compressionModeToStr(ICodec::CompressionMode mode);
};

std::unique_ptr<ICodec> makeCodec(const ICodec::CompressionAlgorithm algo,
                                  config::ArchKind arch = config::ArchKind::UNKNOWN);
}  // namespace vpux
