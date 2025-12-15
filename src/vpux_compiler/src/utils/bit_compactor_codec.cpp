//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/bit_compactor_codec.hpp"

using namespace vpux;

vpux::BitCompactorCodec::BitCompactorCodec(config::ArchKind arch_kind) {
    switch (arch_kind) {
    case config::ArchKind::NPU37XX:
        arch_type_ = vpux::bitc::ArchType::NPU27;
        break;
    case config::ArchKind::NPU40XX:
        arch_type_ = vpux::bitc::ArchType::NPU4;
        break;
    case config::ArchKind::NPU50XX:
        arch_type_ = vpux::bitc::ArchType::NPU5;
        break;
    default:
        VPUX_THROW("Unsupported arck_kind provided to bitcompactor");
        break;
    }
}

mlir::FailureOr<std::vector<uint8_t>> vpux::BitCompactorCodec::compress(std::vector<uint8_t>& data,
                                                                        CompressionMode mode, const Logger& log) const {
    VPUX_THROW_WHEN(data.empty(), "BitCompactorCodec::compress: Empty input data vector");
    VPUX_THROW_WHEN(mode == CompressionMode::FP16 && arch_type_ == vpux::bitc::ArchType::NPU27,
                    "BitCompactorCodec does not support FP16 compression");

    vpux::bitc::BitCompactorConfig config;
    config.arch_type = arch_type_;
    config.mode_fp16_enable = mode == CompressionMode::FP16;

    vpux::bitc::Encoder encoder{};
    std::vector<uint8_t> compressed_data;

    try {
        encoder.encode(config, data, compressed_data);
    } catch (const Exception& e) {
        log.nest().trace("BitCompactorCodec::compress: {0}", e.what());
        return mlir::failure();
    }

    if (data.size() <= compressed_data.size()) {
        log.nest().trace("BitCompactorCodec::compress: uncompressedDataSize <= compressedSize");
        return mlir::failure();
    }

    return compressed_data;
}

bool vpux::BitCompactorCodec::supportsFP16compression() const {
    return arch_type_ != vpux::bitc::ArchType::NPU27;
}
