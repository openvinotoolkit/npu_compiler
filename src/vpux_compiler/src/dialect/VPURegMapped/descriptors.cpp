//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURegMapped/descriptors.hpp"

namespace vpux::VPURegMapped::detail {

std::pair<mlir::ParseResult, std::optional<elf::Version>> parseVersion(mlir::AsmParser& parser) {
    StringRef keyword;
    if (parser.parseOptionalKeyword(&keyword).failed()) {
        // No keywords found, no version specified.
        return {mlir::success(), {}};
    }

    if (keyword != "requires") {
        parser.emitError(parser.getCurrentLocation()) << "version keyword \"" << keyword << "\", expected \"requires\"";
        return {mlir::failure(), {}};
    }

    uint32_t major = 0;
    if (parser.parseInteger(major).failed()) {
        return {mlir::failure(), {}};
    }

    if (parser.parseColon().failed()) {
        return {mlir::failure(), {}};
    }

    uint32_t minor = 0;
    if (parser.parseInteger(minor).failed()) {
        return {mlir::failure(), {}};
    }

    if (parser.parseColon().failed()) {
        return {mlir::failure(), {}};
    }

    uint32_t patch = 0;
    if (parser.parseInteger(patch).failed()) {
        return {mlir::failure(), {}};
    }

    return {mlir::success(), std::optional<elf::Version>{std::in_place_t{}, elf::Version{major, minor, patch}}};
}

}  // namespace vpux::VPURegMapped::detail
