//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpux/utils/core/error.hpp>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp"
#include "vpux/compiler/icompiler.hpp"
#include "vpux/utils/logger/logger.hpp"
#include "vpux_elf/utils/version.hpp"

#include <llvm/ADT/StringRef.h>
#include <mlir/Support/WalkResult.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vpux::backend {

struct BlobCompatibilityInfo {
    uint64_t platformID;
    int64_t numOfTiles;
    elf::Version elfVersion;
    elf::Version miVersion;
};

std::string buildBlobCompatibilityString(const BlobCompatibilityInfo& info);

// Extract MI version from section ops that hold MIVersionInterface as the first op in section content.
// SectionOpT is the section op type walked by the main op.
// GetSectionOpsFn maps SectionOpT -> op range that provides empty(), begin().
template <typename SectionOpT, typename MainOpT, typename GetSectionOpsFn>
elf::Version getMIVersionValue(MainOpT mainOp, GetSectionOpsFn&& getSectionOps) {
    std::optional<elf::Version> res;
    mainOp.walk([&](SectionOpT sectionOp) {
        auto ops = getSectionOps(sectionOp);
        if (ops.empty()) {
            return mlir::WalkResult::skip();
        }

        auto miVersion = mlir::dyn_cast<VPURegMapped::MIVersionInterface>(*ops.begin());
        if (!miVersion) {
            return mlir::WalkResult::skip();
        }

        VPUX_THROW_WHEN(res.has_value(), "Multiple MIVersion data sections found");
        res = miVersion.getVersion();
        return mlir::WalkResult::advance();
    });

    VPUX_THROW_WHEN(!res.has_value(), "MIVersion data section not found");
    return *res;
}

using CalcBlobSizeFn = std::function<elf::Writer()>;
using SerializeFn = std::function<void(uint8_t*, elf::Writer&)>;

// Encapsulates the common ELF export flow for the simple case:
//   calculate blob size → prepare writer → allocate vector → serialize → return.
std::vector<uint8_t> exportToELFCommon(CalcBlobSizeFn calcBlobSize, SerializeFn serializeTo);

BlobView exportToELFCommon(CalcBlobSizeFn calcBlobSize, SerializeFn serializeTo, BlobAllocator& allocator);

}  // namespace vpux::backend
