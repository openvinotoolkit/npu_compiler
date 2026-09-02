//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/backend/export_utils.hpp"
#include "vpux/compiler/version.hpp"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>

#include <algorithm>

namespace vpux::backend {

std::string buildBlobCompatibilityString(const BlobCompatibilityInfo& info) {
    return formatv("compiler={0}.{1};npu={2};t={3};elf={4}.{5}.{6};mi={7}.{8}.{9}", NPU_COMPILER_VERSION_MAJOR,
                   NPU_COMPILER_VERSION_MINOR, info.platformID, info.numOfTiles, info.elfVersion.getMajor(),
                   info.elfVersion.getMinor(), info.elfVersion.getPatch(), info.miVersion.getMajor(),
                   info.miVersion.getMinor(), info.miVersion.getPatch())
            .str();
}

std::vector<uint8_t> exportToELFCommon(CalcBlobSizeFn calcBlobSize, SerializeFn serializeTo) {
    auto elfWriter = calcBlobSize();
    elfWriter.prepareWriter();

    std::vector<uint8_t> blob(elfWriter.getTotalSize());
    serializeTo(blob.data(), elfWriter);

    return blob;
}

BlobView exportToELFCommon(CalcBlobSizeFn calcBlobSize, SerializeFn serializeTo, BlobAllocator& allocator) {
    auto elfWriter = calcBlobSize();
    elfWriter.prepareWriter();

    const auto size = elfWriter.getTotalSize();
    auto blob = allocator.allocate(vpux::Byte{static_cast<int64_t>(size)});
    // For a consistent blob hash make sure that the memory is initialized before serializing.
    // This fill_n is required as the writer will not cover the padding between the sections.
    // The writer will only override the memory for sections inside of the preallocated buffer.
    std::fill_n(blob, size, 0);
    serializeTo(blob, elfWriter);

    return BlobView{blob, static_cast<uint64_t>(size)};
}

}  // namespace vpux::backend
