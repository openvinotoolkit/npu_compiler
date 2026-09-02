//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>

#include "schema/profiling_generated.h"
#include "vpux/utils/profiling/metadata.hpp"

using namespace vpux::profiling;

namespace {

// Builds a minimal, valid ProfilingMeta flatbuffer wrapped into a profiling section payload.
std::vector<uint8_t> makeProfilingPayload() {
    flatbuffers::FlatBufferBuilder builder;
    const auto platform = ProfilingFB::CreatePlatform(builder, /*device=*/static_cast<int8_t>(0));
    const std::vector<flatbuffers::Offset<ProfilingFB::ProfilingSection>> sections;
    const auto profilingBuffer = ProfilingFB::CreateProfilingBufferDirect(builder, &sections, /*size=*/0);
    const auto meta = ProfilingFB::CreateProfilingMeta(builder, PROFILING_METADATA_VERSION_MAJOR,
                                                       PROFILING_METADATA_VERSION_MINOR, platform, profilingBuffer);
    builder.Finish(meta);
    return constructProfilingSectionWithHeader(builder.Release());
}

// Serializes a single-section ELF blob carrying the payload under the given section name and type.
std::vector<uint8_t> makeElfBlobWithSection(const std::vector<uint8_t>& payload, const std::string& sectionName,
                                            elf::Elf_Word sectionType) {
    elf::Writer writer;
    auto section = writer.addBinaryDataSection<uint8_t>(sectionName, sectionType);
    section->setAddrAlign(alignof(uint64_t));
    section->setSize(payload.size());

    writer.prepareWriter();
    std::vector<uint8_t> blob(writer.getTotalSize());
    writer.generateELF(blob.data());
    writer.setSectionsStartAddr(blob.data());
    section->appendData(payload.data(), payload.size());
    return blob;
}

}  // namespace

using ProfilingMetadataSectionTests = ::testing::Test;

// Section carries the stable VPU_SHT_PROF type but not the ".profiling" name -> located by the type ID.
TEST_F(ProfilingMetadataSectionTests, LocatesSectionByStableTypeId) {
    const auto payload = makeProfilingPayload();
    const auto blob = makeElfBlobWithSection(payload, ".renamed_profiling", elf::VPU_SHT_PROF);

    const auto* meta = getProfilingSectionMeta(blob.data(), blob.size());
    ASSERT_NE(meta, nullptr);
    EXPECT_EQ(meta->majorVersion(), PROFILING_METADATA_VERSION_MAJOR);
    EXPECT_EQ(meta->minorVersion(), PROFILING_METADATA_VERSION_MINOR);
}

// Profiling section with different section type -> lookup fails.
TEST_F(ProfilingMetadataSectionTests, ThrowsWhenNoProfilingSection) {
    const auto payload = makeProfilingPayload();
    const auto blob = makeElfBlobWithSection(payload, ".profiling", elf::SHT_PROGBITS);

    EXPECT_ANY_THROW(getProfilingSectionMeta(blob.data(), blob.size()));
}
