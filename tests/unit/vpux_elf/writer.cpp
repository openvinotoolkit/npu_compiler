//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include <array>

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/writer.hpp>

#include "common_test_utils/test_assertions.hpp"

#include <gtest/gtest.h>

namespace {

std::string getSectionName(elf::Reader<elf::ELF_Bitness::Elf64>& reader,
                           const elf::Reader<elf::ELF_Bitness::Elf64>::Section& section) {
    const auto elfHeader = reader.getHeader();
    const auto& testNames = reader.getSection(elfHeader->e_shstrndx);
    const auto strings = testNames.getData<char>();
    return std::string(strings + section.getHeader()->sh_name);
}

std::string getSymbolName(elf::Reader<elf::ELF_Bitness::Elf64>& reader,
                          const elf::Reader<elf::ELF_Bitness::Elf64>::Section& symbolSection,
                          const elf::SymbolEntry& symbol) {
    const auto& symStrTab = reader.getSection(symbolSection.getHeader()->sh_link);
    const auto strings = symStrTab.getData<char>();
    return std::string(strings + symbol.st_name);
}

std::vector<elf::Reader<elf::ELF_Bitness::Elf64>::Section> getSectionsByType(
        elf::Reader<elf::ELF_Bitness::Elf64>& reader, elf::Elf_Word type) {
    std::vector<elf::Reader<elf::ELF_Bitness::Elf64>::Section> res;

    for (size_t i = 0; i < reader.getSectionsNum(); ++i) {
        const auto& section = reader.getSection(i);
        if (section.getHeader()->sh_type == type) {
            res.push_back(section);
        }
    }

    return res;
}

}  // namespace

TEST(ELFWriter, ELFWriterConstructorDoesntThrow) {
    OV_ASSERT_NO_THROW(elf::Writer());
}

TEST(ELFWriter, ELFHeaderForEmptyELFIsCorrect) {
    elf::Writer writer;
    writer.prepareWriter();
    std::vector<uint8_t> blob(writer.getTotalSize());
    OV_ASSERT_NO_THROW(writer.generateELF(blob.data()));

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(blob.data(), blob.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);
    const auto elfHeader = reader.getHeader();
    ASSERT_EQ(elfHeader->e_ident[elf::EI_MAG0], elf::ELFMAG0);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_MAG1], elf::ELFMAG1);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_MAG2], elf::ELFMAG2);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_MAG3], elf::ELFMAG3);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_CLASS], elf::ELFCLASS64);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_DATA], elf::ELFDATA2LSB);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_VERSION], elf::EV_NONE);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_OSABI], 0);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_ABIVERSION], 0);
    ASSERT_EQ(elfHeader->e_type, elf::ET_REL);
    ASSERT_EQ(elfHeader->e_machine, elf::EM_NONE);
    ASSERT_EQ(elfHeader->e_version, elf::EV_NONE);
    ASSERT_EQ(elfHeader->e_entry, 0);
    ASSERT_EQ(elfHeader->e_flags, 0);
    ASSERT_EQ(elfHeader->e_ehsize, sizeof(elf::ELFHeader));
    ASSERT_EQ(elfHeader->e_shentsize, sizeof(elf::SectionHeader));
}

TEST(ELFWriter, BinaryDataSection) {
    constexpr int testAlignment = 64;
    const auto testName = std::string(".test");

    struct TestObject {
        int a;
        float b;
    };
    const auto val1 = TestObject{0, 42};
    const auto val2 = TestObject{42, 0};

    elf::Writer writer;
    auto refSection = writer.addBinaryDataSection<TestObject>(testName);
    refSection->setAddrAlign(testAlignment);
    refSection->setSize(sizeof(val1) + sizeof(val2));

    writer.prepareWriter();
    std::vector<uint8_t> blob(writer.getTotalSize());
    OV_ASSERT_NO_THROW(writer.generateELF(blob.data()));
    OV_ASSERT_NO_THROW(writer.setSectionsStartAddr(blob.data()));

    refSection->appendData(val1);
    refSection->appendData(val2);

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(blob.data(), blob.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);
    const auto binarySections = getSectionsByType(reader, elf::SHT_PROGBITS);
    ASSERT_EQ(binarySections.size(), 1);

    const auto& binarySection = binarySections.front();
    ASSERT_EQ(getSectionName(reader, binarySection), testName);
    ASSERT_EQ(binarySection.getHeader()->sh_addralign, testAlignment);
    ASSERT_EQ(binarySection.getHeader()->sh_entsize, sizeof(TestObject));
    ASSERT_EQ(binarySection.getHeader()->sh_size, sizeof(TestObject) * 2);

    const auto binarySectionData = binarySection.getData<TestObject>();
    ASSERT_EQ(std::tie(binarySectionData[0].a, binarySectionData[0].b), std::tie(val1.a, val1.b));
    ASSERT_EQ(std::tie(binarySectionData[1].a, binarySectionData[1].b), std::tie(val2.a, val2.b));
}

TEST(ELFWriter, EmptySection) {
    const auto testName = std::string(".test");
    constexpr int emptySectionSize = 42;
    constexpr auto emptySectionFlags = elf::SHF_ALLOC;

    elf::Writer writer;
    auto refSection = writer.addEmptySection(testName);
    refSection->setFlags(emptySectionFlags);
    refSection->setSize(emptySectionSize);

    writer.prepareWriter();
    std::vector<uint8_t> blob(writer.getTotalSize());
    OV_ASSERT_NO_THROW(writer.generateELF(blob.data()));

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(blob.data(), blob.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);
    const auto emptySections = getSectionsByType(reader, elf::SHT_NOBITS);
    ASSERT_EQ(emptySections.size(), 1);

    const auto& emptySection = emptySections.front();
    ASSERT_EQ(getSectionName(reader, emptySection), testName);
    ASSERT_EQ(emptySection.getHeader()->sh_size, emptySectionSize);
    ASSERT_EQ(emptySection.getHeader()->sh_flags, emptySectionFlags);
}

TEST(ELFWriter, SymbolSection) {
    const auto testName = std::string(".test");
    constexpr int symbolSize = 42;
    constexpr int symbolValue = 2;
    constexpr auto symbolType = elf::STT_SECTION;

    elf::Writer writer;
    auto emptySection = writer.addEmptySection();

    auto refSection = writer.addSymbolSection(testName);

    auto refSymbol = refSection->addSymbolEntry(testName);
    refSymbol->setValue(symbolValue);
    refSymbol->setSize(symbolSize);
    refSymbol->setType(symbolType);
    refSymbol->setRelatedSection(emptySection);

    writer.prepareWriter();
    std::vector<uint8_t> blob(writer.getTotalSize());
    OV_ASSERT_NO_THROW(writer.generateELF(blob.data()));

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(blob.data(), blob.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);
    const auto symbolSections = getSectionsByType(reader, elf::SHT_SYMTAB);
    ASSERT_EQ(symbolSections.size(), 1);

    const auto& symbolSection = symbolSections.front();
    ASSERT_EQ(getSectionName(reader, symbolSection), testName);
    ASSERT_EQ(symbolSection.getHeader()->sh_entsize, sizeof(elf::SymbolEntry));
    ASSERT_EQ(symbolSection.getHeader()->sh_size, sizeof(elf::SymbolEntry) * 2);

    const auto symbol = symbolSection.getData<elf::SymbolEntry>()[1];

    ASSERT_EQ(getSymbolName(reader, symbolSection, symbol), testName);
    ASSERT_EQ(symbol.st_value, symbolValue);
    ASSERT_EQ(symbol.st_size, symbolSize);
    ASSERT_EQ(elf::elf64STType(symbol.st_info), symbolType);

    const auto relatedSection = reader.getSection(symbol.st_shndx);
    ASSERT_EQ(relatedSection.getHeader()->sh_type, elf::SHT_NOBITS);
}

TEST(ELFWriter, DmaSymbolSection) {
    const auto testName = std::string(".test");

    elf::Writer writer;
    auto dmaSymbolSection = writer.addDmaSymbolSection(testName);
    auto dmaSymbol = dmaSymbolSection->addDmaSymbolEntry();

    elf::DmaSymbolEntry expectedDmaSymbolEntry{
            1, 0xB, {1, 1, 1, 1, 1, 1}, {1, 2, 3, 4, 5, 6}, {0, 0, 1, 0, 0, 0}, {1, 1, 1, 1, 1, 1}, {1, 2, 3, 4, 5, 6},
            4};
    dmaSymbol->setDmaSymbol(expectedDmaSymbolEntry);

    writer.prepareWriter();
    std::vector<uint8_t> blob(writer.getTotalSize());
    OV_ASSERT_NO_THROW(writer.generateELF(blob.data()));

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(blob.data(), blob.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);
    const auto symbolSections = getSectionsByType(reader, elf::VPU_SHT_DMA_SYMBOLS);
    ASSERT_EQ(symbolSections.size(), 1);

    const auto& symbolSection = symbolSections.front();
    ASSERT_EQ(getSectionName(reader, symbolSection), testName);
    ASSERT_EQ(symbolSection.getHeader()->sh_entsize, sizeof(elf::DmaSymbolEntry));
    ASSERT_EQ(symbolSection.getHeader()->sh_size, sizeof(elf::DmaSymbolEntry));

    const auto actualDmaSymbol = symbolSection.getData<elf::DmaSymbolEntry>()[0];

    ASSERT_EQ(std::memcmp(&expectedDmaSymbolEntry, &actualDmaSymbol, sizeof(elf::DmaSymbolEntry)), 0);
}

TEST(ELFWriter, SymbolSectionStableSort) {
    const auto testName = std::string(".test");
    constexpr int symbolValue = 2;
    constexpr auto symbolType = elf::STT_SECTION;
    constexpr std::array<size_t, 31> sizes = {3840, 65536, 65536, 65536, 32768, 16384, 8192,  4096,  2048, 2048, 2048,
                                              2048, 2048,  2048,  2048,  2048,  2048,  2048,  2048,  2048, 2048, 2048,
                                              2048, 2048,  2048,  4096,  8192,  16384, 32768, 65536, 65536};

    elf::Writer writer;
    auto emptySection = writer.addEmptySection();

    auto refSection = writer.addSymbolSection(testName);

    for (size_t i = 0; i < sizes.size(); i++) {
        auto refSymbol = refSection->addSymbolEntry(testName + std::to_string(i));
        refSymbol->setValue(symbolValue);
        refSymbol->setSize(sizes[i]);
        refSymbol->setType(symbolType);
        refSymbol->setRelatedSection(emptySection);
    }

    writer.prepareWriter();
    std::vector<uint8_t> blob(writer.getTotalSize());
    OV_ASSERT_NO_THROW(writer.generateELF(blob.data()));

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(blob.data(), blob.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);
    const auto symbolSections = getSectionsByType(reader, elf::SHT_SYMTAB);
    ASSERT_EQ(symbolSections.size(), 1);

    const auto& symbolSection = symbolSections.front();
    ASSERT_EQ(getSectionName(reader, symbolSection), testName);
    ASSERT_EQ(symbolSection.getHeader()->sh_entsize, sizeof(elf::SymbolEntry));
    ASSERT_EQ(symbolSection.getHeader()->sh_size, sizeof(elf::SymbolEntry) * (sizes.size() + 1));

    ASSERT_EQ(symbolSection.getEntriesNum(), sizes.size() + 1);
    const auto symbols = std::next(symbolSection.getData<elf::SymbolEntry>());
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto& symbol = symbols[i];
        ASSERT_EQ(getSymbolName(reader, symbolSection, symbol), testName + std::to_string(i));
        ASSERT_EQ(symbol.st_value, symbolValue);
        ASSERT_EQ(symbol.st_size, sizes[i]);
        ASSERT_EQ(elf::elf64STType(symbol.st_info), symbolType);

        const auto relatedSection = reader.getSection(symbol.st_shndx);
        ASSERT_EQ(relatedSection.getHeader()->sh_type, elf::SHT_NOBITS);
    }
}

TEST(ELFWriter, RelocationSection) {
    const auto testBinaryDataName = std::string(".test.BinaryData");
    const auto testRelocationName = std::string(".test.Relocation");
    const auto testSymbolSection = std::string(".test.Symbols");
    const auto testSymbolName = std::string(".test.Symbol");
    constexpr auto testSymbolType = elf::STT_SECTION;
    constexpr auto testSymbolSize = 42;
    struct TestObject {
        uint32_t a = 0;
        uint64_t b = 0;
    };

    elf::Writer writer;
    auto refBinaryDataSection = writer.addBinaryDataSection<TestObject>(testBinaryDataName);
    refBinaryDataSection->setSize(sizeof(TestObject));

    auto refSymbolSection = writer.addSymbolSection(testSymbolSection);
    auto refSymbol = refSymbolSection->addSymbolEntry(testSymbolName);
    refSymbol->setType(testSymbolType);
    refSymbol->setSize(testSymbolSize);

    auto refRelocationSection = writer.addRelocationSection(testRelocationName);
    refRelocationSection->setSectionToPatch(refBinaryDataSection);
    refRelocationSection->setSymbolTable(refSymbolSection);
    auto refRelocation = refRelocationSection->addRelocationEntry();
    refRelocation->setSymbol(refSymbol);
    refRelocation->setOffset(sizeof(TestObject::a));
    refRelocation->setAddend(0);

    writer.prepareWriter();
    std::vector<uint8_t> blob(writer.getTotalSize());
    OV_ASSERT_NO_THROW(writer.generateELF(blob.data()));
    OV_ASSERT_NO_THROW(writer.setSectionsStartAddr(blob.data()));
    refBinaryDataSection->appendData(TestObject{});

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(blob.data(), blob.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);
    const auto relocationSections = getSectionsByType(reader, elf::SHT_RELA);
    ASSERT_EQ(relocationSections.size(), 1);

    const auto& relocationSection = relocationSections.front();
    ASSERT_EQ(getSectionName(reader, relocationSection), testRelocationName);
    ASSERT_EQ(relocationSection.getHeader()->sh_entsize, sizeof(elf::RelocationAEntry));
    ASSERT_EQ(relocationSection.getHeader()->sh_size, sizeof(elf::RelocationAEntry));

    const auto relocationToPatch = reader.getSection(relocationSection.getHeader()->sh_info);
    ASSERT_EQ(getSectionName(reader, relocationToPatch), testBinaryDataName);
    ASSERT_EQ(relocationToPatch.getHeader()->sh_type, elf::SHT_PROGBITS);

    auto symbolTable = reader.getSection(relocationSection.getHeader()->sh_link);
    ASSERT_EQ(getSectionName(reader, symbolTable), testSymbolSection);
    ASSERT_EQ(symbolTable.getHeader()->sh_type, elf::SHT_SYMTAB);

    const auto relocation = relocationSection.getData<elf::RelocationAEntry>()[0];
    ASSERT_EQ(relocation.r_addend, 0);
    ASSERT_EQ(relocation.r_offset, sizeof(TestObject::a));

    const auto symbol = symbolTable.getData<elf::SymbolEntry>()[elf::elf64RSym(relocation.r_info)];
    ASSERT_EQ(getSymbolName(reader, symbolTable, symbol), testSymbolName);
    ASSERT_EQ(elf::elf64STType(symbol.st_info), testSymbolType);
    ASSERT_EQ(symbol.st_size, testSymbolSize);
}

TEST(ELFWriter, RelocationSectionForDmaSymbols) {
    const auto testBinaryDataName = std::string(".test.BinaryData");
    const auto testRelocationName = std::string(".test.Relocation");
    const auto testSymbolSection = std::string(".test.Symbols");
    const auto testSymbolName = std::string(".test.Symbol");
    struct TestObject {
        uint32_t a = 0;
        uint64_t b = 0;
    };

    elf::Writer writer;
    auto refBinaryDataSection = writer.addBinaryDataSection<TestObject>(testBinaryDataName);
    refBinaryDataSection->setSize(sizeof(TestObject));

    auto refSymbolSection = writer.addDmaSymbolSection(testSymbolSection);
    auto refSymbol = refSymbolSection->addDmaSymbolEntry();

    elf::DmaSymbolEntry expectedDmaSymbolEntry{
            1, 0xB, {1, 1, 1, 1, 1, 1}, {1, 2, 3, 4, 5, 6}, {0, 0, 1, 0, 0, 0}, {1, 1, 1, 1, 1, 1}, {1, 2, 3, 4, 5, 6},
            4};
    refSymbol->setDmaSymbol(expectedDmaSymbolEntry);

    auto refRelocationSection = writer.addRelocationSection(testRelocationName);
    refRelocationSection->setSectionToPatch(refBinaryDataSection);
    refRelocationSection->setSpecialSymbolTable(refSymbolSection->getIndex());
    auto refRelocation = refRelocationSection->addRelocationEntry();
    refRelocation->setSpecialSymbol(refSymbol->getIndex());
    refRelocation->setOffset(sizeof(TestObject::a));
    refRelocation->setAddend(0);

    writer.prepareWriter();
    std::vector<uint8_t> blob(writer.getTotalSize());
    OV_ASSERT_NO_THROW(writer.generateELF(blob.data()));
    OV_ASSERT_NO_THROW(writer.setSectionsStartAddr(blob.data()));
    refBinaryDataSection->appendData(TestObject{});

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(blob.data(), blob.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);
    const auto relocationSections = getSectionsByType(reader, elf::SHT_RELA);
    ASSERT_EQ(relocationSections.size(), 1);

    const auto& relocationSection = relocationSections.front();
    ASSERT_EQ(getSectionName(reader, relocationSection), testRelocationName);
    ASSERT_EQ(relocationSection.getHeader()->sh_entsize, sizeof(elf::RelocationAEntry));
    ASSERT_EQ(relocationSection.getHeader()->sh_size, sizeof(elf::RelocationAEntry));

    const auto relocationToPatch = reader.getSection(relocationSection.getHeader()->sh_info);
    ASSERT_EQ(getSectionName(reader, relocationToPatch), testBinaryDataName);
    ASSERT_EQ(relocationToPatch.getHeader()->sh_type, elf::SHT_PROGBITS);

    auto symbolTable = reader.getSection(relocationSection.getHeader()->sh_link);
    ASSERT_EQ(getSectionName(reader, symbolTable), testSymbolSection);
    ASSERT_EQ(symbolTable.getHeader()->sh_type, elf::VPU_SHT_DMA_SYMBOLS);

    const auto relocation = relocationSection.getData<elf::RelocationAEntry>()[0];
    ASSERT_EQ(relocation.r_addend, 0);
    ASSERT_EQ(relocation.r_offset, sizeof(TestObject::a));
}

TEST(ELFWriter, SpecialSymReloc) {
    const auto testBinaryDataName = std::string(".test.BinaryData");
    const auto testRelocationName = std::string(".test.Relocation");
    constexpr auto testSpecialSymbolSectionIndex = elf::VPU_RT_SYMTAB;
    constexpr auto testSpecialSymbolIndex = elf::VPU_NNRD_SYM_RTM_DMA0;
    constexpr auto testSpecialRelocationType = elf::R_VPU_64;

    struct TestObject {
        uint32_t a = 0;
        uint64_t b = 0;
    };

    elf::Writer writer;
    auto refBinaryDataSection = writer.addBinaryDataSection<TestObject>(testBinaryDataName);
    refBinaryDataSection->setSize(sizeof(TestObject));

    auto refRelocationSection = writer.addRelocationSection(testRelocationName);
    refRelocationSection->setSectionToPatch(refBinaryDataSection);
    refRelocationSection->setSpecialSymbolTable(testSpecialSymbolSectionIndex);

    auto refRelocation = refRelocationSection->addRelocationEntry();
    refRelocation->setType(testSpecialRelocationType);
    refRelocation->setSpecialSymbol(testSpecialSymbolIndex);
    refRelocation->setOffset(0);
    refRelocation->setAddend(0);

    writer.prepareWriter();
    std::vector<uint8_t> blob(writer.getTotalSize());
    OV_ASSERT_NO_THROW(writer.generateELF(blob.data()));
    OV_ASSERT_NO_THROW(writer.setSectionsStartAddr(blob.data()));
    refBinaryDataSection->appendData(TestObject{});

    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(blob.data(), blob.size());
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&accessor);
    auto relocationSections = getSectionsByType(reader, elf::SHT_RELA);
    ASSERT_EQ(relocationSections.size(), 1);

    auto relocationSection = relocationSections.front();
    ASSERT_EQ(getSectionName(reader, relocationSection), testRelocationName);
    ASSERT_EQ(relocationSection.getHeader()->sh_entsize, sizeof(elf::RelocationAEntry));
    ASSERT_EQ(relocationSection.getHeader()->sh_size, sizeof(elf::RelocationAEntry));

    const auto relocationToPatch = reader.getSection(relocationSection.getHeader()->sh_info);
    ASSERT_EQ(getSectionName(reader, relocationToPatch), testBinaryDataName);
    ASSERT_EQ(relocationToPatch.getHeader()->sh_type, elf::SHT_PROGBITS);
    ASSERT_EQ(relocationSection.getHeader()->sh_link, testSpecialSymbolSectionIndex);

    const auto relocation = relocationSection.getData<elf::RelocationAEntry>()[0];
    ASSERT_EQ(relocation.r_addend, 0);
    ASSERT_EQ(relocation.r_offset, 0);
    ASSERT_EQ(elf::elf64RSym(relocation.r_info), testSpecialSymbolIndex);
    ASSERT_EQ(elf::elf64RType(relocation.r_info), testSpecialRelocationType);
}

TEST(ELFWriter, WrongState) {
    elf::Writer writer;
    constexpr size_t wrongBlobSize = 256;
    std::vector<uint8_t> blob(wrongBlobSize);
    // can not generate elf without call to prepareWriter
    ASSERT_THROW(writer.generateELF(blob.data()), elf::ImplausibleState);
}
