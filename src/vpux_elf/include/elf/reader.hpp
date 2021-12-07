//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <elf/types/data_types.hpp>
#include <elf/types/section_header.hpp>
#include <elf/types/program_header.hpp>
#include <elf/types/elf_header.hpp>

#include <string>
#include <vector>

namespace elf {

class Reader {
public:
    class Section {
    public:
        Section() = delete;
        Section(const SectionHeader* sectionHeader, const char* data, const char* name);

        const SectionHeader* getHeader() const;
        size_t getEntriesNum() const;
        const char* getName() const;

        template<typename T>
        const T* getData() const {
            return reinterpret_cast<const T*>(m_data);
        }

    private:
        const SectionHeader* m_sectionHeader;
        const char* m_data;
        const char* m_name;
    };

    class Segment {
    public:
        Segment() = delete;
        Segment(const ProgramHeader* programHeader, const char* data);

        const ProgramHeader* getHeader() const;
        const char* getData() const;

    private:
        Reader* m_reader;
        const ProgramHeader* m_programHeader;
        const char* m_data;
    };

public:
    explicit Reader(const char* blob, size_t size);

    const char* getBlob() const;
    const ELFHeader* getHeader() const;

    size_t getSectionsNum() const;
    size_t getSegmentsNum() const;

    Section getSection(size_t index);
    Segment getSegment(size_t index);

private:
    const char* m_blob = nullptr;

    const ELFHeader* m_elfHeader = nullptr;
    const SectionHeader* m_sectionHeadersStart = nullptr;
    const ProgramHeader* m_programHeadersStart = nullptr;
    const char* m_sectionHeadersNames = nullptr;
};

} // namespace elf