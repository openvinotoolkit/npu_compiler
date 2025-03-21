//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <sstream>
#include <string_view>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

// clang-format off
// because header file should be in the first
#include <vpux/compiler/dialect/VPU/enums.hpp.inc>
#include <vpux/compiler/dialect/VPU/enums.cpp.inc>
// clang-format on

#include "vpux/utils/core/format.hpp"

enum ActionType { Generate };

static llvm::cl::opt<ActionType> Action(llvm::cl::desc("Actions to perform"),
                                        llvm::cl::values(clEnumValN(Generate, "generate", "")),
                                        llvm::cl::init(Generate));

static llvm::cl::opt<vpux::VPU::ArchKind> Platform(llvm::cl::desc("Specify the platform type"),
                                                   llvm::cl::values(clEnumValN(vpux::VPU::ArchKind::NPU40XX, "NPU40XX",
                                                                               "LNL platform")
                                                                    // clang-format off
        ), llvm::cl::init(vpux::VPU::ArchKind::NPU40XX));
// clang-format on

static std::map<std::string, std::string> platformTypeMap{
        {"NPU40XX", "NPUReg40XX"},
};

template <class... Args>
void throwFormatted(llvm::StringLiteral format, Args&&... args) {
    std::stringstream message;
    vpux::printTo(message, format, std::forward<Args>(args)...);
    throw std::runtime_error(message.str());
}

template <class T>
auto getAs(const llvm::Record* record, std::string_view name) {
    const auto recordValue = record->getValue(name);
    if (!recordValue) {
        throwFormatted("Couldn't find {0} in record\n{1}", name, *record);
    }

    const auto recordValueInit = recordValue->getValue();
    if (!recordValueInit) {
        throwFormatted("Invalid Init for {0} of\n{1}", name, *record);
    }

    if (auto typed = llvm::dyn_cast_if_present<T>(recordValueInit)) {
        return typed;
    } else {
        throwFormatted("Unexpected type for {0} of\n{1}", name, *record);
        return typed;
    }
}

using Records = std::unordered_map<std::string, std::pair<const llvm::Record*, llvm::SmallVector<std::string>>>;

// until C++20 string literals are unavailable as template arguments
// generate a workaround to pass strings (names) as template arguments
// instead of
//
// ```
// template <const char* String>
// struct TemplatedBase
// ...
// struct Foo : TemplatedBase<"string">
// ```
//
// use following
//
// ```
// template <const char* String>
// struct Templated Base
// ...
// inline constexpr char fooName[] = "string";
// ...
// struct Foo : TemplatedBase<fooName>
// ```
//
// "inline" in the above is actually important as definition is in the header
// without it upon each #include it defines separate type, and so different
// type as result of template instantiation - lead to ambiguous build errors
// e.g. the same method is unused and used, but not defined

llvm::raw_ostream& emitNameAsTemplateArgumentWorkAround(llvm::raw_ostream& stream, const Records& fields,
                                                        const Records& registers, const Records& descriptors,
                                                        const std::string& platformTypeName) {
    stream << "namespace vpux::" << platformTypeName << "::detail {\n";
    const auto emitWorkaround = [&stream](std::string_view level, const Records& records) {
        stream << "namespace " << level << " {\n";
        for (const auto& [name, _] : records) {
            stream << "inline constexpr char " << name << "Name[] = \"" << name << "\";\n";
        }
        stream << "}  // namespace " << level << '\n';
    };
    emitWorkaround("Fields", fields);
    emitWorkaround("Registers", registers);
    emitWorkaround("Descriptors", descriptors);
    stream << "}  // namespace vpux::" << platformTypeName << "::detail\n";
    return stream;
}

llvm::raw_ostream& emitForwardDeclarations(llvm::raw_ostream& stream, const Records& fields, const Records& registers,
                                           const std::string& platformTypeName) {
    stream << "namespace vpux::" << platformTypeName << " {\n";
    const auto emitDeclarations = [&stream](std::string_view level, const Records& records) {
        stream << "namespace " << level << " {\n";
        for (const auto& [name, _] : records) {
            stream << "struct " << name << ";\n";
        }
        stream << "}  // namespace " << level << '\n';
    };
    emitDeclarations("Fields", fields);
    emitDeclarations("Registers", registers);
    stream << "}  // namespace vpux::" << platformTypeName << "\n";
    return stream;
}

llvm::raw_ostream& emitDescriptorsDefinitions(llvm::raw_ostream& stream, const Records& descriptors,
                                              const std::string& platformTypeName) {
    stream << "namespace vpux::" << platformTypeName << "::Descriptors {\n";
    constexpr llvm::StringLiteral descriptorTemplate =
            "struct {0} : ::vpux::VPURegMapped::detail::DescriptorTemplate<{0}, "
            "::vpux::{1}::detail::Descriptors::{0}Name, {2}> {{};\n";
    for (const auto& [name, descriptorEntry] : descriptors) {
        const auto [descriptor, parentsNames] = descriptorEntry;
        assert(parentsNames.empty());

        const auto registersListInit = llvm::dyn_cast<llvm::ListInit>(descriptor->getValue("_registers")->getValue());
        std::stringstream registersList;
        for (size_t i = 0; i < registersListInit->getValues().size(); ++i) {
            const auto registerName = llvm::dyn_cast<llvm::StringInit>(registersListInit->getElement(i));
            registersList << "::vpux::" << platformTypeName << "::Registers::" << registerName->getAsUnquotedString();
            if (i < registersListInit->getValues().size() - 1) {
                registersList << ", ";
            }
        }
        vpux::printTo(stream, descriptorTemplate, name, platformTypeName, registersList.str());
    }
    stream << "}  // namespace vpux::" << platformTypeName << "::Descriptors\n";
    return stream;
}

llvm::raw_ostream& emitRegistersDefinitions(llvm::raw_ostream& stream, const Records& registers,
                                            const std::string& platformTypeName) {
    stream << "namespace vpux::" << platformTypeName << "::Registers {\n";
    constexpr llvm::StringLiteral registerTemplate =
            "struct {0} : "
            "::vpux::VPURegMapped::detail::RegisterTemplate<::vpux::{1}::detail::Registers::{0}Name, "
            "{2}, {3}, {4}, {5}> {{};\n";

    for (const auto& [name, registerEntry] : registers) {
        const auto [reg, descriptorsNames] = registerEntry;
        assert(!descriptorsNames.empty());

        std::stringstream descriptorsList;
        descriptorsList << "std::tuple<";
        for (size_t i = 0; i < descriptorsNames.size(); ++i) {
            descriptorsList << "::vpux::" << platformTypeName << "::Descriptors::" << descriptorsNames[i];
            if (i < descriptorsNames.size() - 1) {
                descriptorsList << ", ";
            }
        }
        descriptorsList << '>';

        const auto fieldsListInit = llvm::dyn_cast<llvm::ListInit>(reg->getValue("_fields")->getValue());
        std::stringstream fieldsList;
        for (size_t i = 0; i < fieldsListInit->getValues().size(); ++i) {
            const auto fieldName = llvm::dyn_cast<llvm::StringInit>(fieldsListInit->getElement(i));
            fieldsList << "::vpux::" << platformTypeName << "::Fields::" << fieldName->getAsUnquotedString();
            if (i < fieldsListInit->getValues().size() - 1) {
                fieldsList << ", ";
            }
        }

        const auto offsetVal = getAs<llvm::IntInit>(reg, "_offset")->getValue();
        const auto sizeVal = getAs<llvm::IntInit>(reg, "_size")->getValue();
        vpux::printTo(stream, registerTemplate, name, platformTypeName, descriptorsList.str(), offsetVal, sizeVal,
                      fieldsList.str());
    }
    stream << "}  // namespace vpux::" << platformTypeName << "::Registers\n";
    return stream;
}

llvm::raw_ostream& emitFieldsDefinitions(llvm::raw_ostream& stream, const Records& fields,
                                         const std::string& platformTypeName) {
    stream << "namespace vpux::" << platformTypeName << "::Fields {\n";
    constexpr llvm::StringLiteral fieldTemplate =
            "struct {0} : ::vpux::VPURegMapped::detail::FieldTemplate<::vpux::{1}::detail::Fields::{0}Name, "
            "{2}, {3}, {4}, ::vpux::VPURegMapped::RegFieldDataType::{5}, {6}, {7}, {8}> "
            "{{};\n";

    for (const auto& [name, fieldEntry] : fields) {
        const auto [field, registersNames] = fieldEntry;
        assert(!registersNames.empty());

        const auto offsetVal = getAs<llvm::IntInit>(field, "_offset")->getValue();
        const auto sizeVal = getAs<llvm::IntInit>(field, "_size")->getValue();
        const auto typeVal = getAs<llvm::StringInit>(field, "_type")->getValue();
        const auto versionVal = getAs<llvm::DefInit>(field, "_version")->getDef();
        const auto major = getAs<llvm::IntInit>(versionVal, "major")->getValue();
        const auto minor = getAs<llvm::IntInit>(versionVal, "minor")->getValue();
        const auto patch = getAs<llvm::IntInit>(versionVal, "patch")->getValue();

        std::stringstream parentRegistersArgument;
        parentRegistersArgument << "std::tuple<";
        for (size_t i = 0; i < registersNames.size(); ++i) {
            parentRegistersArgument << "::vpux::" << platformTypeName << "::Registers::" << registersNames[i];
            if (i < registersNames.size() - 1) {
                parentRegistersArgument << ", ";
            }
        }
        parentRegistersArgument << '>';

        vpux::printTo(stream, fieldTemplate, name, platformTypeName, parentRegistersArgument.str(), offsetVal, sizeVal,
                      typeVal, major, minor, patch);
    }
    stream << "}  // namespace vpux::" << platformTypeName << "::Fields\n";
    return stream;
}

llvm::Error generate(llvm::raw_ostream& stream, llvm::RecordKeeper& records, const std::string& platformTypeName) {
    stream << "#pragma once\n";
    stream << '\n';
    stream << "#include <cstdint>\n";
    stream << "#include <string_view>\n";
    stream << '\n';

    Records fields;
    for (auto field : records.getAllDerivedDefinitionsIfDefined(platformTypeName + "_RegFieldWrapper")) {
        fields[getAs<llvm::StringInit>(field, "_name")->getAsUnquotedString()] =
                std::make_pair(field, llvm::SmallVector<std::string>{});
    }

    Records registers;
    for (auto reg : records.getAllDerivedDefinitionsIfDefined(platformTypeName + "_RegisterWrapper")) {
        const auto registerName = getAs<llvm::StringInit>(reg, "_name")->getAsUnquotedString();
        registers[registerName] = std::make_pair(reg, llvm::SmallVector<std::string>{});

        const auto fieldsListInit = llvm::dyn_cast<llvm::ListInit>(reg->getValue("_fields")->getValue());
        for (auto field : fieldsListInit->getValues()) {
            const auto fieldName = llvm::dyn_cast<llvm::StringInit>(field)->getAsUnquotedString();
            assert(fields.count(fieldName) == 1);
            fields[fieldName].second.push_back(registerName);
        }
    }

    Records descriptors;
    for (auto descriptor : records.getAllDerivedDefinitionsIfDefined(platformTypeName + "_RegMappedWrapper")) {
        const auto descriptorName = getAs<llvm::StringInit>(descriptor, "_name")->getAsUnquotedString();
        descriptors[descriptorName] = std::make_pair(descriptor, llvm::SmallVector<std::string>{});

        const auto registersListInit = llvm::dyn_cast<llvm::ListInit>(descriptor->getValue("_registers")->getValue());
        for (auto reg : registersListInit->getValues()) {
            const auto registerName = llvm::dyn_cast<llvm::StringInit>(reg)->getAsUnquotedString();
            assert(registers.count(registerName) == 1);
            registers[registerName].second.push_back(descriptorName);
        }
    }

    emitNameAsTemplateArgumentWorkAround(stream, fields, registers, descriptors, platformTypeName);
    emitForwardDeclarations(stream, fields, registers, platformTypeName);

    emitDescriptorsDefinitions(stream, descriptors, platformTypeName);
    emitRegistersDefinitions(stream, registers, platformTypeName);
    emitFieldsDefinitions(stream, fields, platformTypeName);

    return llvm::Error::success();
}

bool RegGenMain(llvm::raw_ostream& stream, llvm::RecordKeeper& records) {
    auto doGenerate = [](auto& stream, auto& records, auto& platformTypeName) {
        if (auto error = generate(stream, records, platformTypeName)) {
            handleAllErrors(std::move(error), [](const llvm::ErrorInfoBase& error) {
                error.log(llvm::WithColor::error());
                llvm::errs() << '\n';
            });
            return true;
        }
        return false;
    };

    const auto platformTypeName = platformTypeMap[vpux::VPU::stringifyArchKind(Platform).str()];

    switch (Action) {
    case Generate:
        return doGenerate(stream, records, platformTypeName);
    default:
        return true;
    }
    return false;
}

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);
    return llvm::TableGenMain(argv[0], &RegGenMain);
}
