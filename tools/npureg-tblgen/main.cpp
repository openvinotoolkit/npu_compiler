//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string_view>

#include <llvm/ADT/SmallSet.h>
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
#include <vpux/compiler/dialect/config/enums.hpp.inc>
#include <vpux/compiler/dialect/config/enums.cpp.inc>
#include <vpux/compiler/dialect/VPU/enums.hpp.inc>
#include <vpux/compiler/dialect/VPU/enums.cpp.inc>
// clang-format on

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"

enum ActionType { Generate };

static llvm::cl::opt<ActionType> Action(llvm::cl::desc("Actions to perform"),
                                        llvm::cl::values(clEnumValN(Generate, "generate", "")),
                                        llvm::cl::init(Generate));

static llvm::cl::opt<vpux::config::ArchKind> Platform(
        llvm::cl::desc("Specify the platform type"),
        llvm::cl::values(clEnumValN(vpux::config::ArchKind::NPU40XX, "NPU40XX", "LNL platform")
                         // clang-format off
        , clEnumValN(vpux::config::ArchKind::NPU50XX, "NPU50XX", "PTL platform")
        ), llvm::cl::init(vpux::config::ArchKind::NPU40XX));
// clang-format on

static std::map<std::string, std::string> platformTypeMap{
        {"NPU40XX", "NPUReg40XX"},
        {"NPU50XX", "NPUReg50XX"},
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

// SmallInorderSet behaves like a set
// but traverses elements in order of insertion
struct SmallInorderSet {
    llvm::SmallSet<llvm::StringRef, 8> values;
    llvm::SmallVector<llvm::StringRef, 8> order;

    auto contains(llvm::StringRef value) const {
        return values.contains(value);
    }

    auto empty() const {
        return values.empty();
    }

    auto size() const {
        return values.size();
    }

    auto insert(llvm::StringRef value) {
        const auto [position, inserted] = values.insert(value);
        if (inserted) {
            order.push_back(value);
        }
        return inserted;
    }

    auto begin() const {
        return order.begin();
    }

    auto end() const {
        return order.end();
    }
};

struct Node {
    Node() = default;
    Node(const llvm::Record* record): record(record) {
    }

    const llvm::Record* record = nullptr;
    // Use SmallInorderSet for parents and children
    // to quickly detect duplicated entries
    // and have traversal in order of insertion
    // it's important as affects order of template parameters
    // and eventually order in which descriptors are printed
    // and parsed
    SmallInorderSet parents;
    SmallInorderSet children;
};

using Records = llvm::DenseMap<llvm::StringRef, Node>;

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

llvm::Expected<llvm::raw_ostream&> emitDescriptorsDefinitions(llvm::raw_ostream& stream, const Records& descriptors,
                                                              const std::string& platformTypeName) {
    stream << "namespace vpux::" << platformTypeName << "::Descriptors {\n";
    constexpr llvm::StringLiteral descriptorTemplate =
            "struct {0} : ::vpux::VPURegMapped::detail::DescriptorTemplate<{0}, "
            "::vpux::{1}::detail::Descriptors::{0}Name, {2}> {{};\n";

    for (const auto& [name, descriptor] : descriptors) {
        if (!descriptor.parents.empty()) {
            return llvm::make_error<llvm::StringError>(
                    "Descriptor " + std::string(name) + " shouldn't have any parents", llvm::inconvertibleErrorCode());
        }

        if (descriptor.children.empty()) {
            return llvm::make_error<llvm::StringError>("Descriptor " + std::string(name) + " is empty",
                                                       llvm::inconvertibleErrorCode());
        }

        std::stringstream registersList;
        for (const auto [index, registerName] : llvm::enumerate(descriptor.children)) {
            registersList << "::vpux::" << platformTypeName << "::Registers::" << registerName.str();
            if (index < descriptor.children.size() - 1) {
                registersList << ", ";
            }
        }
        vpux::printTo(stream, descriptorTemplate, name, platformTypeName, registersList.str());
    }
    stream << "}  // namespace vpux::" << platformTypeName << "::Descriptors\n";
    return stream;
}

llvm::Expected<llvm::raw_ostream&> emitRegistersDefinitions(llvm::raw_ostream& stream, const Records& registers,
                                                            const std::string& platformTypeName) {
    stream << "namespace vpux::" << platformTypeName << "::Registers {\n";
    constexpr llvm::StringLiteral registerTemplate =
            "struct {0} : "
            "::vpux::VPURegMapped::detail::RegisterTemplate<{0}, ::vpux::{1}::detail::Registers::{0}Name, "
            "{2}, {3}, {4}> {{};\n";

    for (const auto& [name, reg] : registers) {
        if (reg.parents.empty()) {
            return llvm::make_error<llvm::StringError>(
                    "Register " + std::string(name) + " does not belong to any descriptor",
                    llvm::inconvertibleErrorCode());
        }
        if (reg.children.empty()) {
            return llvm::make_error<llvm::StringError>("Register " + std::string(name) + " is empty",
                                                       llvm::inconvertibleErrorCode());
        }

        std::stringstream fieldsList;
        for (const auto [index, fieldName] : llvm::enumerate(reg.children)) {
            fieldsList << "::vpux::" << platformTypeName << "::Fields::" << fieldName.str();
            if (index < reg.children.size() - 1) {
                fieldsList << ", ";
            }
        }

        const auto offsetVal = getAs<llvm::IntInit>(reg.record, "_offset")->getValue();
        const auto sizeVal = getAs<llvm::IntInit>(reg.record, "_size")->getValue();
        vpux::printTo(stream, registerTemplate, name, platformTypeName, offsetVal, sizeVal, fieldsList.str());
    }
    stream << "}  // namespace vpux::" << platformTypeName << "::Registers\n";
    return stream;
}

llvm::Expected<llvm::raw_ostream&> emitFieldsDefinitions(llvm::raw_ostream& stream, const Records& fields,
                                                         const std::string& platformTypeName) {
    stream << "namespace vpux::" << platformTypeName << "::Fields {\n";
    constexpr llvm::StringLiteral fieldTemplate =
            "struct {0} : ::vpux::VPURegMapped::detail::FieldTemplate<{0}, ::vpux::{1}::detail::Fields::{0}Name, "
            "{2}, {3}, {4}, ::vpux::VPURegMapped::RegFieldDataType::{5}, {6}, {7}, {8}> "
            "{{};\n";

    for (const auto& [name, field] : fields) {
        if (field.parents.empty()) {
            return llvm::make_error<llvm::StringError>(
                    "Field " + std::string(name) + " does not belong to any register", llvm::inconvertibleErrorCode());
        }
        if (!field.children.empty()) {
            return llvm::make_error<llvm::StringError>("Field " + std::string(name) + " shouldn't have any children",
                                                       llvm::inconvertibleErrorCode());
        }

        const auto offsetVal = getAs<llvm::IntInit>(field.record, "_offset")->getValue();
        const auto sizeVal = getAs<llvm::IntInit>(field.record, "_size")->getValue();
        const auto typeVal = getAs<llvm::StringInit>(field.record, "_type")->getValue();
        const auto versionVal = getAs<llvm::DefInit>(field.record, "_version")->getDef();
        const auto major = getAs<llvm::IntInit>(versionVal, "major")->getValue();
        const auto minor = getAs<llvm::IntInit>(versionVal, "minor")->getValue();
        const auto patch = getAs<llvm::IntInit>(versionVal, "patch")->getValue();

        std::stringstream parentRegistersArgument;
        parentRegistersArgument << "std::tuple<";
        for (const auto [index, parentName] : llvm::enumerate(field.parents)) {
            parentRegistersArgument << "::vpux::" << platformTypeName << "::Registers::" << parentName.str();
            if (index < field.parents.size() - 1) {
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

// LLVM 20.x TableGenMain callback signature changed to take a const RecordKeeper&
// This generator does not mutate the RecordKeeper, so we propagate const here.
llvm::Error generate(llvm::raw_ostream& stream, const llvm::RecordKeeper& records,
                     const std::string& platformTypeName) {
    stream << "#pragma once\n";
    stream << '\n';
    stream << "#include <cstdint>\n";
    stream << "#include <string_view>\n";
    stream << '\n';

    Records fields;
    for (auto field : records.getAllDerivedDefinitionsIfDefined(platformTypeName + "_RegFieldWrapper")) {
        const auto fieldName = getAs<llvm::StringInit>(field, "_name")->getValue();
        if (!fields.try_emplace(fieldName, field).second) {
            return llvm::make_error<llvm::StringError>(
                    "Field with name " + std::string(fieldName) + " is defined more than once",
                    llvm::inconvertibleErrorCode());
        }
    }

    Records registers;
    for (auto reg : records.getAllDerivedDefinitionsIfDefined(platformTypeName + "_RegisterWrapper")) {
        auto registerName = getAs<llvm::StringInit>(reg, "_name")->getValue();
        if (registers.contains(registerName)) {
            return llvm::make_error<llvm::StringError>(
                    "Register with name " + std::string(registerName) + " is defined more than once",
                    llvm::inconvertibleErrorCode());
        }
        auto registerNode = Node{reg};

        const auto fieldsListInit = llvm::dyn_cast<llvm::ListInit>(reg->getValue("_fields")->getValue());
        for (auto field : fieldsListInit->getValues()) {
            const auto fieldName = llvm::dyn_cast<llvm::StringInit>(field)->getValue();
            if (!fields.contains(fieldName)) {
                return llvm::make_error<llvm::StringError>(
                        "Register " + std::string(registerName) + " contains unknown field " + std::string(fieldName),
                        llvm::inconvertibleErrorCode());
            }
            if (registerNode.children.contains(fieldName)) {
                return llvm::make_error<llvm::StringError>("Register " + std::string(registerName) +
                                                                   " contains field " + std::string(fieldName) +
                                                                   " more than once",
                                                           llvm::inconvertibleErrorCode());
            }
            if (fields.at(fieldName).parents.contains(registerName)) {
                return llvm::make_error<llvm::StringError>("Register " + std::string(registerName) +
                                                                   " contains field " + std::string(fieldName) +
                                                                   " more than once",
                                                           llvm::inconvertibleErrorCode());
            }

            registerNode.children.insert(fieldName);
            fields[fieldName].parents.insert(registerName);
        }

        registers.try_emplace(std::move(registerName), std::move(registerNode));
    }

    Records descriptors;
    for (auto descriptor : records.getAllDerivedDefinitionsIfDefined(platformTypeName + "_RegMappedWrapper")) {
        auto descriptorName = getAs<llvm::StringInit>(descriptor, "_name")->getValue();
        if (registers.contains(descriptorName)) {
            return llvm::make_error<llvm::StringError>(
                    "Descriptor with name " + std::string(descriptorName) + " is defined more than once",
                    llvm::inconvertibleErrorCode());
        }
        auto descriptorNode = Node{descriptor};

        const auto registersListInit = llvm::dyn_cast<llvm::ListInit>(descriptor->getValue("_registers")->getValue());
        for (auto reg : registersListInit->getValues()) {
            const auto registerName = llvm::dyn_cast<llvm::StringInit>(reg)->getValue();
            if (!registers.contains(registerName)) {
                return llvm::make_error<llvm::StringError>("Descriptor " + std::string(descriptorName) +
                                                                   " contains unknown register " +
                                                                   std::string(registerName),
                                                           llvm::inconvertibleErrorCode());
            }
            if (descriptorNode.children.contains(registerName)) {
                return llvm::make_error<llvm::StringError>("Descriptor " + std::string(descriptorName) +
                                                                   " contains register " + std::string(registerName) +
                                                                   " more than once",
                                                           llvm::inconvertibleErrorCode());
            }
            if (registers.at(registerName).parents.contains(descriptorName)) {
                return llvm::make_error<llvm::StringError>("Descriptor " + std::string(descriptorName) +
                                                                   " contains register " + std::string(registerName) +
                                                                   " more than once",
                                                           llvm::inconvertibleErrorCode());
            }

            descriptorNode.children.insert(registerName);
            registers[registerName].parents.insert(descriptorName);
        }

        descriptors.try_emplace(std::move(descriptorName), std::move(descriptorNode));
    }

    emitNameAsTemplateArgumentWorkAround(stream, fields, registers, descriptors, platformTypeName);
    emitForwardDeclarations(stream, fields, registers, platformTypeName);

    if (auto result = emitDescriptorsDefinitions(stream, descriptors, platformTypeName); !result) {
        return result.takeError();
    }

    if (auto result = emitRegistersDefinitions(stream, registers, platformTypeName); !result) {
        return result.takeError();
    }

    if (auto result = emitFieldsDefinitions(stream, fields, platformTypeName); !result) {
        return result.takeError();
    }

    return llvm::Error::success();
}

bool RegGenMain(llvm::raw_ostream& stream, const llvm::RecordKeeper& records) try {
    // Add a top-level try-catch block to handle std::runtime_error exceptions
    // thrown by the parsing logic, resolving a Coverity finding.
    auto doGenerate = [](llvm::raw_ostream& stream, const llvm::RecordKeeper& records,
                         const std::string& platformTypeName) {
        if (auto error = generate(stream, records, platformTypeName)) {
            handleAllErrors(std::move(error), [](const llvm::ErrorInfoBase& error) {
                error.log(llvm::WithColor::error());
                llvm::errs() << '\n';
            });
            return true;
        }
        return false;
    };

    const auto platformTypeName = platformTypeMap[vpux::config::stringifyArchKind(Platform).str()];

    switch (Action) {
    case Generate:
        return doGenerate(stream, records, platformTypeName);
    default:
        return true;
    }
    return false;
} catch (const std::exception& ex) {
    vpux::printTo(llvm::errs(), "Failed to generate register mappings: {0}\n", ex.what());
    return true;
} catch (...) {
    vpux::printTo(llvm::errs(), "Failed to generate register mappings: unknown exception\n");
    return true;
}

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);
    return llvm::TableGenMain(argv[0], &RegGenMain);
}
