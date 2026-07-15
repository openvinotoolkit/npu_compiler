//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/tools/options.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/Support/CommandLine.h>
#include <mlir/Pass/PassOptions.h>

#include <algorithm>

using namespace vpux;
using namespace std::string_literals;

namespace vpux {

namespace {
constexpr StringRef NPU_PLATFORM_OPTION_NAME = "platform";
constexpr StringRef SEPARATOR = "=";
constexpr StringRef INIT_COMPILER_OPTION = "--init-compiler=";
constexpr StringRef INIT_RESOURCES_OPTION = "--init-resources=";

StringRef extractValueUntilWhitespace(StringRef text, size_t startPos) {
    const auto endIt = std::find_if(text.begin() + startPos, text.end(), [](char c) {
        return std::isspace(static_cast<unsigned char>(c));
    });
    return text.substr(startPos, std::distance(text.begin() + startPos, endIt));
}

StringRef parseParameter(StringRef candidate, StringRef optionName) {
    VPUX_THROW_WHEN(optionName.empty(), "Option name can't be null");

    // Case 1: Direct option like --platform=NPU4000
    const auto separateOption = "--"s + optionName.str() + SEPARATOR.str();
    if (candidate.starts_with(separateOption)) {
        return extractValueUntilWhitespace(candidate, separateOption.size());
    }

    // Case 2: Within --init-compiler="..." or --init-resources="..."
    const bool isInitCompiler = candidate.starts_with(INIT_COMPILER_OPTION);
    if (isInitCompiler || candidate.starts_with(INIT_RESOURCES_OPTION)) {
        // Extract the value part after --init-compiler= or --init-resources=
        const auto prefixSize = isInitCompiler ? INIT_COMPILER_OPTION.size() : INIT_RESOURCES_OPTION.size();
        auto valuesPart = candidate.substr(prefixSize);

        // Remove quotes if present
        if (valuesPart.starts_with("\"")) {
            valuesPart = valuesPart.drop_front(1);
        }
        if (valuesPart.ends_with("\"")) {
            valuesPart = valuesPart.drop_back(1);
        }

        // Search for "optionName=" within the values, ensuring it's a complete token
        const auto optionPattern = optionName.str() + SEPARATOR.str();
        if (valuesPart.starts_with(optionPattern)) {
            return extractValueUntilWhitespace(valuesPart, optionPattern.size());
        }

        // Otherwise, search for " optionName=" (preceded by space)
        const auto spacePattern = " "s + optionPattern;
        const auto pos = valuesPart.find(spacePattern);
        if (pos != StringRef::npos) {
            return extractValueUntilWhitespace(valuesPart, pos + spacePattern.size());
        }
    }

    return {};
}

// Parses '-h' / '-help' and other options that end up printing "usage"
// information (or similar) instead of compiling the IR.
bool isSpecialCaseOption(StringRef candidate) {
    // handle '-' and '--' suffixes (it seems that MLIR allows both cases e.g.
    // -help and --help)
    for (size_t i = 0; i < 2; ++i) {
        if (candidate.starts_with('-')) {
            candidate = candidate.drop_front(1);
        }
    }

    const StringLiteral specialCases[] = {"h", "help", "help-hidden", "help-list", "help-list-hidden", "version"};
    return std::any_of(std::begin(specialCases), std::end(specialCases), [&](const StringLiteral& x) {
        return x == candidate;
    });
}
}  // namespace

std::optional<config::Platform> parsePlatform(int argc, char* argv[]) {
    static llvm::cl::OptionCategory vpuxOptOptions("NPU Options");
    static llvm::cl::opt<std::string> npuPlatformOpt(NPU_PLATFORM_OPTION_NAME,
                                                     llvm::cl::desc("NPU platform to compile for"), llvm::cl::init(""),
                                                     llvm::cl::cat(vpuxOptOptions));
    bool npuPlatformCanBeEmpty = false;
    StringRef npuPlatformString{};
    for (int i = 1; i < argc; ++i) {  // Note: argv[0] is always the program name
        npuPlatformCanBeEmpty = npuPlatformCanBeEmpty || isSpecialCaseOption(argv[i]);
        auto maybeNpuPlatform = parseParameter(argv[i], NPU_PLATFORM_OPTION_NAME);
        if (maybeNpuPlatform.empty()) {
            continue;
        }
        VPUX_THROW_UNLESS(npuPlatformString.empty(), "Platform value is ambiguous and can be set only once.");
        npuPlatformString = maybeNpuPlatform;
    }

    if (npuPlatformString.empty()) {
        VPUX_THROW_UNLESS(npuPlatformCanBeEmpty,
                          "Can't get NPU platform value. Did you forget to specify \"--platform\" or "
                          "\"--init-compiler\" with the \"platform\" parameter?");
        return std::nullopt;
    }

    const auto npuPlatform = config::symbolizeEnum<config::Platform>(npuPlatformString);
    if (!npuPlatform.has_value()) {
        VPUX_THROW_UNLESS(npuPlatformCanBeEmpty, "Invalid platform: '{0}'", npuPlatformString);
        return std::nullopt;
    }
    return npuPlatform.value();
}
}  // namespace vpux
