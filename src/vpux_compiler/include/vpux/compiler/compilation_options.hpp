//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/ov/config.hpp"

#include <memory>

namespace vpux {

constexpr bool arePrivateOptionsEnabled() {
#ifdef PRIVATE_COMPILER_OPTIONS_ENABLED
    return true;
#else
    return false;
#endif
}

// Ensures that all compilation options given by the user are parsable. An exception will be thrown if invalid options
// are provided. In case the user has provided private options and the project build has these options excluded, a
// warning will be printed for each of them
void checkCompilerOptions(const vpux::OV::Config& config);

// Parse in the next argument from the given options string. Returns a tuple
// containing [the key of the option, the value of the option, updated
// `options` string pointing after the parsed option].
// Note: compared to the original MLIR implementation, this does not trim the value that provided to the option (e.g.
// the quote characters are still present for strings)
std::tuple<StringRef, StringRef, StringRef> parseNextArg(StringRef options);

template <typename T>
std::unique_ptr<T> parseOnlyPublic(StringRef compilationModeParams, config::ArchKind arch, bool warnForPrivate,
                                   LogLevel logLevel) {
    Logger log("options-parser", logLevel);

    auto publicOptions = std::make_unique<PublicOptions>(arch);
    auto privateOptions = std::make_unique<T>(arch);

    // Parse each option as both a private and a public option. In case the option cannot be parsed as a private option,
    // it means it is an invalid option (because public options are a subset of the private options). In case the option
    // cannot be parsed as a public option, an optional warning is printed upon request
    // Note: when `parseFromString` is called, the value of the options structure is changed in-place based on the value
    // of the given option(s), meaning that the private & public structures will be updated with the values provided by
    // the user
    const auto processOption = [&](StringRef key, StringRef value) {
        auto fullOption = llvm::formatv("{0}={1}", key, value).str();
        auto isOptionPrivate = mlir::succeeded(privateOptions->parseFromString(fullOption, llvm::nulls()));
        if (!isOptionPrivate) {
            log.warning("Unsupported compilation option '{0}'", fullOption);
            return mlir::failure();
        }
        auto isOptionPublic = mlir::succeeded(publicOptions->parseFromString(fullOption, llvm::nulls()));
        if (!isOptionPublic && warnForPrivate) {
            log.warning("Compilation option '{0}' is not valid and will be ignored", fullOption);
        }
        return mlir::success();
    };

    while (!compilationModeParams.empty()) {
        StringRef key, value;
        std::tie(key, value, compilationModeParams) = parseNextArg(compilationModeParams);
        if (key.empty()) {
            continue;
        }
        if (mlir::failed(processOption(key, value))) {
            return nullptr;
        }
    }

    return PublicOptions::createFrom<T>(publicOptions);
}

template <typename T>
std::unique_ptr<T> parseCompilationModeParams(StringRef compilationModeParams, config::ArchKind arch,
                                              bool warnForPrivate = false, LogLevel logLevel = LogLevel::None) {
    if (arePrivateOptionsEnabled()) {
        return T::createFromString(compilationModeParams, arch);
    }
    return parseOnlyPublic<T>(compilationModeParams, arch, warnForPrivate, logLevel);
}

}  // namespace vpux
