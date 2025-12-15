//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/compilation_options.hpp"
#include "vpux/compiler/NPU37XX/pipeline_options.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"
#include "vpux/compiler/NPU50XX/pipeline_options.hpp"
#include "vpux/compiler/pipelines/options_mapper.hpp"

#include <intel_npu/config/options.hpp>

#include <llvm/Support/raw_ostream.h>

namespace vpux {

template <typename Options>
void checkCompilerOptions(const intel_npu::Config& config) {
    const auto warnForPrivate = !arePrivateOptionsEnabled();
    const auto options = parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(),
                                                             getArchKind(config), warnForPrivate, getLogLevel(config));
    VPUX_THROW_WHEN(options == nullptr, "Failed to parse COMPILATION_MODE_PARAMS");
}

void checkCompilerOptions(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == config::ArchKind::NPU37XX) {
        checkCompilerOptions<DefaultHWOptions37XX>(config);
    } else if (arch == config::ArchKind::NPU40XX) {
        checkCompilerOptions<DefaultHWOptions40XX>(config);
    } else if (arch == config::ArchKind::NPU50XX) {
        checkCompilerOptions<DefaultHWOptions50XX>(config);
    }
}

/// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
/// See https://llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Parse in the next argument from the given options string. Returns a tuple
/// containing [the key of the option, the value of the option, updated
/// `options` string pointing after the parsed option].
// Note: compared to the original MLIR implementation, this does not trim the value that provided to the option (e.g.
// the quote characters are still present for strings)
std::tuple<StringRef, StringRef, StringRef> parseNextArg(StringRef options) {
    // Functor used to extract an argument from 'options' and update it to point
    // after the arg.
    auto extractArgAndUpdateOptions = [&](size_t argSize) {
        StringRef str = options.take_front(argSize).trim();
        options = options.drop_front(argSize).ltrim();
        // Handle escape sequences
        if (str.size() > 2) {
            const auto escapePairs = {std::make_pair('\'', '\''), std::make_pair('"', '"'), std::make_pair('{', '}')};
            for (const auto& escape : escapePairs) {
                if (str.front() == escape.first && str.back() == escape.second) {
                    // Don't process additional escape sequences.
                    break;
                }
            }
        }
        return str;
    };
    // Try to process the given punctuation, properly escaping any contained
    // characters.
    auto tryProcessPunct = [&](size_t& currentPos, char punct) {
        if (options[currentPos] != punct) {
            return false;
        }
        size_t nextIt = options.find_first_of(punct, currentPos + 1);
        if (nextIt != StringRef::npos) {
            currentPos = nextIt;
        }
        return true;
    };

    // Parse the argument name of the option.
    StringRef argName;
    for (size_t argEndIt = 0, optionsE = options.size();; ++argEndIt) {
        // Check for the end of the full option.
        if (argEndIt == optionsE || options[argEndIt] == ' ') {
            argName = extractArgAndUpdateOptions(argEndIt);
            return std::make_tuple(argName, StringRef(), options);
        }

        // Check for the end of the name and the start of the value.
        if (options[argEndIt] == '=') {
            argName = extractArgAndUpdateOptions(argEndIt);
            options = options.drop_front();
            break;
        }
    }

    // Parse the value of the option.
    for (size_t argEndIt = 0, optionsE = options.size();; ++argEndIt) {
        // Handle the end of the options string.
        if (argEndIt == optionsE || options[argEndIt] == ' ') {
            StringRef value = extractArgAndUpdateOptions(argEndIt);
            return std::make_tuple(argName, value, options);
        }

        // Skip over escaped sequences.
        char c = options[argEndIt];
        if (tryProcessPunct(argEndIt, '\'') || tryProcessPunct(argEndIt, '"')) {
            continue;
        }
        // '{...}' is used to specify options to passes, properly escape it so
        // that we don't accidentally split any nested options.
        if (c == '{') {
            size_t braceCount = 1;
            for (++argEndIt; argEndIt != optionsE; ++argEndIt) {
                // Allow nested punctuation.
                if (tryProcessPunct(argEndIt, '\'') || tryProcessPunct(argEndIt, '"')) {
                    continue;
                }
                if (options[argEndIt] == '{') {
                    ++braceCount;
                } else if (options[argEndIt] == '}' && --braceCount == 0) {
                    break;
                }
            }
            // Account for the increment at the top of the loop.
            --argEndIt;
        }
    }
    llvm_unreachable("unexpected control flow in pass option parsing");
}

}  // namespace vpux
