//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {

bool DebatcherOptions::isAvailable(const DefaultHWOptionsBase& options) {
    return options.batchCompileMethod == "debatch";
}

std::string DebatcherOptions::getDefaultOptions() {
    std::string ret;
    llvm::raw_string_ostream optionsPrinter(ret);
    DebatcherOptions{}.print(optionsPrinter);
    return ret;
}

std::unique_ptr<DebatcherOptions> DebatcherOptions::create(const DefaultHWOptionsBase& options, Logger log) {
    if (!DebatcherOptions::isAvailable(options)) {
        return {};
    }
    log.info("Compilation proceeds with {0} options value 'debatch', settings: {1}",
             options.batchCompileMethod.getArgStr(), options.debatchCompileMethodSettings);
    VPUX_THROW_UNLESS(options.batchUnrollCompileMethodSettings.empty() ||
                              options.batchUnrollCompileMethodSettings == BatchUnrollOptions::getDefaultOptions(),
                      "DefaultHWOptionsBase is inconsistent: while \"{0}={1}\" was chosen,"
                      " non default value of the mutually exclusive option \"{2}\" has been detected: \"{3}\"",
                      options.batchCompileMethod.getArgStr(), options.batchCompileMethod,
                      options.batchUnrollCompileMethodSettings.getArgStr(), options.batchUnrollCompileMethodSettings);
    std::string settings = options.debatchCompileMethodSettings;
    if (settings.size() >= 2) {
        VPUX_THROW_UNLESS(settings[0] == '{' && settings[settings.size() - 1] == '}',
                          "{0} must be shielded by '{' and '}'", options.batchCompileMethod.getArgStr());
        settings = settings.substr(1, settings.size() - 2);
    }
    return DebatcherOptions::createFromString(settings);
}

std::string DebatcherOptions::to_string() const {
    std::stringstream ss;
    ss << debatcherInliningMethod.getArgStr().data() << ": " << debatcherInliningMethod.getValue();
    return ss.str();
}

bool BatchUnrollOptions::isAvailable(const DefaultHWOptionsBase& options) {
    return options.batchCompileMethod == "unroll";
}

std::string BatchUnrollOptions::getDefaultOptions() {
    std::string ret;
    llvm::raw_string_ostream optionsPrinter(ret);
    BatchUnrollOptions{}.print(optionsPrinter);
    return ret;
}

std::unique_ptr<BatchUnrollOptions> BatchUnrollOptions::create(const DefaultHWOptionsBase& options, Logger log) {
    if (!BatchUnrollOptions::isAvailable(options)) {
        return {};
    }
    log.info("Compilation proceeds with {0} options value 'unroll', settings: {1}",
             options.batchCompileMethod.getArgStr(), options.batchUnrollCompileMethodSettings);
    VPUX_THROW_UNLESS(options.debatchCompileMethodSettings.empty() ||
                              options.debatchCompileMethodSettings == DebatcherOptions::getDefaultOptions(),
                      "DefaultHWOptionsBase is inconsistent: while \"{0}={1}\" was chosen,"
                      " non default value of the mutually exclusive option \"{2}\" has been detected: \"{3}\"",
                      options.batchCompileMethod.getArgStr(), options.batchCompileMethod,
                      options.debatchCompileMethodSettings.getArgStr(), options.debatchCompileMethodSettings);
    std::string settings = options.batchUnrollCompileMethodSettings;
    if (settings.size() >= 2) {
        VPUX_THROW_UNLESS(settings[0] == '{' && settings[settings.size() - 1] == '}',
                          "{0} must be shielded by '{' and '}'", options.batchUnrollCompileMethodSettings.getArgStr());
        settings = settings.substr(1, settings.size() - 2);
    }
    return BatchUnrollOptions::createFromString(settings);
}
}  // namespace vpux
