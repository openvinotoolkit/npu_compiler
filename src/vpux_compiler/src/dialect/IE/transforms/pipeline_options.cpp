//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

namespace vpux {
namespace IE {

DebatcherOpReorderingOptions::DebatcherOpReorderingOptions(const DebatcherOpReorderingOptions& src) {
    *this = src;
}
DebatcherOpReorderingOptions& DebatcherOpReorderingOptions::operator=(const DebatcherOpReorderingOptions& src) {
    overideToTilesPerBatchMode = src.overideToTilesPerBatchMode;
    return *this;
}

bool DebatcherOpReorderingOptions::isAvailable(const DebatcherOptions& options) {
    return options.debatcherInliningMethod.hasValue() && options.debatcherInliningMethod == "reordering";
}

std::string DebatcherOpReorderingOptions::getDefaultOptions() {
    std::string ret;
    llvm::raw_string_ostream optionsPrinter(ret);
    DebatcherOpReorderingOptions{}.print(optionsPrinter);
    return ret;
}

std::unique_ptr<DebatcherOpReorderingOptions> DebatcherOpReorderingOptions::create(const DebatcherOptions& options,
                                                                                   Logger log) {
    if (!DebatcherOpReorderingOptions::isAvailable(options)) {
        return {};
    }

    log.info("Leverage DebatcherOpReorderingOptions as `{0}` has been requested as 'reordering'",
             options.debatcherInliningMethod.getArgStr());
    return std::make_unique<DebatcherOpReorderingOptions>();
}

std::unique_ptr<DebatcherOpReorderingOptions> DebatcherOpReorderingOptions::create(const DefaultHWOptionsBase& options,
                                                                                   Logger log) {
    std::unique_ptr<DebatcherOpReorderingOptions> ret;
    if (auto debatcherOptionsPtr = DebatcherOptions::create(options, log); debatcherOptionsPtr != nullptr) {
        ret = DebatcherOpReorderingOptions::create(*debatcherOptionsPtr, log);
    }
    return ret;
}
}  // namespace IE
}  // namespace vpux
