//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// Error handling - helper macros to throw exceptions.
//

#pragma once

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <llvm/Support/Compiler.h>

namespace vpux {

//
// Exceptions
//

namespace details {

[[noreturn]] void throwFormat(StringRef file, int line, const std::string& message);

}  // namespace details

#define VPUX_THROW(...) vpux::details::throwFormat(__FILE__, __LINE__, vpux::printToString(__VA_ARGS__))

#define VPUX_THROW_UNLESS(_condition_, ...) \
    if (LLVM_UNLIKELY(!(_condition_)))      \
    VPUX_THROW(__VA_ARGS__)

#define VPUX_THROW_WHEN(_condition_, ...) \
    if (LLVM_UNLIKELY(_condition_))       \
    VPUX_THROW(__VA_ARGS__)

}  // namespace vpux
