//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// RAII wrapper for callback to call it at the end of current scope.
//

#pragma once

#include <llvm/ADT/ScopeExit.h>

#include <utility>

namespace vpux {

using llvm::make_scope_exit;

namespace details {

struct ScopeExitTag final {};

template <class Func>
auto operator<<(ScopeExitTag, Func&& func) {
    return make_scope_exit(std::forward<Func>(func));
}

}  // namespace details

#define VPUX_SCOPE_EXIT const auto VPUX_UNIQUE_NAME(scopeExit) = vpux::details::ScopeExitTag{} << [&]()

}  // namespace vpux
