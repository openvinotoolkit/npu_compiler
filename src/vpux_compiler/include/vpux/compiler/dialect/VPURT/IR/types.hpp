//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

//
// BarrierResource
//

namespace vpux {
namespace VPURT {

struct BarrierResource final : public mlir::SideEffects::Resource::Base<BarrierResource> {
    StringRef getName() final {
        return "VPURT::Barrier";
    }
};

}  // namespace VPURT
}  // namespace vpux

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPURT/types.hpp.inc>
#undef GET_TYPEDEF_CLASSES
