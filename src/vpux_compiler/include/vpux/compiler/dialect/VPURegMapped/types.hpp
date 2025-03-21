//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include "vpux/utils/core/string_ref.hpp"

#include <llvm/Support/Format.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <vpux_elf/utils/version.hpp>

//
// Hash
//

namespace elf {
llvm::hash_code hash_value(const elf::Version& version);
}  // namespace elf

namespace vpux::VPURegMapped {
struct RegFieldValue {
    RegFieldValue(uint64_t value = 0, elf::Version version = elf::Version()): value(value), version(version){};

    uint64_t value{};
    elf::Version version{};
};
}  // namespace vpux::VPURegMapped

//
// Generated
//

#include <vpux/compiler/dialect/VPURegMapped/enums.hpp.inc>

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPURegMapped/types.hpp.inc>
#undef GET_TYPEDEF_CLASSES

llvm::FormattedNumber getFormattedValue(uint64_t value);
