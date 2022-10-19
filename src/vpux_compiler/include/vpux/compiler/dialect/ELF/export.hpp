//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/preprocessing.hpp"
#include "vpux_compiler.hpp"

#include <vpux_elf/writer.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

#include <transformations/utils/utils.hpp>

#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {
namespace ELF {

std::vector<uint8_t> exportToELF(
        mlir::ModuleOp module,
        const std::vector<vpux::PreProcessInfo>& preprocessInfo = std::vector<vpux::PreProcessInfo>(),
        const std::vector<std::shared_ptr<const ov::Node>>& parameters = std::vector<std::shared_ptr<const ov::Node>>(),
        const std::vector<std::shared_ptr<const ov::Node>>& results = std::vector<std::shared_ptr<const ov::Node>>(),
        Logger log = Logger::global());

}  // namespace ELF
}  // namespace vpux
