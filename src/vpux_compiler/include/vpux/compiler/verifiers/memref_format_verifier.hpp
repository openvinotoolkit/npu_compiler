//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace mlir {
class Operation;
class PassManager;
}  // namespace mlir

namespace vpux {
namespace verifiers {

void addMemRefFormatVerifier(mlir::PassManager& pm);

}  // namespace verifiers
}  // namespace vpux
