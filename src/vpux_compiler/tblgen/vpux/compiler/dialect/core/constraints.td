//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0.
//

#ifndef VPUX_COMPILER_CORE_CONSTRAINTS
#define VPUX_COMPILER_CORE_CONSTRAINTS

include "mlir/IR/CommonTypeConstraints.td"

// 5D weights used for Grouped MatMul operation (NCE.MatMul)
class 5DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [5]>;

#endif
