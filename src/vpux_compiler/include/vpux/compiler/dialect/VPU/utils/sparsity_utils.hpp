//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/sparsity_constraint.hpp"
#include "vpux/compiler/utils/options.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>

#include <string>

namespace vpux {
namespace VPU {

enum class EnableActivationSparsityMode { AUTO, TRUE, FALSE };

EnableActivationSparsityMode getActSparsityMode(std::string enableActivationSparsityOption);
EnableActivationSparsityMode getActSparsityMode(const StrOption& enableActivationSparsityOption);
bool isActSparsityEnabled(const StrOption& enableActivationSparsityOption);

int64_t getSESize(int64_t channels, const VPU::SparsityConstraint& sparsityConstraint, bool isDepthwise = false);

/*
    Effective sparse output type is the actual tensor IDU sees at its input after applying SETable over the data.

    For example, for a SEAttr with Interpolate Nearest with 2x2 scales, we'll have the following shapes for the sparse
    type components:
    data: [1, 16, 32, 32]
    sparsity_map: [1, 16, 64, 64]
    storage_element_table [1, 1, 64, 64]

    Effective output type will have shape: [1, 16, 64, 64]
*/
mlir::Type getEffectiveSparseOutputType(mlir::Type sparseType);

enum SparsityRemovalFlag {
    Success,
    ClusteredOpInterfaceMissingFail,
    MultiClusterStrategyMissingFail,
    SOKMissingFail,
    SparseOutputMissingFail,
    CatchAllFail
};

SparsityRemovalFlag shouldRemoveOutputSparsity(mlir::Operation* op);

bool isSEOnlyWithoutSMSupported(VPU::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
