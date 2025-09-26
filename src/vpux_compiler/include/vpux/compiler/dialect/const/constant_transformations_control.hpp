//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>

#include <functional>
#include <utility>

namespace vpux::Const {

namespace details::optimization {
using TransformAttrPos = SmallVector<Const::TransformAttrInterface>::iterator;
}

// E#150913: This options structure is a workaround in reality. Ideally,
// constant transformations must be "optimal" IR-wise regardless of the intended
// usages (because optimal IR structure roughly guarantees optimal compile-time
// execution). Currently, this is impossible due to the constant folding design
// (e.g. folding SubView as the first transformations gives best performance).

//! @brief Controls certain aspects of lazy constant folding in the compiler.
struct LazyFoldingOptions {
    //! @brief Constructs default options.
    LazyFoldingOptions();

    using OptimizeConstTransformationsFunc = std::function<std::pair<details::optimization::TransformAttrPos, bool>(
            SmallVector<TransformAttrInterface>&, details::optimization::TransformAttrPos&)>;

    //! @brief Returns optimizations that are applied to constant
    //! transformations.
    std::function<SmallVector<OptimizeConstTransformationsFunc>(mlir::Type)> getFoldingSequenceOptimizations;
};

/** @brief Sets lazy constant folding behavior according to the options.

    @note By default, default-constructed lazy folding options are already set
    into the context.
*/
void setLazyFoldingOptions(mlir::MLIRContext* ctx, const LazyFoldingOptions& options);

//! @brief Returns lazy constant folding options for the Weights separation pipeline.
LazyFoldingOptions getWsFoldingOptions();

}  // namespace vpux::Const
