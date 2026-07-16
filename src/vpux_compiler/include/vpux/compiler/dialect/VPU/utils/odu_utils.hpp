//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include <optional>

namespace vpux {
namespace VPU {

struct ODUS2DD2STransformInfo final {
    vpux::VPU::S2DD2SEnable mode;
    int64_t blockSize;
    int64_t blockSizePow;
};

std::optional<ODUS2DD2STransformInfo> getODUS2DD2STransformInfo(S2DD2SConfigAttr s2dd2sConfigAttr, int64_t rank);

/// Per-dimension scaling factor describing how the ODU transform maps a
/// pre-ODU dimension to its post-ODU counterpart.
///
/// The relationship between pre-ODU and post-ODU dimensions is:
///   post_dim = pre_dim * multiplier / divisor
///
/// For dimensions unaffected by the transform both multiplier and divisor are 1.
struct ODUDimScale {
    int64_t multiplier;
    int64_t divisor;
};

/// @brief Return per-dimension ODU scaling factors (pre-ODU → post-ODU) for the
/// given S2D/D2S transform and tensor @p rank. The returned vector has exactly
/// @p rank elements.
///
/// D2S mode (bs = blockSize, K = rank - 2 spatial dims):
///   batch     [0]        : {1,    1      }
///   channel   [1]        : {1,    bs^K   }
///   spatial   [2..rank-1]: {bs,   1      }
///
/// S2D mode:
///   batch     [0]        : {1,    1      }
///   channel   [1]        : {bs^K, 1      }
///   spatial   [2..rank-1]: {1,    bs     }
///
/// DISABLED mode: all elements are {1, 1}.
SmallVector<ODUDimScale> getODUS2DD2SScaling(const ODUS2DD2STransformInfo& transformInfo, int64_t rank);

/// @brief Apply @p scales to @p preODU, returning the scaled value or failure() if any
/// dimension is not exactly scalable.
///
/// Supported types:
///   - SmallVector<int64_t>: flat array of dimension sizes; no kDynamic handling.
///   - vpux::ShapeInfo: both shape and bounds are scaled. Shape field: dynamic dims
///     (mlir::ShapedType::kDynamic) are preserved as-is. Bounds: always scaled.
///   - vpux::TileInfo: shape and offsets are scaled; axis is preserved.
///     Tiles are always static so kDynamic handling is not needed.
///
/// When @p scales is empty, @p preODU is returned unchanged (identity).
/// When @p loc is absent, failures are returned as silent failure() without diagnostics.
template <typename T>
mlir::FailureOr<T> applyODUScaling(ArrayRef<ODUDimScale> scales, const T& preODU,
                                   std::optional<mlir::Location> loc = std::nullopt);

/// @brief Invert @p scales on @p postODU to recover the pre-ODU value, or failure() if
/// any dimension is not exactly invertible.
///
/// Supported types:
///   - SmallVector<int64_t>: flat array of dimension sizes; no kDynamic handling.
///   - vpux::ShapeInfo: both shape and bounds are inverted. Shape field: dynamic dims
///     (mlir::ShapedType::kDynamic) are preserved as-is. Bounds: always inverted.
///   - vpux::TileInfo: shape and offsets are inverted; axis is preserved.
///     Tiles are always static so kDynamic handling is not needed.
///
/// When @p scales is empty, @p postODU is returned unchanged (identity).
/// When @p loc is absent, failures are returned as silent failure() without diagnostics.
template <typename T>
mlir::FailureOr<T> invertODUScaling(ArrayRef<ODUDimScale> scales, const T& postODU,
                                    std::optional<mlir::Location> loc = std::nullopt);

}  // namespace VPU
}  // namespace vpux
