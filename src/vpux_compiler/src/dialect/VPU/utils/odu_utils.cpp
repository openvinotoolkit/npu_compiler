//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/odu_utils.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/error.hpp"

#include <limits>
#include <type_traits>

using namespace vpux;

namespace {

constexpr int64_t NUM_NON_SPATIAL_DIMS = 2;
constexpr int64_t FIRST_SPATIAL_DIM_INDEX = NUM_NON_SPATIAL_DIMS;
constexpr int64_t MIN_ODU_TRANSFORM_RANK = NUM_NON_SPATIAL_DIMS + 1;

int64_t getS2DD2SBlockSizePow(int64_t blockSize, int64_t numSpatialDims) {
    VPUX_THROW_UNLESS(blockSize > 1, "ODU S2D/D2S block size must be positive > 1, got {0}", blockSize);
    VPUX_THROW_UNLESS(numSpatialDims >= 1, "ODU S2D/D2S requires at least one spatial dim, got {0}", numSpatialDims);

    int64_t blkSizePow = 1;
    for (int64_t i = 0; i < numSpatialDims; ++i) {
        VPUX_THROW_UNLESS(blkSizePow <= std::numeric_limits<int64_t>::max() / blockSize,
                          "ODU S2D/D2S block size power overflow: block size {0}, K {1}", blockSize, numSpatialDims);
        blkSizePow *= blockSize;
    }

    return blkSizePow;
}

std::optional<vpux::VPU::ODUS2DD2STransformInfo> getODUS2DD2STransformInfoFromAttr(
        vpux::VPU::S2DD2SConfigAttr s2dd2sConfigAttr, int64_t rank) {
    if (s2dd2sConfigAttr == nullptr) {
        return std::nullopt;
    }

    const auto mode = s2dd2sConfigAttr.getEnable().getValue();
    if (mode == vpux::VPU::S2DD2SEnable::DISABLED) {
        return std::nullopt;
    }

    VPUX_THROW_UNLESS(rank >= MIN_ODU_TRANSFORM_RANK,
                      "ODU S2D/D2S config requires at least a 3-D shape (N, C, spatial...), got rank {0}", rank);

    const auto blockSize = s2dd2sConfigAttr.getBlkSize().getValue().getSExtValue();
    VPUX_THROW_UNLESS(blockSize > 1, "ODU S2D/D2S block size must be positive > 1, got {0}", blockSize);

    const auto blockSizePow = getS2DD2SBlockSizePow(blockSize, rank - NUM_NON_SPATIAL_DIMS);
    return vpux::VPU::ODUS2DD2STransformInfo{mode, blockSize, blockSizePow};
}

}  // namespace

std::optional<vpux::VPU::ODUS2DD2STransformInfo> vpux::VPU::getODUS2DD2STransformInfo(S2DD2SConfigAttr s2dd2sConfigAttr,
                                                                                      int64_t rank) {
    return getODUS2DD2STransformInfoFromAttr(s2dd2sConfigAttr, rank);
}

SmallVector<vpux::VPU::ODUDimScale> vpux::VPU::getODUS2DD2SScaling(const ODUS2DD2STransformInfo& transformInfo,
                                                                   int64_t rank) {
    SmallVector<vpux::VPU::ODUDimScale> scales(rank, {1, 1});

    if (transformInfo.mode == vpux::VPU::S2DD2SEnable::DISABLED) {
        return scales;
    }

    VPUX_THROW_UNLESS(rank >= MIN_ODU_TRANSFORM_RANK, "ODU S2D/D2S scaling requires at least rank {0}, got {1}",
                      MIN_ODU_TRANSFORM_RANK, rank);

    const auto cIdx = vpux::Dims4D::Act::C.ind();
    if (transformInfo.mode == vpux::VPU::S2DD2SEnable::D2S) {
        scales[cIdx] = {1, transformInfo.blockSizePow};
        for (int64_t i{FIRST_SPATIAL_DIM_INDEX}; i < rank; ++i) {
            scales[i] = {transformInfo.blockSize, 1};
        }
    } else {
        // S2D
        scales[cIdx] = {transformInfo.blockSizePow, 1};
        for (int64_t i{FIRST_SPATIAL_DIM_INDEX}; i < rank; ++i) {
            scales[i] = {1, transformInfo.blockSize};
        }
    }

    return scales;
}

bool vpux::VPU::isS2DD2SDimSplitSupported(S2DD2SConfigAttr s2dd2sCfg, Dim outputDim) {
    if (s2dd2sCfg == nullptr) {
        return true;
    }
    const auto mode = s2dd2sCfg.getEnable().getValue();
    if (mode == vpux::VPU::S2DD2SEnable::DISABLED) {
        return true;
    }
    if (outputDim != Dims4D::Act::C) {
        return true;
    }
    // BLOCK_FIRST requires interleaved channel access on the input side to produce a
    // contiguous output — C-dim splits are therefore not supported.
    return s2dd2sCfg.getVariant().getValue() != vpux::VPU::S2DD2SVariant::BLOCK_FIRST;
}

bool vpux::VPU::isODUScalingNeutral(ArrayRef<ODUDimScale> scales) {
    return llvm::all_of(scales, [](const ODUDimScale& s) {
        return s.multiplier == s.divisor;
    });
}

// Returns the primary dimension count for any supported ODU value type.
template <typename T>
static size_t oduRank(const T& val) {
    if constexpr (std::is_same_v<T, SmallVector<int64_t>>) {
        return val.size();
    } else {
        return val.shape.size();
    }
}

// Apply a scale factor to a single integer dimension value.
// forward=true:  out = in * multiplier / divisor  (apply)
// forward=false: out = in * divisor   / multiplier (invert)
// When skipIfDynamic is true, mlir::ShapedType::kDynamic values are left unchanged.
// When loc is absent, failures are returned silently without emitting diagnostics.
static mlir::LogicalResult scaleDim(int64_t& val, const vpux::VPU::ODUDimScale& scale, size_t dimIdx,
                                    StringLiteral field, bool forward, bool skipIfDynamic,
                                    std::optional<mlir::Location> loc) {
    if (skipIfDynamic && val == mlir::ShapedType::kDynamic) {
        return mlir::success();
    }
    const int64_t num = forward ? scale.multiplier : scale.divisor;
    const int64_t den = forward ? scale.divisor : scale.multiplier;
    if (den == 0) {
        if (loc) {
            return errorAt(*loc, "ODU scaling: {0} dim[{1}] has zero {2}", field, dimIdx,
                           forward ? "divisor" : "multiplier");
        }
        return mlir::failure();
    }
    auto prod = val * num;
    if (prod % den != 0) {
        if (loc) {
            return errorAt(*loc, "ODU scaling: {0} dim[{1}] * {2} ({3} * {4}) is not divisible by {5}", field, dimIdx,
                           forward ? "multiplier" : "divisor", val, num, den);
        }
        return mlir::failure();
    }
    val = prod / den;
    return mlir::success();
}

// Core implementation shared by applyODUScaling and invertODUScaling.
// forward=true applies the transform; forward=false inverts it.
template <typename T>
static mlir::FailureOr<T> scaleODU(ArrayRef<vpux::VPU::ODUDimScale> scales, const T& input, bool forward,
                                   std::optional<mlir::Location> loc) {
    static_assert(std::is_same_v<T, SmallVector<int64_t>> || std::is_same_v<T, vpux::ShapeInfo> ||
                          std::is_same_v<T, vpux::TileInfo>,
                  "scaleODU supports only SmallVector<int64_t>, vpux::ShapeInfo, and vpux::TileInfo");
    if (scales.empty()) {
        return input;
    }
    const auto rank = oduRank(input);
    if (scales.size() != rank) {
        if (loc) {
            return errorAt(*loc, "ODU scaling rank mismatch: scales size {0} != rank {1}", scales.size(), rank);
        }
        return mlir::failure();
    }
    T result = input;
    if constexpr (std::is_same_v<T, SmallVector<int64_t>>) {
        for (size_t i = 0; i < scales.size(); ++i) {
            if (mlir::failed(scaleDim(result[i], scales[i], i, "shape", forward, /*skipIfDynamic=*/false, loc))) {
                return mlir::failure();
            }
        }
    } else if constexpr (std::is_same_v<T, vpux::ShapeInfo>) {
        if (!result.bounds.empty() && scales.size() != result.bounds.size()) {
            if (loc) {
                return errorAt(*loc, "ODU scaling rank mismatch: scales size {0} != bounds rank {1}", scales.size(),
                               result.bounds.size());
            }
            return mlir::failure();
        }
        for (size_t i = 0; i < result.shape.size(); ++i) {
            if (mlir::failed(scaleDim(result.shape[i], scales[i], i, "shape", forward, /*skipIfDynamic=*/true, loc))) {
                return mlir::failure();
            }
        }
        for (size_t i = 0; i < result.bounds.size(); ++i) {
            if (mlir::failed(
                        scaleDim(result.bounds[i], scales[i], i, "bounds", forward, /*skipIfDynamic=*/false, loc))) {
                return mlir::failure();
            }
        }
    } else {  // TileInfo
        auto scaleField = [&](vpux::Shape& dims, StringLiteral field) -> mlir::LogicalResult {
            for (size_t i = 0; i < scales.size(); ++i) {
                auto& val = dims[vpux::Dim(i)];
                if (mlir::failed(scaleDim(val, scales[i], i, field, forward, /*skipIfDynamic=*/false, loc))) {
                    return mlir::failure();
                }
            }
            return mlir::success();
        };
        if (mlir::failed(scaleField(result.shape, "shape")) || mlir::failed(scaleField(result.offsets, "offsets"))) {
            return mlir::failure();
        }
    }
    return result;
}

template <typename T>
mlir::FailureOr<T> vpux::VPU::applyODUScaling(ArrayRef<vpux::VPU::ODUDimScale> scales, const T& preODU,
                                              std::optional<mlir::Location> loc) {
    return scaleODU(scales, preODU, /*forward=*/true, loc);
}

template <typename T>
mlir::FailureOr<T> vpux::VPU::invertODUScaling(ArrayRef<vpux::VPU::ODUDimScale> scales, const T& postODU,
                                               std::optional<mlir::Location> loc) {
    return scaleODU(scales, postODU, /*forward=*/false, loc);
}

template mlir::FailureOr<SmallVector<int64_t>> vpux::VPU::applyODUScaling<SmallVector<int64_t>>(
        ArrayRef<vpux::VPU::ODUDimScale>, const SmallVector<int64_t>&, std::optional<mlir::Location>);
template mlir::FailureOr<vpux::ShapeInfo> vpux::VPU::applyODUScaling<vpux::ShapeInfo>(ArrayRef<vpux::VPU::ODUDimScale>,
                                                                                      const vpux::ShapeInfo&,
                                                                                      std::optional<mlir::Location>);
template mlir::FailureOr<vpux::TileInfo> vpux::VPU::applyODUScaling<vpux::TileInfo>(ArrayRef<vpux::VPU::ODUDimScale>,
                                                                                    const vpux::TileInfo&,
                                                                                    std::optional<mlir::Location>);
template mlir::FailureOr<SmallVector<int64_t>> vpux::VPU::invertODUScaling<SmallVector<int64_t>>(
        ArrayRef<vpux::VPU::ODUDimScale>, const SmallVector<int64_t>&, std::optional<mlir::Location>);
template mlir::FailureOr<vpux::ShapeInfo> vpux::VPU::invertODUScaling<vpux::ShapeInfo>(ArrayRef<vpux::VPU::ODUDimScale>,
                                                                                       const vpux::ShapeInfo&,
                                                                                       std::optional<mlir::Location>);
template mlir::FailureOr<vpux::TileInfo> vpux::VPU::invertODUScaling<vpux::TileInfo>(ArrayRef<vpux::VPU::ODUDimScale>,
                                                                                     const vpux::TileInfo&,
                                                                                     std::optional<mlir::Location>);
