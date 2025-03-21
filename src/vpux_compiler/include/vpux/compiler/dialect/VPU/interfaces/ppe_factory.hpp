//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <memory>
#include <variant>

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux::VPU {
/*!
 * @brief Interface for creating architecture-specific PPE attributes.
 */
class IPpeFactory {
public:
    /*!
     * @brief Generates the complete PPE attribute for the given operation, taking into account potential post ops.
     */
    [[nodiscard]] virtual PPEAttr retrievePPEAttribute(mlir::Operation* operation) const = 0;

    virtual ~IPpeFactory() = default;
};

using PpeIfcPtr = std::unique_ptr<const IPpeFactory>;

/*!
 * @brief Interface for modifying the clamp interval of an existing PPE attribute across different architectures.
 */
struct IPpeAdapterClamp {
    [[nodiscard]] virtual std::pair<double, double> getClamps(vpux::VPU::PPEAttr orig) const = 0;
    [[nodiscard]] virtual vpux::VPU::PPEAttr updateClamps(vpux::VPU::PPEAttr orig, PPEAttr newClamps) const = 0;
    [[nodiscard]] virtual vpux::VPU::PPEAttr intersectClamps(vpux::VPU::PPEAttr orig, double newLow, double newHigh,
                                                             mlir::Type outputElemType) const = 0;

    virtual ~IPpeAdapterClamp() = default;
};

/*!
 * @brief Interface for modifying the scale factor of an existing PPE attribute across different architectures.
 */
struct IPpeAdapterScale {
    [[nodiscard]] virtual SmallVector<double> getScale(vpux::VPU::PPEAttr orig) const = 0;
    [[nodiscard]] virtual vpux::VPU::PPEAttr updateScale(vpux::VPU::PPEAttr orig, ArrayRef<double> scale) const = 0;

    virtual ~IPpeAdapterScale() = default;
};

/*!
 * @brief Interface for modifying the quant scale, shift, post-shift parameters of an existing PPE attribute across
 * different architectures.
 */
struct IPpeAdapterQuantParams {
    [[nodiscard]] virtual vpux::VPU::PPEAttr recomputeQuantParams(PPEAttr orig, mlir::Type inputElemType,
                                                                  mlir::Type outputElemType,
                                                                  ArrayRef<int64_t> kernelShape) const = 0;

    virtual ~IPpeAdapterQuantParams() = default;
};

/*!
 * @brief Interface for modifying the scale factor of an existing PPE attribute across different architectures.
 */
struct IPpeAdapterFpPreluAlpha {
    [[nodiscard]] virtual SmallVector<double> getFpPreluAlpha(vpux::VPU::PPEAttr orig) const = 0;
    [[nodiscard]] virtual vpux::VPU::PPEAttr updateFpPreluAlpha(vpux::VPU::PPEAttr orig,
                                                                ArrayRef<double> fpPreluAlpha) const = 0;

    [[nodiscard]] virtual bool hasQuantScalingThroughPreluAlpha(vpux::VPU::PPEAttr orig) const = 0;
    // Helper which strips (sets to 1.0) the quantization scale of a type if the given PPEAttr applies quantization
    // scale through pReluAlpha. Useful when the type's scale is used for other computations, like generating the scale
    // table.
    [[nodiscard]] mlir::Type adaptTypeForPreluAlphaScaling(vpux::VPU::PPEAttr orig, mlir::Type elemType) const;

    virtual ~IPpeAdapterFpPreluAlpha() = default;
};

/*!
 * @brief Interface for modifying the mode of an existing PPE attribute across different architectures.
 */
struct IPpeAdapterMode {
    [[nodiscard]] virtual vpux::VPU::PPEMode getMode(vpux::VPU::PPEAttr orig) const = 0;
    [[nodiscard]] virtual vpux::VPU::PPEAttr updateMode(vpux::VPU::PPEAttr orig, vpux::VPU::PPEMode mode) const = 0;

    virtual ~IPpeAdapterMode() = default;
};

/*!
 * @brief Interface for switching on/off the weights table usage.
 */
struct IPpeAdapterWeightsTableInfo {
    [[nodiscard]] virtual bool hasWeightsTable(vpux::VPU::PPEAttr orig) const = 0;
    [[nodiscard]] virtual vpux::VPU::PPEAttr discardWeightsTableIfPresent(vpux::VPU::PPEAttr orig,
                                                                          double perTensorScale = 1.0,
                                                                          double perTensorBias = 0.0) const = 0;
    [[nodiscard]] virtual vpux::VPU::PPEAttr useWeightsTable(vpux::VPU::PPEAttr orig) const = 0;

    virtual ~IPpeAdapterWeightsTableInfo() = default;
};

}  // namespace vpux::VPU
