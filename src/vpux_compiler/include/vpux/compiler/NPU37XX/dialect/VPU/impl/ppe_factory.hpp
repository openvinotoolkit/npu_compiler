//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/ppe_factory.hpp"

namespace vpux::VPU::arch37xx {

/*!
 * @brief Interface for creating NPU37/NPU40 Integer PPE attributes.
 */
class PpeFactory final :
        public vpux::VPU::IPpeFactory,
        public vpux::VPU::IPpeAdapterClamp,
        public vpux::VPU::IPpeAdapterScale,
        public vpux::VPU::IPpeAdapterQuantParams,
        public vpux::VPU::IPpeAdapterFpPreluAlpha,
        public vpux::VPU::IPpeAdapterMode {
    class AttrBuilder {
        // Helper class for handling PPE fields prior to instancing mlir attributes.
    private:
        mlir::MLIRContext* _ctx;

    public:
        PPEMode mode = PPEMode::NOOP;
        int32_t clampLow = std::numeric_limits<int32_t>::min();
        int32_t clampHigh = std::numeric_limits<int32_t>::max();
        int32_t lReluMult = 1;
        uint32_t lReluShift = 0;
        std::optional<SmallVector<double>> quantScale;
        std::optional<SmallVector<int64_t>> quantMult;
        std::optional<SmallVector<int64_t>> quantShift;
        std::optional<int64_t> quantPostShift;
        std::optional<SmallVector<int64_t>> in1QuantMult;
        std::optional<SmallVector<int64_t>> in2QuantMult;
        float fpPReluAlpha = 1.0f;

        AttrBuilder(mlir::MLIRContext* ctx);
        AttrBuilder(const AttrBuilder&) = default;
        AttrBuilder(AttrBuilder&&) noexcept = default;
        ~AttrBuilder() = default;

        AttrBuilder& operator=(AttrBuilder&) = default;
        AttrBuilder& operator=(AttrBuilder&&) noexcept = default;

        [[nodiscard]] PPEIntAttr getAttr() const;
    };

public:
    PpeFactory() = default;

    // --- IPpeFactory Implementation ---

    // @brief Generates the complete PPE attribute for the given operation, taking into account potential post ops and
    // quantization.
    [[nodiscard]] vpux::VPU::PPEAttr retrievePPEAttribute(mlir::Operation* operation) const override;

    // --- IPpeAdapterClamp Implementation ---

    // @brief Returns the clamp interval of the PPE attribute as a pair of (clamp_low, clamp_high).
    [[nodiscard]] std::pair<double, double> getClamps(vpux::VPU::PPEAttr orig) const override;
    // @brief Replaces the clamp interval of the original PPE Attribute with the clamps of another PPE Attribute.
    [[nodiscard]] vpux::VPU::PPEAttr updateClamps(vpux::VPU::PPEAttr orig, PPEAttr newClamps) const override;
    // @brief Sets the clamp interval to the intersection between the original clamps and a given interval.
    [[nodiscard]] vpux::VPU::PPEAttr intersectClamps(vpux::VPU::PPEAttr orig, double newLow, double newHigh,
                                                     mlir::Type outputElemType) const override;

    // --- IPpeAdapterScale Implementation ---

    // @brief Returns the scale factor of the PPE Attribute.
    [[nodiscard]] SmallVector<double> getScale(vpux::VPU::PPEAttr orig) const override;
    // @brief Modifies the scale factor of the PPE Attribute.
    [[nodiscard]] vpux::VPU::PPEAttr updateScale(vpux::VPU::PPEAttr orig, ArrayRef<double> scale) const override;

    // --- IPpeAdapterFpPreluAlpha Implementation ---

    // @brief Returns the fpPreluAlpha of the PPE Attribute.
    [[nodiscard]] SmallVector<double> getFpPreluAlpha(vpux::VPU::PPEAttr orig) const override;
    // @brief Modifies the fpPreluAlpha of the PPE Attribute.
    [[nodiscard]] vpux::VPU::PPEAttr updateFpPreluAlpha(vpux::VPU::PPEAttr orig,
                                                        ArrayRef<double> fpPreluAlpha) const override;
    // @brief Checks if the fpPreluAlpha is used to apply quantization scaling through the fpPreluAlpha field.
    [[nodiscard]] bool hasQuantScalingThroughPreluAlpha(vpux::VPU::PPEAttr orig) const override;

    // --- IPpeAdapterMode Implementation ---

    // @brief Returns the mode of the PPE Attribute.
    [[nodiscard]] vpux::VPU::PPEMode getMode(vpux::VPU::PPEAttr orig) const override;
    // @brief Modifies the mode of the PPE Attribute.
    [[nodiscard]] vpux::VPU::PPEAttr updateMode(vpux::VPU::PPEAttr orig, vpux::VPU::PPEMode mode) const override;

    // --- IPpeAdapterQuantParams Implementation ---

    // @brief Recomputes the approximated quantization parameters of the PPE Attribute. Given the shape of the kernel,
    // AveragePoolOp's can be emulated through the same parameters.
    [[nodiscard]] vpux::VPU::PPEAttr recomputeQuantParams(PPEAttr orig, mlir::Type inputElemType,
                                                          mlir::Type outputElemType,
                                                          ArrayRef<int64_t> kernelShape) const override;

private:  // methods
    // casts opaque PPE attributes to the type expected by this factory
    vpux::VPU::PPEIntAttr castToConcreteAttr(PPEAttr ppeAttr) const;

    void applyStaticScale(mlir::Operation* op, AttrBuilder& builder) const;
    void configureAttrForAvgPool(mlir::Operation* operation, AttrBuilder& builder) const;
    void calculateFpPReluAlpha(mlir::Operation* operation, PpeFactory::AttrBuilder& builder) const;

    // build attribute for Eltwise ops: Add
    AttrBuilder retrieveEltwisePPEAttribute(mlir::Operation* operation) const;
    // build attribute for Non-Eltwise ops: MaxPool, AvgPool, Convolution
    AttrBuilder retrieveNonEltwisePPEAttribute(mlir::Operation* operation) const;

    // callbacks for handling post-operations
    AttrBuilder callbackReluOp(vpux::IE::LayerWithPostOpInterface operation) const;
    AttrBuilder callbackClampOp(vpux::IE::LayerWithPostOpInterface operation) const;
    AttrBuilder callbackLeakyReluOp(vpux::IE::LayerWithPostOpInterface operation) const;
};

}  // namespace vpux::VPU::arch37xx
