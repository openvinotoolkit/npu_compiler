//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/ppe_factory.hpp"

namespace vpux::VPU::arch50xx {

/*!
 * @brief Interface for creating NPU50 Float PPE attributes.
 */
class PpeFactory final :
        public vpux::VPU::IPpeFactory,
        public vpux::VPU::IPpeAdapterClamp,
        public vpux::VPU::IPpeAdapterScaleBias,
        public vpux::VPU::IPpeAdapterFpPreluAlpha,
        public vpux::VPU::IPpeAdapterMode,
        public vpux::VPU::IPpeAdapterWeightsTableInfo {
    class AttrBuilder {
        // Helper class for handling PPE fields prior to instancing mlir attributes.
    private:
        mlir::MLIRContext* _ctx;

    public:
        PPEMode mode = PPEMode::NOOP;
        float clampLow = std::numeric_limits<float>::lowest();
        float clampHigh = std::numeric_limits<float>::max();
        std::optional<double> scale;
        SmallVector<double> pReluAlpha = {1.0};
        std::optional<double> bias;
        double adder = 0.0;
        std::optional<SmallVector<double>> in1Mult;
        std::optional<SmallVector<double>> in2Mult;
        std::optional<std::vector<uint16_t>> sprLUT;

        AttrBuilder(mlir::MLIRContext* ctx);
        AttrBuilder(const AttrBuilder&) = default;
        AttrBuilder(AttrBuilder&&) noexcept = default;
        ~AttrBuilder() = default;

        AttrBuilder& operator=(const AttrBuilder&) = default;
        AttrBuilder& operator=(AttrBuilder&&) noexcept = default;

        [[nodiscard]] PPEFpAttr getAttr() const;
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
    // @brief Clears clamp values and sets them to limits of dtype
    [[nodiscard]] vpux::VPU::PPEAttr discardClamp(vpux::VPU::PPEAttr orig, mlir::Type outputElemType) const override;

    // --- IPpeAdapterScaleBias Implementation ---

    // @brief Returns the scale factor of the PPE Attribute.
    [[nodiscard]] std::optional<SmallVector<double>> getScale(vpux::VPU::PPEAttr orig) const override;
    // @brief Returns the bias of the PPE Attribute.
    [[nodiscard]] std::optional<double> getBias(vpux::VPU::PPEAttr orig) const override;
    // @brief Modifies the scale factor of the PPE Attribute.
    [[nodiscard]] vpux::VPU::PPEAttr updateScale(vpux::VPU::PPEAttr orig, ArrayRef<double> scale) const override;
    // @brief Sets new per-tensor bias overriding the existing data.
    [[nodiscard]] vpux::VPU::PPEAttr updateBias(vpux::VPU::PPEAttr orig, double perTensorBias) const override;
    // @brief Removes scale and bias from a ppe attr
    [[nodiscard]] vpux::VPU::PPEAttr discardScaleBias(vpux::VPU::PPEAttr orig) const override;

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

    // --- IPpeAdapterWeightsTableInfo Implementation ---

    // @brief Checks if a weights table is used to store the scale and bias of the coresponding operation.
    [[nodiscard]] bool hasWeightsTable(vpux::VPU::PPEAttr orig) const override;
    // @brief Sets new per-tensor scale and bias overriding the existing WT data. If the scale and bias are already
    // per-tensor their original value is kept.
    [[nodiscard]] vpux::VPU::PPEAttr discardWeightsTableIfPresent(vpux::VPU::PPEAttr orig, double perTensorScale,
                                                                  double perTensorBias) const override;
    // @brief Sets the scale and bias to null inside the PPE attribute, indicating that these must be retrieved from the
    // WT.
    [[nodiscard]] vpux::VPU::PPEAttr useWeightsTable(vpux::VPU::PPEAttr orig) const override;

private:
    // casts opaque PPE attributes to the type expected by this factory
    vpux::VPU::PPEFpAttr castToConcreteAttr(PPEAttr ppeAttr) const;

    // build attribute for Eltwise ops: Add, Subtract, Multiply
    AttrBuilder retrieveEltwisePPEAttribute(mlir::Operation* operation) const;
    // build attribute for Non-Eltwise ops: MaxPool, AvgPool, Convolution
    AttrBuilder retrieveNonEltwisePPEAttribute(mlir::Operation* operation) const;

    // callbacks for handling post-operations
    void callbackDefault(mlir::Operation* operation, AttrBuilder& builder) const;
    template <typename PostOpAttr>
    void callback(vpux::IE::LayerWithPostOpInterface, PostOpAttr, AttrBuilder&) const = delete;

    void fillSprLookupTable(std::vector<uint16_t>& sprLUTData,
                            FuncRef<void(std::vector<uint16_t>&)> fillSaturationTable,
                            FuncRef<void(std::vector<uint16_t>&)> fillLutCfg,
                            FuncRef<void(std::vector<uint16_t>&)> fillSlopeIntercept, uint16_t specialConfig) const;
};

}  // namespace vpux::VPU::arch50xx
