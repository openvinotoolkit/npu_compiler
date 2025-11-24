//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux {
namespace config {

constexpr StringRef PIPELINE_OPTIONS = "Options";

vpux::config::PipelineOptionsOp getPipelineOptionsOp(mlir::MLIRContext& ctx, mlir::ModuleOp moduleOp);

// This function returns the value of the attrName constraint from PipelineOptions
template <typename T = size_t>
T getConstraint(mlir::Operation* op, StringRef attrName) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<config::PipelineOptionsOp>(config::PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch constraint");

    auto attrValue = pipelineOptionOp.lookupSymbol<config::OptionOp>(attrName);
    VPUX_THROW_WHEN(attrValue == nullptr, "Failed to find config.OptionOp attribute: {0}", attrName);
    if constexpr (std::is_same_v<T, size_t> || std::is_same_v<T, int64_t>) {
        auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attrValue.getOptionValue());
        VPUX_THROW_WHEN(intAttr == nullptr, "Failed to fetch attr: {0}", attrName);
        // E-174296: Possible loss of accuracy and Conversion of int64_t to unsigned
        return static_cast<T>(intAttr.getValue().getSExtValue());
    } else if constexpr (std::is_same_v<T, double>) {
        auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attrValue.getOptionValue());
        VPUX_THROW_WHEN(floatAttr == nullptr, "Failed to fetch attr: {0}", attrName);
        return floatAttr.getValueAsDouble();
    } else if constexpr (std::is_same_v<T, bool>) {
        auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attrValue.getOptionValue());
        VPUX_THROW_WHEN(boolAttr == nullptr, "Failed to fetch attr: {0}", attrName);
        return boolAttr.getValue();
    } else if constexpr (std::is_same_v<T, llvm::SmallVector<uint32_t>>) {
        auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attrValue.getOptionValue());
        VPUX_THROW_WHEN(arrayAttr == nullptr, "Failed to fetch array attr: {0}", attrName);

        llvm::SmallVector<uint32_t> result;
        result.reserve(arrayAttr.size());
        for (auto elem : arrayAttr) {
            auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(elem);
            VPUX_THROW_WHEN(intAttr == nullptr, "Expected IntegerAttr inside ArrayAttr for: {0}", attrName);
            result.push_back(static_cast<uint32_t>(intAttr.getInt()));
        }
        return result;
    } else {
        // To have T in error message
        static_assert(!sizeof(T), "Unsupported type for constraint");
    }
}

std::optional<bool> tryGetBoolPassOption(mlir::ModuleOp module, StringRef attrName);

template <class T>
mlir::Attribute getAttributeFromOption(mlir::MLIRContext* ctx, mlir::Pass::Option<T>& optionValue) {
    if constexpr (std::is_same_v<T, bool>) {
        return mlir::BoolAttr::get(ctx, optionValue.getValue());
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), optionValue.getValue());
    } else if constexpr (std::is_same_v<T, std::string>) {
        return mlir::StringAttr::get(ctx, optionValue.getValue());
    } else if constexpr (std::is_same_v<T, double>) {
        return mlir::FloatAttr::get(mlir::Float64Type::get(ctx), optionValue.getValue());
    } else if constexpr (std::is_same_v<T, vpux::WeightsTableReuseMode>) {
        return mlir::IntegerAttr::get(getUInt64Type(ctx), static_cast<size_t>(optionValue.getValue()));
    } else {
        // To have T in error message
        static_assert(!sizeof(T), "Unsupported option type for attribute conversion");
    }
}

}  // namespace config
}  // namespace vpux
