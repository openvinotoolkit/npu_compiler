//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"

namespace vpux {
namespace VPU {

constexpr StringRef AUTO_PADDING_ODU = "VPU.AutoPaddingODU";
constexpr StringRef AUTO_PADDING_IDU = "VPU.AutoPaddingIDU";

constexpr std::string_view outChanAttrName = "output_channels";

constexpr int64_t WIDTH16_CHANNEL_LIMIT = 10;
constexpr int64_t FP16_WIDTH = 16;

bool hasAutoPaddingODU(mlir::ModuleOp);
bool hasAutoPaddingIDU(mlir::ModuleOp);
bool inputCompatibleWithAutoPad(vpux::NDTypeInterface);
bool outputCompatibleWithAutoPad(vpux::NDTypeInterface);
bool hasOnlyOutPadding(mlir::ModuleOp);
bool hasOnlyInPadding(mlir::ModuleOp);
bool canAutopadOutput(mlir::Operation*, std::optional<vpux::NDTypeInterface> type = std::nullopt);
bool isODUSupportedOperation(mlir::Operation*);
bool hasOnlyDirectSWConsumers(mlir::Operation*);

}  // namespace VPU
}  // namespace vpux
