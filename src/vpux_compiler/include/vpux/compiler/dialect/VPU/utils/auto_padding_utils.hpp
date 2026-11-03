//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/dpu_fwd.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <optional>

namespace vpux {
namespace VPU {

class NCEOpInterface;

constexpr std::string_view INPUT_PADDING_ATTR_NAME = "input_padding";
constexpr std::string_view OUTPUT_PADDING_ATTR_NAME = "output_padding";

// Hardware limitation for NPU5
constexpr int64_t WIDTH16_CHANNEL_LIMIT = 10;
constexpr int64_t FP16_WIDTH = 16;

bool areChannelsCompatibleWithIDUAutoPad(int64_t inputChannels, int64_t elemTypeBitWidth);
bool areChannelsCompatibleWithODUAutoPad(int64_t outputChannels, int64_t elemTypeBitWidth);
bool inputCompatibleWithAutoPad(vpux::NDTypeInterface);

/// Shape info needed by outputCompatibleWithAutoPad()/canAutopadOutput() when the caller
/// already has them and constructing a full NDTypeInterface is undesirable.
struct AutoPadPreS2DD2SInfo {
    int64_t rank;
    int64_t channels;
    int64_t elemTypeBitWidth;
};

bool outputCompatibleWithAutoPad(mlir::Operation* op, vpux::NDTypeInterface type);
bool outputCompatibleWithAutoPad(const AutoPadPreS2DD2SInfo& info);
bool canAutopadInput(mlir::Operation*);
bool canAutopadOutput(mlir::Operation*, std::optional<vpux::NDTypeInterface> type = std::nullopt);
bool canAutopadOutput(mlir::Operation*, const AutoPadPreS2DD2SInfo& info);
bool canConsumeIDUAutopad(VPU::NCEConvolutionOp nceConvOp, LogCb logCb = emptyLogCb);
bool canConsumeODUAutopad(VPU::NCEOpInterface nceOp, LogCb logCb = emptyLogCb);
std::optional<int64_t> getWeightsChannelsAutopad(mlir::Operation* op,
                                                 std::optional<vpux::NDTypeInterface> optType = std::nullopt);
std::optional<int64_t> getWeightsChannelsAutopad(mlir::Operation* op, const AutoPadPreS2DD2SInfo& info);

}  // namespace VPU
}  // namespace vpux
