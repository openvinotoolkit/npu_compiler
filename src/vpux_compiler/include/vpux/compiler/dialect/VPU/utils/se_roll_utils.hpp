//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

namespace vpux::VPU {

bool isSupportedSEPRoll(IE::RollOp op, vpux::LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                        bool supportsInputActCompression = false);
bool isSupportedSEPRoll(VPU::RollOp op, vpux::LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                        bool supportsInputActCompression = false);

DimArr getRollSEPConvTilingOrder(VPU::SERollAttr seAttr);
bool isRollSEPConvCompatibleWithClusterStrategy(VPU::SERollAttr seAttr, VPU::MultiClusterStrategy strategy);

}  // namespace vpux::VPU
