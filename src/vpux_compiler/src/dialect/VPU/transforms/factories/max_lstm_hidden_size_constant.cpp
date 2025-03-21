//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/max_lstm_hidden_size_constant.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/max_lstm_hidden_size_constant.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

constexpr int64_t maxLstmSequenceHiddenSizeConstant = 0;
constexpr int64_t maxLstmCellHiddenSizeConstant = 0;

int64_t VPU::getMaxLstmSequenceHiddenSizeConstant(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX: {
        return maxLstmSequenceHiddenSizeConstant;
    }
    default: {
        return VPU::arch40xx::getMaxLstmSequenceHiddenSizeConstant();
    }
    }
}

int64_t VPU::getMaxLstmCellHiddenSizeConstant(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX: {
        return maxLstmCellHiddenSizeConstant;
    }
    default: {
        return VPU::arch40xx::getMaxLstmCellHiddenSizeConstant();
    }
    }
}
