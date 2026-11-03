//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/attention_dma_utils.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

namespace {

struct AttentionDMAConfig {
    int64_t qHeadSize;
    int64_t tSL;
    int64_t sSL;
    int64_t eDim;
    int64_t eVDim;
};

const llvm::SmallVector<AttentionDMAConfig> LEGAL_NON_FLASH_ATTENTION_DMA_CONFIGS = {
        {12, 3600, 3600, 16, 16},
};

}  // namespace

bool vpux::VPU::isLegalNonFlashAttentionDMAConfig(int64_t qHeadSize, int64_t tSL, int64_t sSL, int64_t eDim,
                                                  int64_t eVDim) {
    return llvm::any_of(LEGAL_NON_FLASH_ATTENTION_DMA_CONFIGS, [&](const AttentionDMAConfig& config) {
        return config.qHeadSize == qHeadSize && config.tSL == tSL && config.sSL == sSL && config.eDim == eDim &&
               config.eVDim == eVDim;
    });
}
