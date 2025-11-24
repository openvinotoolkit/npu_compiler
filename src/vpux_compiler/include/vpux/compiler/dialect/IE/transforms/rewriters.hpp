//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"

namespace vpux {
namespace IE {

//
// AdjustForVPU Pipeline
//

void registerPerAxisFQConcatRewriters(RewriterRegistry& registry, Logger log = Logger::global());
void registerConvertShuffleChannelsRewriters(RewriterRegistry& registry, Logger log = Logger::global());
void registerMergeTileWithSliceRewriters(RewriterRegistry& registry, Logger log = Logger::global());
void registerConvertLargeConvToMultiConvWithAddRewriters(RewriterRegistry& registry, Logger log = Logger::global());
void registerMergeWeightsSharedConvRewriters(RewriterRegistry& registry, Logger log = Logger::global());
void registerFusePadOpsRewriters(RewriterRegistry& registry, Logger log = Logger::global());
void registerFuseActivationOpsRewriters(RewriterRegistry& registry, bool enableFuseClamp = false,
                                        Logger log = Logger::global());
}  // namespace IE
}  // namespace vpux
