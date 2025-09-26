//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"

namespace vpux {

//
// DefaultHWOptions40XX
//

struct DefaultHWOptions40XX final :
        public IE::arch40xx::DefaultHWOptions,
        VPU::arch40xx::DefaultHWOptions,
        VPUIP::arch40xx::DefaultHWOptions,
        mlir::PassPipelineOptions<DefaultHWOptions40XX> {
    DefaultHWOptions40XX() = default;
    DefaultHWOptions40XX(config::ArchKind arch): PublicOptions(arch) {
    }

    static std::unique_ptr<DefaultHWOptions40XX> createFromString(StringRef options, config::ArchKind arch) {
        auto result = std::make_unique<DefaultHWOptions40XX>(arch);
        if (mlir::failed(result->parseFromString(options))) {
            return nullptr;
        }
        return result;
    }
};

//
// BackendCompilationOptions40XX
//

struct BackendCompilationOptions40XX final : public BackendCompilationOptionsBase<BackendCompilationOptions40XX> {};

//
// BackendCompilationOptions40XX
//

void setupParamsAccordingToOptimizationLevel(int optimizationLevel, DefaultHWOptions40XX& compilationOptions,
                                             bool useWlm);
void setupPWLMParams(DefaultHWOptions40XX& compilationOptions, LogLevel logLevel = LogLevel::None);

}  // namespace vpux
