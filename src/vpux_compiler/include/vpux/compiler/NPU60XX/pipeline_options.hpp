//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU60XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIP/transforms/passes.hpp"

namespace vpux {
//
// DefaultHWOptions60XX
//

struct DefaultHWOptions60XX final :
        public IE::arch60xx::DefaultHWOptions,
        VPU::arch60xx::DefaultHWOptions,
        VPUIP::arch60xx::DefaultHWOptions,
        mlir::PassPipelineOptions<DefaultHWOptions60XX> {
    DefaultHWOptions60XX() = default;
    DefaultHWOptions60XX(config::ArchKind arch): PublicOptions(arch) {
    }

    static std::unique_ptr<DefaultHWOptions60XX> createFromString(StringRef options, config::ArchKind arch) {
        auto result = std::make_unique<DefaultHWOptions60XX>(arch);
        if (mlir::failed(result->parseFromString(options))) {
            return nullptr;
        }
        return result;
    }

    BoolOption enableAutoPaddingIDU{*this, "enable-auto-padding-idu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(true)};

    BoolOption enableAutoPaddingODU{*this, "enable-auto-padding-odu",
                                    llvm::cl::desc("Enable auto padding for output channels"), llvm::cl::init(true)};

    BoolOption enableIsReduceSupported{*this, "enable-is-reduce-supported",
                                       ::llvm::cl::desc("Enable reduce operations on NCE"), ::llvm::cl::init(false)};
};

//
// BackendCompilationOptions60XX
//

struct BackendCompilationOptions60XX final : public BackendCompilationOptionsBase<BackendCompilationOptions60XX> {
    BackendCompilationOptions60XX() {
        enableMemorySideCache = true;
    }
};

template <typename OptionsT>
void setupPWLMParams60XX(OptionsT& compilationOptions) {
    bool isWorkloadManagementBarrierProgrammingModeSet =
            compilationOptions.workloadManagementBarrierProgrammingMode.hasValue();

    if (!isWorkloadManagementBarrierProgrammingModeSet) {
        switch (compilationOptions.workloadManagementMode) {
        case WorkloadManagementMode::PWLM_V0_1_PAGES:
            compilationOptions.workloadManagementBarrierProgrammingMode =
                    WorkloadManagementBarrierProgrammingMode::LEGACY;
            break;
        case WorkloadManagementMode::FWLM_V1_PAGES:
            compilationOptions.workloadManagementBarrierProgrammingMode =
                    WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED;
            break;
        default:
            compilationOptions.workloadManagementBarrierProgrammingMode =
                    WorkloadManagementBarrierProgrammingMode::UNKNOWN;
            break;
        }
    }
}

template <typename OptionsT>
void setupSparsityOptions60XX(OptionsT& options, const Logger& log) {
    if (options.enableActivationSparsity.hasValue()) {
        log.warning("Activation sparsity is not supported. Setting '{0}' to false",
                    options.enableActivationSparsity.getArgStr());
    }
    if (options.enableWeightsSparsity.hasValue()) {
        log.warning("Overriding option {0} with {1}", options.enableWeightsSparsity.getArgStr(), false);
    }
    options.enableActivationSparsity = "false";
    options.enableWeightsSparsity = false;
}

}  // namespace vpux
