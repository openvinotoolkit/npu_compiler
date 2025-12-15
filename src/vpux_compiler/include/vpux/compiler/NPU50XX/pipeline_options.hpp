//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU50XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIP/transforms/passes.hpp"

namespace vpux {

//
// DefaultHWOptions50XX
//

struct DefaultHWOptions50XX final :
        public IE::arch50xx::DefaultHWOptions,
        VPU::arch50xx::DefaultHWOptions,
        VPUIP::arch50xx::DefaultHWOptions,
        mlir::PassPipelineOptions<DefaultHWOptions50XX> {
    DefaultHWOptions50XX() = default;
    DefaultHWOptions50XX(config::ArchKind arch): PublicOptions(arch) {
    }

    static std::unique_ptr<DefaultHWOptions50XX> createFromString(StringRef options, config::ArchKind arch) {
        auto result = std::make_unique<DefaultHWOptions50XX>(arch);
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

    mlir::detail::PassOptions::Option<WeightsTableReuseMode> weightsTableReuseMode{
            *this, "weights-table-reuse-mode",
            ::llvm::cl::desc("Option for enabling weights table reuse for different modes."),
            ::llvm::cl::values(clEnumValN(WeightsTableReuseMode::ENABLED, "ENABLED",
                                          "Fully enable weights table reuse for all operations"),
                               clEnumValN(WeightsTableReuseMode::VF_ENABLED, "VF_ENABLED",
                                          "Enable weights table reuse for pure vertical fusion region only"),
                               clEnumValN(WeightsTableReuseMode::DISABLED, "DISABLED",
                                          "Disable weights table reuse for all operations")),
            ::llvm::cl::init(WeightsTableReuseMode::VF_ENABLED)};
};

//
// BackendCompilationOptions50XX
//

struct BackendCompilationOptions50XX final : public BackendCompilationOptionsBase<BackendCompilationOptions50XX> {
    BackendCompilationOptions50XX() {
        enableMemorySideCache = true;
    }

    mlir::detail::PassOptions::Option<VPURegMapped::NPU5PPEBackwardsCompatibilityMode>
            npu5PPEBackwardsCompatibilityMode{
                    *this, "npu5-ppe-backwards-compatibility-mode",
                    ::llvm::cl::desc("NPU5 PPE Backwards Compatibility Mode. In backwards compatible mode (ENABLED), "
                                     "NPU5 PPE HW "
                                     "can use NPU4-style PPE configs."),
                    ::llvm::cl::init(VPURegMapped::NPU5PPEBackwardsCompatibilityMode::DISABLED),
                    ::llvm::cl::values(clEnumValN(VPURegMapped::NPU5PPEBackwardsCompatibilityMode::DISABLED, "DISABLED",
                                                  "NPU5 PPE Backwards Compatibility Mode DISABLED"),
                                       clEnumValN(VPURegMapped::NPU5PPEBackwardsCompatibilityMode::ENABLED, "ENABLED",
                                                  "NPU5 PPE Backwards Compatibility Mode ENABLED"))};
};

void setupPWLMParams50XX(DefaultHWOptions50XX& compilationOptions, LogLevel logLevel = LogLevel::None);

}  // namespace vpux
