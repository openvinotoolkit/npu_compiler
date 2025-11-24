//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/developer_build_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/options.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>
#include <mlir/Transforms/Passes.h>

#include <memory>

namespace vpux {

//
// PublicOptions
//

struct PublicOptions : mlir::PassPipelineOptions<PublicOptions> {
    IntOption optimizationLevel{
            *this, "optimization-level",
            llvm::cl::desc("Set compilation optimization level, enabled starting from NPU4."
                           "Possible values: 0 - optimization for compilation time,"
                           "1 - optimization for execution time (default),"
                           "2 - high optimization for execution time,"
                           "3 - maximize HW utilization, resulting in higher compilation time and memory footprint. "
                           "NOTE: This configuration doesn't guarantee compatibility."),
            llvm::cl::init(1)};

    StrOption performanceHintOverride{*this, "performance-hint-override",
                                      llvm::cl::desc("Set performance hint for compiler to set up number of tiles."
                                                     "Possible values: latency, efficiency (default)"),
                                      llvm::cl::init("efficiency")};

    StrOption computeLayersWithHigherPrecision{
            *this, "compute-layers-with-higher-precision",
            llvm::cl::desc("Enable compute layers with higher precision for the specified layer types"),
            llvm::cl::init("")};

    StrOption enableActivationSparsity{*this, "enable-activation-sparsity",
                                       llvm::cl::desc("Enable activation sparsity"), llvm::cl::init("auto")};
    static std::string getDefaultEnableActivationSparsity(config::ArchKind arch) {
        switch (arch) {
        case config::ArchKind::NPU37XX:
        case config::ArchKind::NPU40XX:
            return "auto";
        default:
            return "false";
        }
    }

    BoolOption enableWeightsSparsity{*this, "enable-weights-sparsity", llvm::cl::desc("Enable weights sparsity"),
                                     llvm::cl::init(true)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations")};
    static bool getDefaultEnableSEPtrsOperations(config::ArchKind arch) {
        switch (arch) {
        case config::ArchKind::NPU40XX:
            return true;
        default:
            return false;
        }
    }

    // TODO: E#159215 Remove this option once external dependencies remove it as well.
    BoolOption enableWDBlockArgumentInput{
            *this, "enable-wd-blockarg-input",
            llvm::cl::desc("This option was deprecated but some external tools pass it, currently it has no effect."),
            llvm::cl::init(true)};

    BoolOption enableSplitBilinerIntoHAndW{*this, "split-bilinear-into-H-and-W",
                                           llvm::cl::desc("Enable split-bilinear-into-H-and-W pass"),
                                           llvm::cl::init(false)};

    BoolOption enableOutputPipelining{*this, "output-pipelining", llvm::cl::desc("Enable output pipelining"),
                                      llvm::cl::init(true)};

    BoolOption enableOutputEnsurance{
            *this, "enable-output-ensurance",
            llvm::cl::desc(
                    "Enable output size ensurance when checking nce op shapes in EnsureNCEOpsSizeRequirements pass"),
            llvm::cl::init(true)};

    BoolOption accumulateMatmulWithDPU{*this, "accumulate-matmul-with-dpu",
                                       llvm::cl::desc("Accumulate unrolled Matmul results with DPU"),
                                       llvm::cl::init(false)};

    BoolOption fuseFQAndMulWithNonConstInput{*this, "fuse-fq-and-mul-with-non-const-input",
                                             llvm::cl::desc("Enable fuse-fq-and-mul pass with non const input"),
                                             llvm::cl::init(false)};

    mlir::detail::PassOptions::Option<WorkloadManagementMode> workloadManagementMode{
            *this, "workload-management-mode",
            ::llvm::cl::desc("Option for enabling WLM enqueue barriers search algorithm at VPURT. To be used only for "
                             "experiments."),
            ::llvm::cl::values(
                    clEnumValN(WorkloadManagementMode::PWLM_V2_PAGES, "PWLM_V2_PAGES",
                               "Partial WLM with split into subgraphs"),
                    clEnumValN(WorkloadManagementMode::PWLM_V1_BARRIER_FIFO, "PWLM_V1_BARRIER_FIFO",
                               "Partial WLM, enqueue barriers search algorithm at VPURT ENABLED"),
                    clEnumValN(WorkloadManagementMode::PWLM_V0_LCA, "PWLM_V0_LCA",
                               "Partial WLM, enqueue barriers search algorithm at VPURT DISABLED. Use LCA based "
                               "enqueue algorithm at VPUMI"))};
    static WorkloadManagementMode getDefaultWorkloadManagementMode(config::ArchKind arch) {
        switch (arch) {
        case config::ArchKind::NPU40XX:
            return WorkloadManagementMode::PWLM_V0_LCA;
        default:
            return WorkloadManagementMode::PWLM_V0_LCA;
        }
    }

    BoolOption enableDPUProfiling{*this, "dpu-profiling", llvm::cl::desc("Enable DPU task profiling"),
                                  llvm::cl::init(true)};

    StrOption enableDMAProfiling{*this, "dma-profiling",
                                 llvm::cl::desc("Enable DMA task profiling (true, false, static)"),
                                 llvm::cl::init("true")};
    static std::string getDefaultEnableDMAProfiling(config::ArchKind arch) {
        switch (arch) {
        case config::ArchKind::NPU40XX:
            // Enable for 40XX once RT will be ready, follow up #E95864
            return "false";
        default:
            return "true";
        }
    }

    BoolOption enableSWProfiling{*this, "sw-profiling", llvm::cl::desc("Enable SW task profiling"),
                                 llvm::cl::init(true)};

    BoolOption enableDumpTaskStats{*this, "dump-task-stats",
                                   ::llvm::cl::desc("Enable dumping statistics of Task operations"),
                                   ::llvm::cl::init(vpux::isDeveloperBuild())};

    BoolOption enableScheduleTrace{*this, "enable-schedule-trace",
                                   llvm::cl::desc("Enable compile time schedule analysis and trace"),
                                   llvm::cl::init(false)};

    StrOption dpuDryRun{*this, "dpu-dry-run",
                        llvm::cl::desc("Patch DPU tasks to disable their functionality (none|stub|strip)"),
                        llvm::cl::init("none")};

    BoolOption shaveDryRun{*this, "shave-dry-run", llvm::cl::desc("Enable shave dry run stripping"),
                           llvm::cl::init(false)};

    //
    // Constructors
    //

    PublicOptions() = default;
    PublicOptions(config::ArchKind arch) {
        enableActivationSparsity.setValue(getDefaultEnableActivationSparsity(arch));
        enableSEPtrsOperations.setValue(getDefaultEnableSEPtrsOperations(arch));
        if (arch != config::ArchKind::NPU40XX) {
            workloadManagementMode.setValue(getDefaultWorkloadManagementMode(arch));
        }
        enableDMAProfiling.setValue(getDefaultEnableDMAProfiling(arch));
    }

    static std::unique_ptr<PublicOptions> createFromString(StringRef options, config::ArchKind arch) {
        auto result = std::make_unique<PublicOptions>(arch);
        if (mlir::failed(result->parseFromString(options))) {
            return nullptr;
        }
        return result;
    }

    template <typename T>
    static std::unique_ptr<T> createFrom(const std::unique_ptr<PublicOptions>& publicOptions) {
        auto options = std::make_unique<T>();
        if (publicOptions != nullptr) {
            options->matchAndCopyOptionValuesFrom(*publicOptions);
        }
        return options;
    }
};

}  // namespace vpux
