//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
namespace vpux::VPURegMapped {
#define GEN_PASS_DECL_DEDUCEDYNAMICMAPPEDINFERENCEVERSION
#define GEN_PASS_DEF_DEDUCEDYNAMICMAPPEDINFERENCEVERSION
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp.inc"
}  // namespace vpux::VPURegMapped

using namespace vpux;

namespace {

class DeduceDynamicMappedInferenceVersion :
        public VPURegMapped::impl::DeduceDynamicMappedInferenceVersionBase<DeduceDynamicMappedInferenceVersion> {
public:
    explicit DeduceDynamicMappedInferenceVersion(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    Logger _log;
};

void DeduceDynamicMappedInferenceVersion::safeRunOnModule() {
    auto moduleOp = getOperation();
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp netInfo;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, netFunc);

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());

    auto elfMain = mainOps[0];
    auto getVersionFromOps = [](ELF::MainOp main) -> elf::Version {
        elf::Version maxVersion;
        for (auto dataSection : main.getOps<ELF::DataSectionOp>()) {
            auto regFieldOps = dataSection.getBlock()->getOps<VPURegMapped::NPURegDescriptorOpInterface>();

            for (auto regFieldOp : regFieldOps) {
                const auto version = regFieldOp.getVersion();
                maxVersion = std::max(maxVersion, version);
            }
        }
        VPUX_THROW_UNLESS(maxVersion.checkValidity(), "There are no NPURegDescriptorOpInterface ops found.");
        return maxVersion;
    };

    auto newVersion = getVersionFromOps(elfMain);

    auto setNewVersion = [&](ELF::MainOp main) -> void {
        for (auto dataSection : main.getOps<ELF::DataSectionOp>()) {
            auto miVersionOps = dataSection.getBlock()->getOps<VPURegMapped::SerializedVersionInterface>();
            for (auto op : miVersionOps) {
                op.setVersion(newVersion);
            }
        }
    };

    setNewVersion(elfMain);
}

}  // namespace

//
// createDeduceMappedInferenceVersion
//

std::unique_ptr<mlir::Pass> VPURegMapped::createDeduceDynamicMappedInferenceVersionPass(Logger log) {
    return std::make_unique<DeduceDynamicMappedInferenceVersion>(log);
}
