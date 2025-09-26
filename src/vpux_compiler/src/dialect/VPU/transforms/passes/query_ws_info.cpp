//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_QUERYWSINFO
#define GEN_PASS_DEF_QUERYWSINFO
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class QueryWSInfo final : public VPU::impl::QueryWSInfoBase<QueryWSInfo> {
public:
    explicit QueryWSInfo(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    QueryWSInfo(std::optional<Byte> limit, const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());

        if (limit.has_value()) {
            memoryLimit = limit.value().count();
        }
    }

private:
    void safeRunOnModule() final;

    static constexpr vpux::Byte DEFAULT_MEMORY_LIMIT = vpux::Byte(std::numeric_limits<int64_t>::max());
    vpux::Byte getMemoryLimit() const {
        return memoryLimit.hasValue() ? vpux::Byte(memoryLimit.getValue()) : DEFAULT_MEMORY_LIMIT;
    }
};

void QueryWSInfo::safeRunOnModule() {
    const auto& info = getAnalysis<VPU::WeightsSeparationInfo>();
    auto splits = info.getCollectedSplits();
    llvm::sort(splits);
    auto slicedSplits = VPU::sliceAccordingToMemoryLimit(_log, splits, getMemoryLimit());

    auto moduleOp = getOperation();
    moduleOp->setAttr("VPU.WsTotalInitPartCount", getIntAttr(&getContext(), slicedSplits.size()));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createQueryWSInfoPass(const Logger& log) {
    return std::make_unique<QueryWSInfo>(log);
}

std::unique_ptr<mlir::Pass> vpux::VPU::createQueryWSInfoPass(std::optional<Byte> limit, const Logger& log) {
    return std::make_unique<QueryWSInfo>(limit, log);
}
