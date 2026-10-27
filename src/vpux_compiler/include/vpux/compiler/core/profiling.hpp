//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_fwd.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops_fwd.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/profiling/common.hpp"

namespace vpux::VPUIP {
class DMATypeOpInterface;
}  // namespace vpux::VPUIP

namespace vpux {

VPUIP::DpuProfilingMetadataAttr getDpuProfilingMetaAttr(mlir::MLIRContext* ctx, unsigned bufferId, unsigned taskId,
                                                        unsigned maxVariants, unsigned numVariants, unsigned clusterId);

VPUIP::DmaProfilingMetadataAttr getDmaProfilingMetaAttrBegin(mlir::MLIRContext* ctx);

VPUIP::DmaProfilingMetadataAttr getDmaProfilingMetaAttr(mlir::MLIRContext* ctx, unsigned index);

VPUIP::SwProfilingMetadataAttr getSwProfilingMetaAttr(mlir::MLIRContext* ctx, size_t bufferId, size_t bufferOffset,
                                                      size_t clusterSize, size_t dataIndex, size_t tileId,
                                                      size_t clusterId);

enum class DMAProfilingMode { DISABLED, SCRATCH, SW, STATIC_HWP, DYNAMIC_HWP };

DMAProfilingMode getDMAProfilingMode(config::ArchKind arch, const std::string& optionValue);

// Post processing of profiling is relay on uniqueness of locations, but this may be violated. To ensure that all names
// are unique this class is used
class NameUniqifier {
public:
    NameUniqifier(vpux::Logger& log): _log(log) {
    }

    mlir::Location getUniqueLoc(mlir::Location baseLoc) {
        VPUX_THROW_WHEN(mlir::isa<mlir::UnknownLoc>(baseLoc), "Unknown location");
        std::string strLoc = stringifyPrimaryLocation(baseLoc);
        if (_counter.count(strLoc) == 0) {
            _counter[strLoc] = 1;
            return baseLoc;
        }
        _counter[strLoc]++;
        _log.warning("Duplicate location attribute: '{0}'", baseLoc);
        return appendLoc(baseLoc, "Duplicated_{0}", _counter[strLoc]);
    }

private:
    vpux::Logger _log;
    std::map<std::string, size_t> _counter;
};

// Utility structure to save information of individual task that is profiled
template <class InnerTask>
struct TaskSignature {
    InnerTask _task;
    unsigned _maxSubTasks;
    SmallVector<unsigned> _subTasksAtCluster;
};

mlir::BlockArgument addNewProfilingOutput(mlir::MLIRContext* ctx, mlir::func::FuncOp& netFunc,
                                          net::NetworkInfoOp& netOp, mlir::MemRefType outputType,
                                          profiling::ExecutorType execType);

/**
 * @brief utility function to check if a given task belongs to a set of profiled DMA operations,
 * i.e. has DMATypeOpInterface defined
 */
bool isProfiledDmaTask(VPURT::TaskOp taskOp);

/**
 * @brief convenience function to define dma_hwp_id HWP attribute for a given operation
 *
 * @param ctx - mlir context
 * @param op - profiled operation
 * @param dmaHwpId - HWP id value. Set to 0 to ignore the profiling entry.
 */
void setDmaHwpIdAttribute(mlir::MLIRContext* ctx, VPUIP::DMATypeOpInterface& op, int32_t dmaHwpId);
/**
 * @brief check whether HWP argument was used in any profiled DMA operation
 * dma_hwp_id attribute set
 *
 * @return true if at least one profiled DMA operation has dma_hwp_id argument set, false otherwise.
 */
bool isDmaHwpUsedInVPURT(mlir::func::FuncOp& func);
/**
 * @brief check whether HWP argument was used in any profiled DMA operation
 * dma_hwp_id attribute set
 *
 * @return false for architectures < VPUX40XX.
 * For other architectures true if at least one profiled DMA operation has dma_hwp_id argument set,
 * false otherwise.
 *
 */
bool isDmaHwpUsedInVPURT(mlir::ModuleOp& module);

/**
 * @brief check whether NetworkInfoOp defines profiling outputs
 *
 * @return true, if there are profiling outputs defined, false - otherwise
 *
 */
bool isProfilingEnabled(net::NetworkInfoOp netOp);

/**
 * @brief return profiling section of specific type extracted from module's DataInfoOps
 *
 * @return profiling section of type secType, if found, otherwise - empty value.
 *
 */
std::optional<VPUIP::ProfilingSectionOp> getProfilingSection(mlir::ModuleOp module, profiling::ExecutorType secType);
}  // namespace vpux
