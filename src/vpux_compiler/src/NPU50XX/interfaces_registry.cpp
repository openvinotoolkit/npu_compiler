//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/interfaces_registry.hpp"

#include <mlir/IR/DialectRegistry.h>

#include "vpux/compiler/NPU37XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/conversion/passes/VPU2VPUIP/bufferizable_op_interface.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops_interfaces.hpp"

#include "vpux/compiler/ShaveCodeGen/ops_interfaces.hpp"

namespace vpux {

void InterfacesRegistry50XX::registerInterfaces(mlir::DialectRegistry& registry) {
    // NB: arch37xx::ElemTypeInfoOpModel can be re-used for 50XX
    IE::arch37xx::registerElemTypeInfoOpInterfaces(registry);
    // NB: arch40xx ExecutorOpModel is shared for all NPU40XX+ architectures
    IE::arch40xx::registerExecutorOpInterfaces(registry);
    VPU::arch50xx::registerLayerWithPostOpModelInterface(registry);
    // NB: arch37xx::LayoutInfo can be re-used for 50XX
    VPU::arch37xx::registerLayoutInfoOpInterfaces(registry);
    // NB: arch37xx::DDRAccessOpModel can be re-used for 50XX
    VPU::arch37xx::registerDDRAccessOpModelInterface(registry);
    // NB: arch37xx::LayerWithPermuteInterfaceForIE can be re-used for 50XX
    VPU::arch37xx::registerLayerWithPermuteInterfaceForIE(registry);
    VPU::arch50xx::registerUnrollBatchOpInterfaces(registry);
    VPU::arch37xx::registerNCEOpInterface(registry);
    // NB: arch50xx::registerClusterBroadcastingOpInterfaces uses its own logic
    VPU::arch50xx::registerClusterBroadcastingOpInterfaces(registry);
    VPU::arch40xx::registerSCFTilingOpsInterfaces(registry);
    VPUIP::arch50xx::registerAlignedChannelsOpInterfaces(registry);
    // NB: arch40xx::AlignedWorkloadChannelsOp can be re-used for 50XX
    VPUIP::arch40xx::registerAlignedWorkloadChannelsOpInterfaces(registry);
    // NB: arch40xx::BufferizableOp can be re-used for 50XX
    vpux::arch40xx::registerBufferizableOpInterfaces(registry);
    // NB: arch50xx::DPUInvariantExpandOp/DPUVariantExpandOp uses its own logic
    VPUIPDPU::arch50xx::registerDPUExpandOpInterfaces(registry);
    // NB: arch40xx::DPUVariantVerifierOpModel can be re-used for 50XX
    VPUIPDPU::arch40xx::registerVerifiersOpInterfaces(registry);
    VPU::arch50xx::registerICostModelUtilsInterface(registry);
    VPU::arch50xx::registerSWTilingInfoOpInterface(registry);
    ShaveCodeGen::registerShaveCodeGenOpInterfaces(registry);
}

}  // namespace vpux
