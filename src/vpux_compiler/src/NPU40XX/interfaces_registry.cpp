//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/interfaces_registry.hpp"

#include <mlir/IR/DialectRegistry.h>

#include "vpux/compiler/NPU37XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/conversion/passes/VPU2VPUIP/bufferizable_op_interface.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops_interfaces.hpp"

namespace vpux {

void InterfacesRegistry40XX::registerInterfaces(mlir::DialectRegistry& registry) {
    // NB: arch37xx::ElemTypeInfoOpModel can be re-used for 40XX
    IE::arch37xx::registerElemTypeInfoOpInterfaces(registry);
    // NB: arch37xx::LayerWithPostOpModel can be re-used for 40XX
    VPU::arch37xx::registerLayerWithPostOpModelInterface(registry);
    // NB: arch37xx::LayoutInfo can be re-used for 40XX
    VPU::arch37xx::registerLayoutInfoOpInterfaces(registry);
    // NB: arch37xx::DDRAccessOpModel can be re-used for 40XX
    VPU::arch37xx::registerDDRAccessOpModelInterface(registry);
    // NB: arch37xx::LayerWithPermuteInterfaceForIE can be re-used for 40XX
    VPU::arch37xx::registerLayerWithPermuteInterfaceForIE(registry);
    VPU::arch37xx::registerUnrollBatchOpInterfaces(registry);
    VPU::arch37xx::registerNCEOpInterface(registry);
    // NB: arch40xx::registerClusterBroadcastingOpInterfaces uses its own logic
    VPU::arch40xx::registerClusterBroadcastingOpInterfaces(registry);
    // NB: arch37xx::AlignedChannelsOpModel can be re-used for 40XX
    VPUIP::arch37xx::registerAlignedChannelsOpInterfaces(registry);
    // NB: arch40xx::AlignedWorkloadChannelsOp uses itself logic
    VPUIP::arch40xx::registerAlignedWorkloadChannelsOpInterfaces(registry);
    // NB: arch40xx::BufferizableOp uses its own logic
    vpux::arch40xx::registerBufferizableOpInterfaces(registry);
    // NB: arch40xx::DPUInvariantExpandOp/DPUVariantExpandOp uses its own logic
    VPUIPDPU::arch40xx::registerDPUExpandOpInterfaces(registry);
    // NB: arch40xx::VerifiersOpModel uses its own logic
    VPUIPDPU::arch40xx::registerVerifiersOpInterfaces(registry);
}

}  // namespace vpux
