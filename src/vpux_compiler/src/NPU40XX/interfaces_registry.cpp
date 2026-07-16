//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/interfaces_registry.hpp"

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
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

#include "vpux/compiler/ShaveCodeGen/ops_interfaces.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops_interfaces.hpp"

namespace vpux {

void InterfacesRegistry40XX::registerInterfaces(mlir::DialectRegistry& registry) {
    // NB: arch37xx::ElemTypeInfoOpModel can be re-used for 40XX
    IE::arch37xx::registerElemTypeInfoOpInterfaces(registry);
    // NB: arch40xx uses its own ExecutorOpModel logic
    IE::arch40xx::registerExecutorOpInterfaces(registry);
    // NB: arch37xx::QuantizedLayerOpModel can be re-used for 40XX
    IE::arch37xx::registerQuantizedLayerOpInterfaces(registry);
    // NB: arch37xx::MPEEngineInfoOpModel can be re-used for 40XX
    IE::arch37xx::registerMPEEngineInfoOpInterfaces(registry);
    // NB: arch37xx::SEOpModel can be re-used for 40XX
    IE::arch37xx::registerSEOpInterfaces(registry);
    // NB: arch37xx::AlignedChannelsOpModel can be re-used for 40XX
    IE::arch37xx::registerAlignedChannelsOpInterfaces(registry);
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
    // NB: arch40xx::registerSCFTilingOpsInterfaces uses its own logic
    VPU::arch40xx::registerSCFTilingOpsInterfaces(registry);
    // NB: arch40xx::AlignedWorkloadChannelsOp uses itself logic
    VPUIP::arch40xx::registerAlignedWorkloadChannelsOpInterfaces(registry);
    // NB: arch40xx::BufferizableOp uses its own logic
    vpux::arch40xx::registerBufferizableOpInterfaces(registry);
    // NB: arch40xx::DPUInvariantExpandOp/DPUVariantExpandOp uses its own logic
    VPUIPDPU::arch40xx::registerDPUExpandOpInterfaces(registry);
    // NB: arch40xx::VerifiersOpModel uses its own logic
    VPUIPDPU::arch40xx::registerVerifiersOpInterfaces(registry);
    // NB: arch37xx::ICostModelUtilsInterface can be re-used for 40XX
    VPU::arch37xx::registerICostModelUtilsInterface(registry);
    VPU::arch37xx::registerSWTilingInfoOpInterface(registry);
    // NB: arch37xx::PPECapability can be re-used for 40XX (bias stored as int32)
    VPU::arch37xx::registerPPECapabilityInterface(registry);
    ShaveCodeGen::registerShaveCodeGenOpInterfaces(registry);
    Shave::registerShaveOpInterfaces(registry);
    VPU::arch40xx::registerLayerWithDmaInterface(registry);
    VPU::registerAlignedChannelsOpInterfacesVPU(registry);
}

}  // namespace vpux
