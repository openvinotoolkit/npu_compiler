//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/interfaces_registry.hpp"

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
#include "vpux/compiler/NPU50XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/NPU60XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops_interfaces.hpp"

#include "vpux/compiler/ShaveCodeGen/ops_interfaces.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

namespace vpux {

void InterfacesRegistry60XX::registerInterfaces(mlir::DialectRegistry& registry) {
    IE::arch60xx::registerElemTypeInfoOpInterfaces(registry);
    // NB: arch40xx ExecutorOpModel is shared for all NPU40XX+ architectures
    IE::arch40xx::registerExecutorOpInterfaces(registry);
    // NB: arch60xx::QuantizedLayerOpModel has NPU6-specific input type constraints
    IE::arch60xx::registerQuantizedLayerOpInterfaces(registry);
    IE::arch60xx::registerMPEEngineInfoOpInterfaces(registry);
    // NB: arch37xx::SEOpModel can be re-used for 60XX
    IE::arch37xx::registerSEOpInterfaces(registry);
    // NB: arch50xx::registerAlignedChannelsOpInterfaces can be re-used for 60XX
    IE::arch50xx::registerAlignedChannelsOpInterfaces(registry);
    VPU::arch50xx::registerLayerWithPostOpModelInterface(registry);
    VPU::arch60xx::registerLayerWithDmaInterface(registry);
    // NB: arch37xx::LayoutInfo can be re-used for 60XX
    VPU::arch37xx::registerLayoutInfoOpInterfaces(registry);
    // NB: arch37xx::DDRAccessOpModel can be re-used for 60XX
    VPU::arch37xx::registerDDRAccessOpModelInterface(registry);
    VPU::arch60xx::registerLayerWithPermuteInterfaceForIE(registry);
    VPU::arch50xx::registerUnrollBatchOpInterfaces(registry);
    VPU::arch60xx::registerNCEOpInterface(registry);
    VPU::arch40xx::registerSCFTilingOpsInterfaces(registry);
    // NB: arch60xx::registerClusterBroadcastingOpInterfaces uses its own logic
    VPU::arch60xx::registerClusterBroadcastingOpInterfaces(registry);
    // NB: arch40xx::AlignedWorkloadChannelsOp can be re-used for 60XX
    VPUIP::arch40xx::registerAlignedWorkloadChannelsOpInterfaces(registry);
    // NB: arch40xx::BufferizableOp can be re-used for 60XX
    vpux::arch40xx::registerBufferizableOpInterfaces(registry);
    // NB: arch60xx::DPUExpandOpInterfaces
    VPUIPDPU::arch60xx::registerDPUExpandOpInterfaces(registry);
    // NB: arch60xx::VerifiersOpInterfaces
    VPUIPDPU::arch60xx::registerVerifiersOpInterfaces(registry);
    // NB: arch50xx::ICostModelUtilsInterface can be re-used for 60XX
    VPU::arch50xx::registerICostModelUtilsInterface(registry);
    VPU::arch50xx::registerSWTilingInfoOpInterface(registry);
    // NB: arch50xx::PPECapability can be re-used for 60XX (bias stored as f32)
    VPU::arch50xx::registerPPECapabilityInterface(registry);
    ShaveCodeGen::registerShaveCodeGenOpInterfaces(registry);
    Shave::registerShaveOpInterfaces(registry);
    VPU::registerAlignedChannelsOpInterfacesVPU(registry);
}

}  // namespace vpux
