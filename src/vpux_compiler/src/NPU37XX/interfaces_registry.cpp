//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/interfaces_registry.hpp"
#include <mlir/IR/DialectRegistry.h>

#include "vpux/compiler/NPU37XX/conversion/passes/VPU2VPUIP/bufferizable_op_interface.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops_interfaces.hpp"

namespace vpux {

void InterfacesRegistry37XX::registerInterfaces(mlir::DialectRegistry& registry) {
    IE::arch37xx::registerElemTypeInfoOpInterfaces(registry);
    VPU::arch37xx::registerLayerWithPostOpModelInterface(registry);
    VPU::arch37xx::registerLayoutInfoOpInterfaces(registry);
    VPU::arch37xx::registerDDRAccessOpModelInterface(registry);
    VPU::arch37xx::registerLayerWithPermuteInterfaceForIE(registry);
    VPU::arch37xx::registerUnrollBatchOpInterfaces(registry);
    VPU::arch37xx::registerNCEOpInterface(registry);
    VPU::arch37xx::registerClusterBroadcastingOpInterfaces(registry);
    VPUIP::arch37xx::registerAlignedChannelsOpInterfaces(registry);
    VPUIP::arch37xx::registerAlignedWorkloadChannelsOpInterfaces(registry);
    vpux::arch37xx::registerBufferizableOpInterfaces(registry);
    VPUIPDPU::arch37xx::registerVerifiersOpInterfaces(registry);
    VPU::arch37xx::registerICostModelUtilsInterface(registry);
    VPU::arch37xx::registerSWTilingInfoOpInterface(registry);
}

}  // namespace vpux
