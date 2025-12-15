//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/managed_mapped_inference_rewriter.hpp"

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace NPUReg50XX;
using namespace NPUReg50XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg50xx {

mlir::LogicalResult ManagedMappedInferenceRewriter::matchAndRewrite(VPUASM::ManagedMappedInferenceOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    VpuManagedMappedInference managedMappedInferenceDescriptor;
    managedMappedInferenceDescriptor.write<Fields::MMI_final_barrier>(origOp.getFinalBarrierId());
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_work_item>(origOp.getWorkItemsCount());
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_task_configs>(origOp.getBarrierCount());
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_initial_barriers>(
            origOp.getBootstrapBarriersCount());
    managedMappedInferenceDescriptor.write<Fields::MMI_bootstrap_workitems_count>(origOp.getBootsrapWorkItemsCount());
    managedMappedInferenceDescriptor.write<Fields::MMI_actshv_used>(origOp.getActshvUsed());
    managedMappedInferenceDescriptor.write<Fields::MMI_dpu_used>(origOp.getDpuUsed());
    managedMappedInferenceDescriptor.write<Fields::MMI_media_used>(origOp.getMediaUsed());
    managedMappedInferenceDescriptor.write<Fields::MMI_dma_from_ddr_used>(origOp.getDmaFromDdrUsed());
    managedMappedInferenceDescriptor.write<Fields::MMI_dma_from_cmx_used>(origOp.getDmaFromCmxUsed());
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_nnrt_config>(1);
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_barriers_configuration>(
            origOp.getBarrierConfigurationTasksCount());
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_num_of_barrier_reprogrammings>(
            origOp.getBarriersReprogrammingCount());

    auto barrieProgrammingMode = static_cast<npu40xx::nn_public::VpuManagedMappedInference::VpuBarrierProgrammingMode>(
            origOp.getWorkloadManagementBarrierProgrammingMode());
    managedMappedInferenceDescriptor.write<Fields::MMI_barrier_programming_mode>(barrieProgrammingMode);

    managedMappedInferenceDescriptor.write<Fields::MMI_barrier_configuration_stride>(
            origOp.getBarrierConfigurationStride());

    if (origOp.getDisableDmaSwFifo()) {
        managedMappedInferenceDescriptor.write<Fields::MMI_DisableDmaSwFifo>(1);
    }

    managedMappedInferenceDescriptor.write<Fields::MMI_model_identifier>(_modelIdentifier);

    rewriter.create<NPUReg50XX::ManagedMappedInferenceOp>(origOp->getLoc(),                              //
                                                          origOp.getSymNameAttr(),                       //
                                                          origOp.getNnrtConfigAttr(),                    //
                                                          origOp.getMappedInferenceVersionAttr(),        //
                                                          origOp.getDmaTasksAttr(),                      //
                                                          origOp.getWorkItemsAttr(),                     //
                                                          origOp.getBarrierTasksAttr(),                  //
                                                          origOp.getBootstrapBarriersAttr(),             //
                                                          origOp.getBarrierConfigurationTasksAttr(),     //
                                                          origOp.getNumOfBarrierReprogrammingsAttr(),    //
                                                          std::move(managedMappedInferenceDescriptor));  //

    rewriter.eraseOp(origOp);

    return mlir::success();
}
}  // namespace vpuasm2npureg50xx
}  // namespace vpux
