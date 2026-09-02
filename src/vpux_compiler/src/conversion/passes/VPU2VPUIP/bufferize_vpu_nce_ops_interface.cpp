//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_vpu_nce_ops_interface.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/allocate_buffers.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/weight_table_offset_utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {

void addppeAttr(const Logger& log, mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp& nceOp, VPU::PPEAttr ppeAttr) {
    log.nest().trace("Adding PPE attribute '{0}'", ppeAttr);
    nceOp.addPPETask(builder, ppeAttr);
}

void addDPUTasks(const Logger& log, VPUIP::NCEClusterTaskOp nceOp, mlir::OpBuilder& rewriter, mlir::Region& workloads,
                 bool isNCEPermute) {
    log.nest().trace("Adding DPU tasks");

    const auto dpuWorkloads = workloads.getOps<VPU::DPUWorkloadOp>();

    const auto wtOffsetBuilder = VPUIP::WtOffsetBuilder::create(nceOp, workloads);

    for (auto dpuTaskOp : dpuWorkloads) {
        SmallVector<int64_t> ends;
        const auto offsets = dpuTaskOp.getConstOutputOffsets();
        const auto sizes = dpuTaskOp.getConstOutputSizes();
        ends.reserve(sizes.size());

        llvm::transform(llvm::seq<size_t>(0, sizes.size()), std::back_inserter(ends), [&](size_t index) {
            return offsets[index] + sizes[index] - 1;
        });

        mlir::ArrayAttr inStartAttr = nullptr;
        mlir::ArrayAttr inEndAttr = nullptr;
        const auto isGroupedMatMul = offsets.size() == DimsGroups5D::Act::numDims;

        SmallVector<int64_t> outDpuStart, outDpuEnd;
        // Update workloads padding, offsets and sizes after reshape and layout changes.
        if (isNCEPermute) {
            // Reshape Offsets and Sizes from CHW to HCW layout
            outDpuStart = {offsets[Dims4D::Act::H.ind()], offsets[Dims4D::Act::C.ind()], offsets[Dims4D::Act::W.ind()]};
            outDpuEnd = {ends[Dims4D::Act::H.ind()], ends[Dims4D::Act::C.ind()], ends[Dims4D::Act::W.ind()]};

            if (dpuTaskOp.getStaticInOffsetsAttr() != nullptr && dpuTaskOp.getStaticInSizesAttr() != nullptr) {
                const auto inOffset = dpuTaskOp.getStaticInOffsetsAttr();
                const auto inSizes = dpuTaskOp.getStaticInSizesAttr();
                const SmallVector<int64_t> inDpuStart{inOffset[Dims4D::Act::H.ind()], inOffset[Dims4D::Act::C.ind()],
                                                      inOffset[Dims4D::Act::W.ind()]};
                const SmallVector<int64_t> inDpuEnd{inOffset[Dims4D::Act::H.ind()] + inSizes[Dims4D::Act::H.ind()] - 1,
                                                    inOffset[Dims4D::Act::C.ind()] + inSizes[Dims4D::Act::C.ind()] - 1,
                                                    inOffset[Dims4D::Act::W.ind()] + inSizes[Dims4D::Act::W.ind()] - 1};

                inStartAttr = getIntArrayAttr(rewriter, inDpuStart);
                inEndAttr = getIntArrayAttr(rewriter, inDpuEnd);
            }
        } else if (isGroupedMatMul) {
            // This part is for grouped Matmul which has 5D input/output
            // Logic is same only dimensions are adjusted for 5D
            const auto dimC = DimsGroups5D::Act::C;
            const auto dimH = DimsGroups5D::Act::H;
            const auto dimW = DimsGroups5D::Act::W;
            outDpuStart = {offsets[dimW.ind()], offsets[dimH.ind()], offsets[dimC.ind()]};
            outDpuEnd = {ends[dimW.ind()], ends[dimH.ind()], ends[dimC.ind()]};

            if (dpuTaskOp.getStaticInOffsetsAttr() != nullptr && dpuTaskOp.getStaticInSizesAttr() != nullptr) {
                const auto inOffset = dpuTaskOp.getStaticInOffsetsAttr();
                const auto inSizes = dpuTaskOp.getStaticInSizesAttr();

                const SmallVector<int64_t> inDpuStart{inOffset[dimW.ind()], inOffset[dimH.ind()], inOffset[dimC.ind()]};
                const SmallVector<int64_t> inDpuEnd{inOffset[dimW.ind()] + inSizes[dimW.ind()] - 1,
                                                    inOffset[dimH.ind()] + inSizes[dimH.ind()] - 1,
                                                    inOffset[dimC.ind()] + inSizes[dimC.ind()] - 1};

                inStartAttr = getIntArrayAttr(rewriter, inDpuStart);
                inEndAttr = getIntArrayAttr(rewriter, inDpuEnd);
            }

        } else {
            // as soon as we need workload_x, workload_y, workload_z coords
            outDpuStart = {offsets[Dims4D::Act::W.ind()], offsets[Dims4D::Act::H.ind()], offsets[Dims4D::Act::C.ind()]};
            outDpuEnd = {ends[Dims4D::Act::W.ind()], ends[Dims4D::Act::H.ind()], ends[Dims4D::Act::C.ind()]};

            if (dpuTaskOp.getStaticInOffsetsAttr() != nullptr && dpuTaskOp.getStaticInSizesAttr() != nullptr) {
                const auto inOffset = dpuTaskOp.getStaticInOffsetsAttr();
                const auto inSizes = dpuTaskOp.getStaticInSizesAttr();

                const SmallVector<int64_t> inDpuStart{inOffset[Dims4D::Act::W.ind()], inOffset[Dims4D::Act::H.ind()],
                                                      inOffset[Dims4D::Act::C.ind()]};
                const SmallVector<int64_t> inDpuEnd{inOffset[Dims4D::Act::W.ind()] + inSizes[Dims4D::Act::W.ind()] - 1,
                                                    inOffset[Dims4D::Act::H.ind()] + inSizes[Dims4D::Act::H.ind()] - 1,
                                                    inOffset[Dims4D::Act::C.ind()] + inSizes[Dims4D::Act::C.ind()] - 1};

                inStartAttr = getIntArrayAttr(rewriter, inDpuStart);
                inEndAttr = getIntArrayAttr(rewriter, inDpuEnd);
            }
        }

        auto dpuTask = nceOp.addDPUTask(
                rewriter, getIntArrayAttr(rewriter, outDpuStart), getIntArrayAttr(rewriter, outDpuEnd), inStartAttr,
                inEndAttr, dpuTaskOp.getPadAttribute(), dpuTaskOp.getMpeMode(), dpuTaskOp.getClusterIdAttr());
        wtOffsetBuilder->maybeSetWeightTableOffsetAttr(dpuTask, outDpuStart[2], outDpuEnd[2]);
    }
}

//
// Create VPUIP.NCEClusterTask and ensure sparse types interact with the operation as individual buffers
//

// Holds the buffers backing every output of an NCE operation. Each non-data field is
// nullptr when the corresponding output is absent. Sparsity-map presence is derived
// from the main output's MLIR type (see allocateNceOutputBuffers below) rather than
// from a caller-set flag, so callers cannot forget to opt in.
struct NceOutputBuffers {
    mlir::Value data;
    mlir::Value sparsityMap;
    mlir::Value reduceMaxXy;
    mlir::Value reduceMinXy;
    mlir::Value reduceMinMaxTensor;
};

// Allocates the data buffer for `mainOutput` (transparently expanding a SparseTensor
// main output into data + sparsity-map buffers) and one buffer per supplied non-null
// reduce output. Pass nullptr for reduce outputs the op does not produce.
NceOutputBuffers allocateNceOutputBuffers(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                          mlir::Value mainOutput, mlir::Value reduceMaxXy = nullptr,
                                          mlir::Value reduceMinXy = nullptr, mlir::Value reduceMinMaxTensor = nullptr) {
    SmallVector<mlir::Value> origOutputs{mainOutput};
    if (reduceMaxXy != nullptr) {
        origOutputs.push_back(reduceMaxXy);
    }
    if (reduceMinXy != nullptr) {
        origOutputs.push_back(reduceMinXy);
    }
    if (reduceMinMaxTensor != nullptr) {
        origOutputs.push_back(reduceMinMaxTensor);
    }

    const auto buffers = VPUIP::allocateBuffers(log, loc, builder, origOutputs, /*individualBuffers=*/true);

    NceOutputBuffers result;
    size_t idx = 0;
    VPUX_THROW_UNLESS(idx < buffers.size(), "allocateBuffers returned no buffers for the main NCE output");
    result.data = buffers[idx++];

    if (mlir::isa<vpux::VPU::SparseTensorType>(mainOutput.getType())) {
        VPUX_THROW_UNLESS(idx < buffers.size(),
                          "Sparse main output requires a sparsity-map buffer (got {0} buffers, expected at least 2)",
                          buffers.size());
        result.sparsityMap = buffers[idx++];
    }
    if (reduceMaxXy != nullptr) {
        VPUX_THROW_UNLESS(idx < buffers.size(), "Missing buffer for reduceMaxXy output");
        result.reduceMaxXy = buffers[idx++];
    }
    if (reduceMinXy != nullptr) {
        VPUX_THROW_UNLESS(idx < buffers.size(), "Missing buffer for reduceMinXy output");
        result.reduceMinXy = buffers[idx++];
    }
    if (reduceMinMaxTensor != nullptr) {
        VPUX_THROW_UNLESS(idx < buffers.size(), "Missing buffer for reduceMinMaxTensor output");
        result.reduceMinMaxTensor = buffers[idx++];
    }
    VPUX_THROW_UNLESS(idx == buffers.size(),
                      "Unexpected extra output buffers (consumed {0} of {1}); reduce output may itself be sparse", idx,
                      buffers.size());
    return result;
}

struct NCEClusterTaskParams {
    struct Weights {
        mlir::Value weights;
        mlir::Value weightsTable;
        mlir::Value weightTableDataPtr;
        mlir::Value weightTableScale;
        mlir::Value weightTableBias;
        mlir::Value weightTableZeroPoints;
    };

    struct Kernel {
        mlir::ArrayAttr kernelSizeAttr;
        mlir::ArrayAttr kernelStridesAttr;
        vpux::VPU::PaddingAttr kernelPaddingAttr;
    };

    // Required attributes: They have to explicitly be set by the user.
    mlir::Value input;
    Weights weights;
    NceOutputBuffers outputs;
    vpux::VPUIP::NCETaskType taskType;
    Kernel kernel;
    mlir::Region& workloads;

    NCEClusterTaskParams(mlir::Value input, const Weights& weights, const NceOutputBuffers& outputs,
                         vpux::VPUIP::NCETaskType taskType, const Kernel& kernel, mlir::Region& workloads)
            : input(input),
              weights(weights),
              outputs(outputs),
              taskType(taskType),
              kernel(kernel),
              workloads(workloads) {
    }

    // Optional attributes
    bool isSuperdense = false;
    VPU::PPEAttr ppeAttr = nullptr;
    mlir::Attribute dpuCostAttr = nullptr;
    mlir::BoolAttr isInplace = nullptr;
    bool isPermuteQuantize = false;
    mlir::IntegerAttr cmSpPattern = nullptr;
    bool inputChannelsCompression = false;
    bool isNCEPermute = false;
    bool smallKernelOptimization = false;
    VPU::MPEEngineAttr mpeEngineAttr = nullptr;
    VPU::EltwiseTypeAttr eltwiseType = nullptr;
    TilingLoopIndexAttr tilingLoopIndex = nullptr;
    VFLoopIndexAttr vfLoopIndex = nullptr;
    VFLoopTileIndexAttr vfLoopTileIndex = nullptr;
    VPU::S2DD2SConfigAttr s2dD2sConfig = nullptr;
};

SmallVector<mlir::Value> createNCEClusterTask(mlir::OpBuilder& rewriter, mlir::Location loc,
                                              const NCEClusterTaskParams& params, Logger log = Logger::global()) {
    const auto getIndividualBuffers = [&](mlir::Value value) {
        mlir::Value data = value;
        mlir::Value sparsityMap = nullptr;
        mlir::Value seTable = nullptr;
        if (value != nullptr && mlir::isa<vpux::VPUIP::SparseBufferType>(value.getType())) {
            auto ungroupedOp = rewriter.create<VPUIP::UngroupSparseBufferOp>(loc, value);
            data = ungroupedOp.getData();
            sparsityMap = ungroupedOp.getSparsityMap();
            seTable = ungroupedOp.getStorageElementTable();
        }
        return std::make_tuple(data, sparsityMap, seTable);
    };

    mlir::Value inputData, inputSparsityMap, inputSETable;
    std::tie(inputData, inputSparsityMap, inputSETable) = getIndividualBuffers(params.input);

    mlir::Value weightsData, weightsSparsityMap;
    std::tie(weightsData, weightsSparsityMap, std::ignore) = getIndividualBuffers(params.weights.weights);

    const auto& outputs = params.outputs;
    mlir::SmallVector<mlir::Value> reduceMinMaxTensor;
    if (outputs.reduceMinMaxTensor != nullptr) {
        reduceMinMaxTensor.push_back(outputs.reduceMinMaxTensor);
    }

    auto nceClusterTask = rewriter.create<VPUIP::NCEClusterTaskOp>(
            loc, inputData, inputSparsityMap, inputSETable, weightsData, weightsSparsityMap,
            params.weights.weightsTable,
            /*weight_table_data_ptr=*/params.weights.weightTableDataPtr, /*weight_table_sp_ptr=*/nullptr,
            params.weights.weightTableScale, params.weights.weightTableBias,
            /*weight_zero_points=*/params.weights.weightTableZeroPoints,
            /*sprLookupTable=*/nullptr, /*palletLookupTable=*/nullptr, inputData, inputSparsityMap, inputSETable,
            outputs.data, outputs.sparsityMap, outputs.data, outputs.sparsityMap, /*profiling_data=*/nullptr,
            /*dynamic_sequence_length*/ nullptr, outputs.reduceMaxXy, outputs.reduceMinXy, reduceMinMaxTensor,
            params.taskType, params.kernel.kernelSizeAttr, params.kernel.kernelStridesAttr,
            params.kernel.kernelPaddingAttr,
            /*is_continued=*/false, params.cmSpPattern,
            /*is_segmented=*/false,
            /*out_channel_offset=*/nullptr, params.inputChannelsCompression, /*isZeroOffsetWeightsTable=*/false,
            params.isSuperdense, params.isInplace,
            /*input_se_size=*/nullptr,
            /*output_se_size=*/nullptr, params.isPermuteQuantize, params.smallKernelOptimization, params.mpeEngineAttr,
            params.eltwiseType, /*sparsity_config*/ nullptr, /*dynamic_scale_config=*/nullptr, /*local_region=*/nullptr,
            params.s2dD2sConfig);

    addDPUTasks(log, nceClusterTask, rewriter, params.workloads, params.isNCEPermute);
    addppeAttr(log, rewriter, nceClusterTask, params.ppeAttr);

    if (params.dpuCostAttr != nullptr) {
        nceClusterTask->setAttr(DPUCost, params.dpuCostAttr);
    }

    SmallVector<mlir::Value> results;
    if (nceClusterTask.getOutputSparsityMap() != nullptr) {
        auto groupedOp = rewriter.create<VPUIP::GroupSparseBufferOp>(loc, nceClusterTask.getOutput(),
                                                                     nceClusterTask.getOutputSparsityMap());
        results.push_back(groupedOp.getOutput());
    } else {
        if (nceClusterTask.getOutput()) {
            results.push_back(nceClusterTask.getOutput());
        }
    }

    if (params.tilingLoopIndex != nullptr) {
        nceClusterTask->setAttr(TILING_LOOP_INDEX_ATTR_NAME, params.tilingLoopIndex);
    }
    if (params.vfLoopIndex != nullptr) {
        nceClusterTask->setAttr(VF_LOOP_INDEX_ATTR_NAME, params.vfLoopIndex);
    }
    if (params.vfLoopTileIndex != nullptr) {
        nceClusterTask->setAttr(VF_LOOP_TILE_INDEX_ATTR_NAME, params.vfLoopTileIndex);
    }

    if (nceClusterTask.getMaxPerXy()) {
        results.push_back(mlir::cast<mlir::Value>(nceClusterTask.getMaxPerXy()));
    }
    if (nceClusterTask.getMinPerXy()) {
        results.push_back(mlir::cast<mlir::Value>(nceClusterTask.getMinPerXy()));
    }
    if (!nceClusterTask.getMinMaxPerTensor().empty()) {
        results.push_back(mlir::cast<mlir::Value>(nceClusterTask.getMinMaxPerTensor()[0]));
    }
    return results;
}

bool isSuperdenseOp(mlir::Operation* nceOp) {
    auto outType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    const auto outputOrder = outType.getDimsOrder();
    const auto outputShape = outType.getShape();
    const auto outElemType = outType.getElementType();

    // Check output shape for each cluster
    if (auto distributedTensorType = mlir::dyn_cast<VPU::DistributedTensorType>(outType)) {
        auto tiledComputeShapes = distributedTensorType.getPerClusterComputeShapes();
        for (auto& computeShape : tiledComputeShapes) {
            if (VPU::NCESparsity::isSuperdenseRequired(outputOrder, computeShape, outElemType)) {
                return true;
            }
        }
        return false;
    }

    return VPU::NCESparsity::isSuperdenseRequired(outputOrder, outputShape, outElemType);
}

SmallVector<int64_t> calculateWCHShape(ArrayRef<int64_t> shape) {
    const int64_t tensorSizeZ = shape[Dims4D::Act::W.ind()];
    const int64_t tensorSizeY = shape[Dims4D::Act::C.ind()];
    const int64_t tensorSizeX = shape[Dims4D::Act::H.ind()];
    return {shape[Dims4D::Act::N.ind()], tensorSizeZ, tensorSizeY, tensorSizeX};
}

VPU::DistributedTensorType createCustomDistributedTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                             NDTypeInterface targetType,
                                                             VPU::DistributionInfoAttr origDistTensorAttr,
                                                             mlir::UnitAttr equalMemoryAndComputeView, ShapeRef shape) {
    auto* ctx = clusteredOp->getContext();

    const auto memSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto order = mlir::AffineMapAttr::get(targetType.getDimsOrder().toAffineMap(ctx));
    auto elemType = targetType.getElementType();

    const auto origDistTensorCtx = origDistTensorAttr.getContext();

    auto newNumTilesAttr = origDistTensorAttr.getNumTiles();
    if (newNumTilesAttr != nullptr) {
        auto numTiles = parseIntArrayAttr<int64_t>(newNumTilesAttr);
        newNumTilesAttr = getIntArrayAttr(origDistTensorCtx, calculateWCHShape(numTiles));
    }

    const auto activationTensorDistributionModeAttr =
            VPU::DistributionModeAttr::get(ctx, origDistTensorAttr.getMode().getValue());
    // Padding adaptions
    auto newPadAttr = origDistTensorAttr.getPads();
    if (newPadAttr != nullptr) {
        const auto fullInputChannels = mlir::cast<NDTypeInterface>(clusteredOp.getOperation()->getOperand(0).getType())
                                               .getShape()[Dims4D::Act::C];
        const auto fullOutputChannels = mlir::cast<NDTypeInterface>(clusteredOp.getOperation()->getResult(0).getType())
                                                .getShape()[Dims4D::Act::C];

        newPadAttr = VPU::getPaddingAttr(origDistTensorCtx, PadInfo(origDistTensorAttr.getPads().getTop().getInt(),
                                                                    origDistTensorAttr.getPads().getBottom().getInt(),
                                                                    0, fullOutputChannels - fullInputChannels));
    }
    auto newKernelAttr = origDistTensorAttr.getKernel();
    if (newKernelAttr != nullptr) {
        auto newKernel = parseIntArrayAttr<int64_t>(newKernelAttr);
        newKernelAttr = getIntArrayAttr(origDistTensorCtx,
                                        SmallVector<int64_t>{/*neutral val*/ 1, newKernel[Dims4D::Kernel::Y.ind()]});
    }
    auto newStridesAttr = origDistTensorAttr.getStrides();
    if (newStridesAttr != nullptr) {
        auto newStrides = parseIntArrayAttr<int64_t>(newStridesAttr);
        newStridesAttr = getIntArrayAttr(origDistTensorCtx,
                                         SmallVector<int64_t>{/*neutral val*/ 1, newStrides[Dims4D::Strides::Y.ind()]});
    }
    auto newAlignmentAttr = origDistTensorAttr.getAlignment();
    if (newAlignmentAttr != nullptr) {
        auto newAlignment = parseIntArrayAttr<int64_t>(newAlignmentAttr);
        newAlignmentAttr = getIntArrayAttr(origDistTensorCtx, calculateWCHShape(newAlignment));
    }

    auto calculateWCHShapeForArrayOfArray = [origDistTensorCtx](const mlir::ArrayAttr shape) -> mlir::ArrayAttr {
        if (shape != nullptr) {
            auto newIntShape = parseIntArrayOfArrayAttr<int64_t>(shape);
            for (size_t i = 0; i < newIntShape.size(); i++) {
                newIntShape[i] = calculateWCHShape(newIntShape[i]);
            }
            return getIntArrayOfArray(origDistTensorCtx, newIntShape);
        }
        return nullptr;
    };

    auto newMemoryNumTilesAttr = origDistTensorAttr.getMemoryNumTiles();
    if (newMemoryNumTilesAttr != nullptr) {
        auto memoryNumTiles = parseIntArrayAttr<int64_t>(newMemoryNumTilesAttr);
        newMemoryNumTilesAttr = getIntArrayAttr(origDistTensorCtx, calculateWCHShape(memoryNumTiles));
    }

    auto distributedTensorAttr = VPU::DistributionInfoAttr::get(
            ctx, activationTensorDistributionModeAttr, newNumTilesAttr, newKernelAttr, newPadAttr, newStridesAttr,
            origDistTensorAttr.getNumClusters(), newAlignmentAttr, origDistTensorAttr.getUniformDistributedSegments(),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getComputeShapes()),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getComputeOffsets()),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getMemoryShapes()),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getMemoryOffsets()), equalMemoryAndComputeView,
            newMemoryNumTilesAttr);

    return VPU::DistributedTensorType::get(ctx, ArrayRef(calculateWCHShape(shape.raw())), elemType, order, memSpace,
                                           distributedTensorAttr);
}

}  // namespace

//
// bufferize VPU::NCEConvolutionOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEConvolutionOp origOp,
                                      VPU::NCEConvolutionOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEConvolutionOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Get dimensions
    //

    const auto filterShape = Shape(origOp.getStaticRawFilterShape());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    VPUX_THROW_WHEN(KY == mlir::ShapedType::kDynamic || KX == mlir::ShapedType::kDynamic,
                    "NCEConvolutionOp requires static KY/KX during bufferization");

    //
    // Prepare output buffer for DPU
    //
    const auto outputs =
            allocateNceOutputBuffers(log, origOp.getLoc(), rewriter, origOp.getOutput(), origOp.getReduceXyMax(),
                                     origOp.getReduceXyMin(), origOp.getReduceTensorMinMax());

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    const auto taskType = VPUIP::NCETaskType::CONV;
    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    bool isSuperdense = false;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdense = true;
    }

    mlir::IntegerAttr cmSpPattern = nullptr;
    auto inputShape = mlir::cast<NDTypeInterface>(newArgs.getInput().getType()).getShape();
    if (inputShape.size() == 4 && inputShape[Dims4D::Act::C] < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) {
        const auto pattern = (static_cast<int64_t>(1) << inputShape[Dims4D::Act::C]) - 1;
        cmSpPattern = getIntAttr(ctx, pattern);
    }

    const auto loopAttributes = getLoopAttributes(origOp);
    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{newArgs.getFilter(), newArgs.getWeightsTable(),
                                          /*weightTableDataPtr=*/nullptr, newArgs.getWeightTableScale(),
                                          newArgs.getWeightTableBias(), newArgs.getWeightZeroPoints()},
            outputs, taskType, NCEClusterTaskParams::Kernel{kernelSizeAttr, origOp.getStrides(), origOp.getPadAttr()},
            origOp.getWorkloads());
    params.isSuperdense = isSuperdense;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.cmSpPattern = cmSpPattern;
    params.mpeEngineAttr = origOp.getMpeEngineAttr();
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopTileIndex = loopAttributes.vfLoopTileIndex;

    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEMaxPoolOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* /*ctx*/, VPU::NCEMaxPoolOp origOp,
                                      VPU::NCEMaxPoolOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEMaxPoolOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Prepare output buffer for DPU
    //
    const auto outputs =
            allocateNceOutputBuffers(log, origOp.getLoc(), rewriter, origOp.getOutput(), origOp.getReduceXyMax(),
                                     origOp.getReduceXyMin(), origOp.getReduceTensorMinMax());

    //
    // Create NCE per-cluster Operation
    //

    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    bool isSuperdense = false;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdense = true;
    }

    const auto mpeEngineAttr = origOp.getMpeEngineAttr();
    const auto loopAttributes = getLoopAttributes(origOp);
    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{/*weights=*/nullptr, newArgs.getWeightsTable(),
                                          /*weightTableDataPtr=*/nullptr,
                                          /*weightTableScale=*/newArgs.getWeightTableScale(),
                                          /*weightTableBias=*/newArgs.getWeightTableBias(),
                                          /*weightTableZeroPoints=*/nullptr},
            outputs, VPUIP::NCETaskType::MAXPOOL,
            NCEClusterTaskParams::Kernel{origOp.getKernelSize(), origOp.getStrides(), origOp.getPadAttr()},
            origOp.getWorkloads());
    params.isSuperdense = isSuperdense;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopTileIndex = loopAttributes.vfLoopTileIndex;
    params.s2dD2sConfig = origOp.getS2dd2sConfigAttr();

    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEAveragePoolOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* /*ctx*/, VPU::NCEAveragePoolOp origOp,
                                      VPU::NCEAveragePoolOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEAveragePoolOp", 0);
    //
    // Prepare output buffer for DPU
    //

    const auto outputs = allocateNceOutputBuffers(log, origOp.getLoc(), rewriter, origOp.getOutput());

    //
    // Create NCE per-cluster Operation
    //

    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    bool isSuperdense = false;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdense = true;
    }

    bool isSmallKernelOptimization = VPU::NCEInvariant::isSmallKernelOptimizationSupported(origOp);

    const auto mpeEngineAttr = origOp.getMpeEngineAttr();
    const auto loopAttributes = getLoopAttributes(origOp);

    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{/*weights=*/nullptr, /*weightsTable=*/nullptr, /*weightTableDataPtr=*/nullptr,
                                          /*weightTableScale=*/newArgs.getWeightTableScale(),
                                          /*weightTableBias=*/newArgs.getWeightTableBias(),
                                          /*weightTableZeroPoints=*/nullptr},
            outputs, VPUIP::NCETaskType::AVEPOOL,
            NCEClusterTaskParams::Kernel{origOp.getKernelSize(), origOp.getStrides(), origOp.getPadAttr()},
            origOp.getWorkloads());
    params.isSuperdense = isSuperdense;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.smallKernelOptimization = isSmallKernelOptimization;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopTileIndex = loopAttributes.vfLoopTileIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEDepthConvolutionOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEDepthConvolutionOp origOp,
                                      VPU::NCEDepthConvolutionOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEDepthConvolutionOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Get dimensions
    //

    const auto filterShape = origOp.getStaticRawFilterShape();
    const auto KY = filterShape[Dims4D::Filter::KY.ind()];
    const auto KX = filterShape[Dims4D::Filter::KX.ind()];
    VPUX_THROW_WHEN(KY == mlir::ShapedType::kDynamic || KX == mlir::ShapedType::kDynamic,
                    "NCEDepthConvolutionOp requires static KY/KX during bufferization");

    //
    // Prepare output buffer for DPU
    //

    const auto outputs =
            allocateNceOutputBuffers(log, origOp.getLoc(), rewriter, origOp.getOutput(), origOp.getReduceXyMax(),
                                     origOp.getReduceXyMin(), origOp.getReduceTensorMinMax());

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    auto ppeAttr = origOp.getPpeAttr();

    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    bool isSuperdense = false;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdense = true;
    }

    bool isSmallKernelOptimization = VPU::NCEInvariant::isSmallKernelOptimizationSupported(origOp);

    const auto mpeEngineAttr = origOp.getMpeEngineAttr();
    const auto loopAttributes = getLoopAttributes(origOp);

    NCEClusterTaskParams params(newArgs.getInput(),
                                NCEClusterTaskParams::Weights{
                                        newArgs.getFilter(), newArgs.getWeightsTable(), newArgs.getWeightTableDataPtr(),
                                        newArgs.getWeightTableScale(), newArgs.getWeightTableBias(), nullptr},
                                outputs, VPUIP::NCETaskType::DWCONV,
                                NCEClusterTaskParams::Kernel{kernelSizeAttr, origOp.getStrides(), origOp.getPadAttr()},
                                origOp.getWorkloads());
    params.isSuperdense = isSuperdense;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.smallKernelOptimization = isSmallKernelOptimization;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopTileIndex = loopAttributes.vfLoopTileIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEInterpolateOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEInterpolateOp origOp,
                                      VPU::NCEInterpolateOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEInterpolateOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto filterShape = Shape(origOp.getStaticRawFilterShape());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    VPUX_THROW_WHEN(KY == mlir::ShapedType::kDynamic || KX == mlir::ShapedType::kDynamic,
                    "NCEInterpolateOp requires static KY/KX during bufferization");

    auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));

    log.nest().trace("Allocating output buffer");

    auto newLoc = appendLoc(origOp.getLoc(), "interpolate");

    const auto outputs = allocateNceOutputBuffers(log, newLoc, rewriter, origOp.getOutput());

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    bool isSuperdense = false;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdense = true;
    }

    const auto mpeEngineAttr = origOp.getMpeEngineAttr();
    const auto loopAttributes = getLoopAttributes(origOp);

    auto nceOpInterface = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    // Interpolate operation is being convert to Convolution here. Parameter weightTableDataPtr is set to nullptr,
    // because Convolution doesn't require data-pointer table
    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{newArgs.getWeights(), newArgs.getWeightsTable(),
                                          /*weightTableDataPtr=*/nullptr, newArgs.getWeightTableScale(),
                                          newArgs.getWeightTableBias(), /*weightTableZeroPoints=*/nullptr},
            outputs, VPUIP::NCETaskType::CONV,
            NCEClusterTaskParams::Kernel{kernelSizeAttr, getIntArrayAttr(ctx, nceOpInterface.getStridesVal()),
                                         nceOpInterface.getPad()},
            origOp.getWorkloads());
    params.isSuperdense = isSuperdense;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopTileIndex = loopAttributes.vfLoopTileIndex;
    auto nceOp = createNCEClusterTask(rewriter, newLoc, params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEEltwiseOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* /*ctx*/, VPU::NCEEltwiseOp origOp,
                                      VPU::NCEEltwiseOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEEltwiseOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Prepare output buffer for DPU
    //

    const auto outputs =
            allocateNceOutputBuffers(log, origOp.getLoc(), rewriter, origOp.getOutput(), origOp.getReduceXyMax(),
                                     origOp.getReduceXyMin(), origOp.getReduceTensorMinMax());

    //
    // Create NCE per-cluster Operation
    //

    auto ppeAttr = origOp.getPpeAttr();

    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    bool isSuperdense = false;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdense = true;
    }

    const auto mpeEngineAttr = origOp.getMpeEngineAttr();
    const auto loopAttributes = getLoopAttributes(origOp);

    NCEClusterTaskParams params(newArgs.getInput1(),
                                NCEClusterTaskParams::Weights{newArgs.getInput2(), /*weightsTable=*/nullptr,
                                                              /*weightTableDataPtr=*/nullptr,
                                                              /*weightTableScale=*/newArgs.getWeightTableScale(),
                                                              /*weightTableBias=*/newArgs.getWeightTableBias(),
                                                              /*weightTableZeroPoints=*/nullptr},
                                outputs, VPUIP::NCETaskType::ELTWISE,
                                NCEClusterTaskParams::Kernel{nullptr, nullptr, nullptr}, origOp.getWorkloads());
    params.isSuperdense = isSuperdense;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.isInplace = origOp.getIsInplaceAttr();
    params.mpeEngineAttr = mpeEngineAttr;
    params.eltwiseType = origOp.getOpTypeAttr();
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopTileIndex = loopAttributes.vfLoopTileIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEReduceOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEReduceOp origOp,
                                      VPU::NCEReduceOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEReduceOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Prepare output buffer for DPU
    //

    const auto outputs = allocateNceOutputBuffers(log, origOp.getLoc(), rewriter, origOp.getOutput());

    //
    // Create NCE per-cluster Operation
    //

    auto nceTaskType = VPU::configureNCEReduceTaskType(origOp);
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    bool isSuperdense = false;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdense = true;
    }

    const auto mpeEngineAttr = origOp.getMpeEngineAttr();
    const auto loopAttributes = getLoopAttributes(origOp);
    auto nceOpInterface = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());

    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{nceOpInterface.getWeightsOperand(), nceOpInterface.getWeightsTableOperand(),
                                          nullptr, nullptr, nullptr, nullptr},
            outputs, nceTaskType,
            NCEClusterTaskParams::Kernel{getIntArrayAttr(ctx, nceOpInterface.getKernelSizeVal()),
                                         getIntArrayAttr(ctx, nceOpInterface.getStridesVal()), nceOpInterface.getPad()},
            origOp.getWorkloads());
    params.isSuperdense = isSuperdense;
    params.ppeAttr = origOp.getPpeAttr();
    params.dpuCostAttr = dpuCostAttr;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopTileIndex = loopAttributes.vfLoopTileIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCECompressConvolutionOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCECompressConvolutionOp origOp,
                                      VPU::NCECompressConvolutionOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCECompressConvolutionOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Get dimensions
    //

    const auto filterShape = Shape(origOp.getStaticRawFilterShape());

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    VPUX_THROW_WHEN(KY == mlir::ShapedType::kDynamic || KX == mlir::ShapedType::kDynamic,
                    "NCECompressConvolutionOp requires static KY/KX during bufferization");

    const auto channelAlignValue = VPU::NCEInvariant::getAlignment(
            mlir::cast<vpux::NDTypeInterface>(newArgs.getFilter().getType()).getElementType());

    const auto finalShape = SmallVector<int64_t>({filterShape[Dims4D::Filter::OC], channelAlignValue, KY, KX});
    auto shapeCastWeightsOp = rewriter.create<VPUIP::ShapeCastOp>(origOp->getLoc(), newArgs.getFilter(),
                                                                  getIntArrayAttr(origOp.getContext(), finalShape));
    //
    // Prepare output buffer for DPU
    //

    const auto outputs = allocateNceOutputBuffers(log, origOp.getLoc(), rewriter, origOp.getOutput());

    //
    // Create NCE per-cluster Operation
    //
    auto inputType = newArgs.getInput().getType();
    const auto inputShape = mlir::cast<vpux::NDTypeInterface>(inputType).getShape();
    const auto finalInputShape = vpux::Shape(
            {inputShape[Dims4D::Act::N], channelAlignValue, inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W]});
    auto finalInputShapeAttr = getIntArrayAttr(origOp.getContext(), finalInputShape);

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    bool isSuperdense = false;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdense = true;
    }
    auto inputShapeCastOp =
            rewriter.create<VPUIP::ShapeCastOp>(origOp->getLoc(), newArgs.getInput(), finalInputShapeAttr);
    const bool inputChannelsCompression = true;

    const auto mpeEngineAttr = origOp.getMpeEngineAttr();
    const auto loopAttributes = getLoopAttributes(origOp);

    NCEClusterTaskParams params(inputShapeCastOp.getResult(),
                                NCEClusterTaskParams::Weights{shapeCastWeightsOp.getResult(), newArgs.getWeightsTable(),
                                                              nullptr, nullptr, nullptr, nullptr},
                                outputs, VPUIP::NCETaskType::CONV,
                                NCEClusterTaskParams::Kernel{kernelSizeAttr, origOp.getStrides(), origOp.getPadAttr()},
                                origOp.getWorkloads());
    params.isSuperdense = isSuperdense;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.cmSpPattern = origOp.getCmSpPatternAttr();
    params.inputChannelsCompression = inputChannelsCompression;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopTileIndex = loopAttributes.vfLoopTileIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEPermuteOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEPermuteOp origOp,
                                      VPU::NCEPermuteOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEPermuteOp", 0);

    const auto& ppeConfig = VPU::getPpeConfig(ctx);

    auto copyDistTensorType = mlir::dyn_cast<VPU::DistributedTensorType>(origOp->getOperand(0).getType());
    if (copyDistTensorType != nullptr) {
        log.trace("Got '{0}' Multi Tile at '{1}'", origOp->getName(), origOp->getLoc());

        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                          origOp);

        const auto loc = origOp.getLoc();
        const auto copyDistTensorAttr = copyDistTensorType.getDistribution();
        auto targetType = mlir::cast<NDTypeInterface>(origOp.getOperand().getType());
        targetType = targetType.changeDimsOrder(DimsOrder::NHWC);

        auto castToDistType = createCustomDistributedTensorType(clusteredOp, targetType, copyDistTensorAttr,
                                                                copyDistTensorAttr.getEqualMemoryAndComputeView(),
                                                                targetType.getShape());

        auto outBufferTypeInViewOp = vpux::getBufferType(castToDistType);
        const auto castLoc = appendLoc(loc, "cast number of input tiles");
        // ViewOp Input
        // Reshape to NxWxCxH
        // Layout change to NHWC
        auto inputViewOp = rewriter.create<VPUIP::ViewOp>(castLoc, outBufferTypeInViewOp, newArgs.getInput());

        // Manual update output type
        auto outType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
        auto outTypeShape = outType.getShape();
        targetType = targetType.changeElemType(outType.getElementType());
        auto origOutDistribution = mlir::cast<VPU::DistributedTensorType>(outType).getDistribution();
        auto newDistType =
                createCustomDistributedTensorType(clusteredOp, targetType, origOutDistribution,
                                                  origOutDistribution.getEqualMemoryAndComputeView(), outTypeShape);
        auto newOutputType = newDistType.changeDimsOrder(DimsOrder::NWCH);

        //
        // Prepare output buffer for DPU
        //
        auto bufferType = vpux::getBufferType(newOutputType);

        auto ppeAttr = origOp.getPpeAttr();
        const auto& modeAdapter = ppeConfig.getFactoryAs<vpux::VPU::IPpeAdapterMode>();
        ppeAttr = modeAdapter.updateMode(ppeAttr, vpux::VPU::PPEMode::ADD);

        bool isSuperdense = false;
        if (isSuperdenseOp(origOp)) {
            VPUX_THROW_WHEN(mlir::isa<VPU::SparseTensorType>(origOp->getResult(0).getType()),
                            "Output cannot be sparse and super-dense at the same time");
            isSuperdense = true;
        }

        const auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;
        const bool isPermuteQuantize = true;
        const auto mpeEngineAttr = origOp.getMpeEngineAttr();
        const auto loopAttributes = getLoopAttributes(origOp);

        log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
        const auto outputBuffers =
                VPUIP::allocateBuffersOfType(log.nest(), loc, rewriter, bufferType, /*individualBuffers=*/true);
        VPUX_THROW_UNLESS(outputBuffers.size() == 1,
                          "NCEPermute expects a single output buffer (got {0}); main output is not sparse",
                          outputBuffers.size());
        NceOutputBuffers outputs;
        outputs.data = outputBuffers[0];

        NCEClusterTaskParams params(
                inputViewOp.getResult(),
                NCEClusterTaskParams::Weights{inputViewOp.getResult(), nullptr, nullptr, nullptr, nullptr, nullptr},
                outputs, VPUIP::NCETaskType::ELTWISE, NCEClusterTaskParams::Kernel{nullptr, nullptr, nullptr},
                origOp.getWorkloads());
        params.isSuperdense = isSuperdense;
        params.ppeAttr = ppeAttr;
        params.dpuCostAttr = dpuCostAttr;
        params.isPermuteQuantize = isPermuteQuantize;
        params.isNCEPermute = true;
        params.mpeEngineAttr = mpeEngineAttr;
        params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
        params.vfLoopIndex = loopAttributes.vfLoopIndex;
        params.vfLoopTileIndex = loopAttributes.vfLoopTileIndex;
        auto nceOpResult = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

        // ViewOp Output
        // Reshape to NxCxHxW
        // Layout change to NHWC

        auto outputViewOp = rewriter.create<VPUIP::ViewOp>(
                origOp.getLoc(), vpux::getBufferType(origOp.getResult().getType()), nceOpResult[0]);

        mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, outputViewOp.getResult());

        return mlir::success();
    }

    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());

    log.trace("Got '{0}' Single Tile '{1}'", origOp->getName(), origOp->getLoc());

    // ViewOp Input
    // Reshape to NxWxCxH
    // Layout change to NHWC
    const auto inputShape = getShape(newArgs.getInput());
    const auto targetShape = calculateWCHShape(inputShape.raw());

    auto inType = mlir::cast<vpux::NDTypeInterface>(newArgs.getInput().getType());
    const auto targetInOutOrder = DimsOrder::NHWC;
    inType = inType.changeShape(ShapeRef(targetShape));
    inType = inType.changeDimsOrder(targetInOutOrder);
    auto viewOpIn = rewriter.create<VPUIP::ViewOp>(origOp.getLoc(), inType, newArgs.getInput());

    auto ppeAttr = origOp.getPpeAttr();
    const auto& modeAdapter = ppeConfig.getFactoryAs<vpux::VPU::IPpeAdapterMode>();
    ppeAttr = modeAdapter.updateMode(ppeAttr, vpux::VPU::PPEMode::ADD);

    // Manual update output type
    const auto outNCEPermuteShape = calculateWCHShape(outType.getShape().raw());
    outType = outType.changeShape(ShapeRef(outNCEPermuteShape));
    outType = outType.changeDimsOrder(DimsOrder::NWCH);

    //
    // Prepare output buffer for DPU
    //
    auto bufferType = vpux::getBufferType(outType);

    log.nest().trace("Allocating result buffer of type '{0}' for value type '{1}'", bufferType, outType);
    const auto outputBuffers =
            VPUIP::allocateBuffersOfType(log.nest(), origOp.getLoc(), rewriter, bufferType, /*individualBuffers=*/true);
    VPUX_THROW_UNLESS(outputBuffers.size() == 1,
                      "NCEPermute expects a single output buffer (got {0}); main output is not sparse",
                      outputBuffers.size());
    NceOutputBuffers outputs;
    outputs.data = outputBuffers[0];

    bool isSuperdense = false;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdense = true;
    }

    const auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;
    const bool isPermuteQuantize = true;

    const auto mpeEngineAttr = origOp.getMpeEngineAttr();
    const auto loopAttributes = getLoopAttributes(origOp);

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    NCEClusterTaskParams params(
            viewOpIn.getResult(),
            NCEClusterTaskParams::Weights{viewOpIn.getResult(), nullptr, nullptr, nullptr, nullptr, nullptr}, outputs,
            VPUIP::NCETaskType::ELTWISE, NCEClusterTaskParams::Kernel{nullptr, nullptr, nullptr},
            origOp.getWorkloads());
    params.isSuperdense = isSuperdense;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.isPermuteQuantize = isPermuteQuantize;
    params.isNCEPermute = true;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopTileIndex = loopAttributes.vfLoopTileIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    // ViewOp Output
    // Reshape to NxCxHxW
    // Layout change to NHWC
    auto viewOpOutType = mlir::cast<vpux::NDTypeInterface>(nceOp[0].getType()).changeDimsOrder(targetInOutOrder);
    viewOpOutType = viewOpOutType.changeShape(getShape(origOp.getOutput()));
    auto viewOpOut = rewriter.create<VPUIP::ViewOp>(origOp.getLoc(), viewOpOutType, nceOp[0]);
    SmallVector<mlir::Value> results;
    results.push_back(viewOpOut.getResult());
    if (nceOp.size() > 1) {
        results.append(nceOp.begin() + 1, nceOp.end());
    }
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, results);

    return mlir::success();
}

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEMatMulOp origOp,
                                      VPU::NCEMatMulOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEMatMulOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    //
    // Get dimensions
    //
    const auto filterShape = Shape(origOp.getStaticRawFilterShape());

    const auto KY = filterShape[DimsGroups5D::Filter::KY];
    const auto KX = filterShape[DimsGroups5D::Filter::KX];
    VPUX_THROW_WHEN(KY == mlir::ShapedType::kDynamic || KX == mlir::ShapedType::kDynamic,
                    "NCEMatMulOp requires static KY/KX during bufferization");

    //
    // Prepare output buffer for DPU
    //
    const auto outputs =
            allocateNceOutputBuffers(log, origOp.getLoc(), rewriter, origOp.getOutput(), origOp.getReduceXyMax(),
                                     origOp.getReduceXyMin(), origOp.getReduceTensorMinMax());

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    const auto taskType = VPUIP::NCETaskType::CONV;
    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    bool isSuperdense = false;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdense = true;
    }

    const auto loopAttributes = getLoopAttributes(origOp);
    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{newArgs.getWeights(), newArgs.getWeightsTable(), nullptr,
                                          newArgs.getWeightTableScale(), newArgs.getWeightTableBias(),
                                          newArgs.getWeightZeroPoints()},
            outputs, taskType, NCEClusterTaskParams::Kernel{kernelSizeAttr, origOp.getStrides(), origOp.getPadAttr()},
            origOp.getWorkloads());
    params.isSuperdense = isSuperdense;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.mpeEngineAttr = origOp.getMpeEngineAttr();
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopTileIndex = loopAttributes.vfLoopTileIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// registerVpuNceBufferizableOpInterfaces
//

void vpux::registerVpuNceBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::NCEConvolutionOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEConvolutionOp>>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEMaxPoolOp>>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEAveragePoolOp>>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEDepthConvolutionOp>>(*ctx);
        VPU::NCEInterpolateOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEInterpolateOp>>(*ctx);
        VPU::NCEEltwiseOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEEltwiseOp>>(*ctx);
        VPU::NCECompressConvolutionOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCECompressConvolutionOp>>(
                *ctx);
        VPU::NCEPermuteOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEPermuteOp>>(*ctx);
        VPU::NCEMatMulOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEMatMulOp>>(*ctx);
        VPU::NCEReduceOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEReduceOp>>(*ctx);
    });
}
