//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <llvm/Support/JSON.h>

#include <array>
#include <fstream>
#include <set>
#include <string>

namespace nb {
enum class CaseType {
    DMA,
    ZMajorConvolution,
    SparseZMajorConvolution,
    DepthWiseConv,
    EltwiseAdd,
    EltwiseMult,
    MaxPool,
    AvgPool,
    DifferentClustersDPU,
    MultiClustersDPU,
    HaloMultiClustering,
    ActShave,
    ReadAfterWriteDPUDMA,
    ReadAfterWriteDMADPU,
    ReadAfterWriteACTDMA,
    ReadAfterWriteDMAACT,
    ReadAfterWriteDPUACT,
    ReadAfterWriteACTDPU,
    RaceConditionDMA,
    RaceConditionDPU,
    RaceConditionDPUDMA,
    RaceConditionDPUDMAACT,
    RaceConditionDPUACT,
    RaceCondition,
    StorageElementTableDPU,
    DualChannelDMA,
    Unknown
};

std::string to_string(CaseType case_);
CaseType to_case(llvm::StringRef str);

enum class CompilerBackend { Flatbuffer, ELF };

std::string to_string(CompilerBackend compilerBackend);
llvm::Optional<nb::CompilerBackend> to_compiler_backend(llvm::StringRef str);

enum class DType { U4, I4, U8, I8, I32, FP8, FP16, FP32, BF16, UNK };

enum class MemoryLocation { CMX0, CMX1, DDR, Unknown };
MemoryLocation to_memory_location(llvm::StringRef str);
std::string to_string(MemoryLocation memoryLocation);

DType to_dtype(llvm::StringRef str);
std::string to_string(DType dtype);
MVCNN::Permutation to_odu_permutation(llvm::StringRef str);
struct QuantParams {
    bool present = false;
    double scale = 0.;
    std::int64_t zeropoint = 0;
    std::int64_t low_range = 0;
    std::int64_t high_range = 0;
};

enum class SegmentationType { SOK = 1, SOH = 2, SOW = 3, SOHW = 4, SOHK = 5 };
std::string to_string(SegmentationType segmentationType);

enum class SwizzlingKey { key0, key1, key2, key3, key4, key5 };

template <typename E>
constexpr auto to_underlying(E e) -> typename std::underlying_type<E>::type {
    return static_cast<typename std::underlying_type<E>::type>(e);
}

struct Shape {};

// Input and weight layers have similar structure in rtl config descriptor
struct InputLayer {
    DType dtype = DType::UNK;
    QuantParams qp;
    std::array<std::int64_t, 4> shape = {0};
};

struct WeightLayer {
    DType dtype = DType::UNK;
    QuantParams qp;
    std::array<std::int64_t, 4> shape = {0};
    std::string filename;
};

struct DMAparams {
    MemoryLocation srcLocation = MemoryLocation::Unknown;
    MemoryLocation dstLocation = MemoryLocation::Unknown;
    int64_t engine = 0;
};

struct ConvLayer {
    std::array<std::int64_t, 2> stride = {0};
    std::array<std::int64_t, 4> pad = {0};
    std::int64_t group = 0;
    std::int64_t dilation = 0;
    bool compress = false;
    vpux::VPU::MPEMode cube_mode = vpux::VPU::MPEMode::CUBOID_16x16;
};

struct PoolLayer {
    std::string pool_type = "max";
    std::array<std::int64_t, 2> kernel_shape = {0};
    std::array<std::int64_t, 2> stride = {0};
    std::array<std::int64_t, 4> pad = {0};
    std::int64_t group = 0;
    std::int64_t dilation = 0;
};

struct RaceConditionParams {
    size_t iterationsCount = 0;
    size_t requestedClusters = 0;

    // requested DPU's per cluster for DPUop case
    // requested ActShave's per cluster for ActShaveOp case
    // requested DMA engines for DMAop case
    size_t requestedUnits = 0;
};

struct DPUTaskParams {
    std::size_t inputCluster = 0;
    std::vector<std::size_t> outputClusters;
    std::size_t weightsCluster = 0;
    std::size_t weightsTableCluster = 0;
};

struct MultiClusterDPUParams {
    std::vector<std::size_t> taskClusters;
    SegmentationType segmentation = SegmentationType::SOK;
    bool broadcast = false;
};

struct HaloParams {
    // clusters on which the DPUTasks will run
    std::vector<std::size_t> taskClusters;
    // type of multiclustering tensor segmentation (SOK, SOH, SOW etc.)
    SegmentationType segmentation = SegmentationType::SOK;
    // number of clusters assigned per each dim, for segmentation on multiple dims
    // e.g. SOHK clustersPerDim = {2, 3} -> height is split in 2, output channels are split in 3
    // e.g. SOHW clustersPerDim = {3, 2} -> height is split in 3, width is split in 2
    std::vector<std::size_t> clustersPerDim;
    // halo size over height, to be used for strategies SOH, SOHK
    int64_t heightHaloSize = 0;
    // halo size over width, to be used for strategies SOW, SOHW
    int64_t widthHaloSize = 0;
};

struct OutputLayer {
    std::array<std::int64_t, 4> shape = {0};
    DType dtype = DType::UNK;
    QuantParams qp;
};

enum class ActivationType {
    None,
    ReLU,
    ReLUX,
    LeakyReLU,
    Mish,
    HSwish,
    Sigmoid,
    Softmax,
    vau_sigm,
    vau_sqrt,
    vau_tanh,
    vau_log,
    vau_exp,
    lsu_b16,
    lsu_b16_vec,
    vau_dp4,
    vau_dp4a,
    vau_dp4m,
    sau_dp4,
    sau_dp4a,
    sau_dp4m,

    Unknown
};

ActivationType to_activation_type(llvm::StringRef str);
std::string to_string(ActivationType activationType);

struct ActivationLayer {
    ActivationType activationType = ActivationType::None;
    double alpha = 0.;
    double maximum = 0;
    size_t axis = 0;
    // TODO: add support for activation functions that take parameters
};

enum class SETablePattern { SwitchLines, OriginalInput };

class TestCaseJsonDescriptor {
public:
    TestCaseJsonDescriptor(llvm::StringRef jsonString = "");
    TestCaseJsonDescriptor(llvm::json::Object jsonObject);
    void parse(llvm::json::Object jsonObject);
    llvm::SmallVector<InputLayer> getInputLayerList() const {
        return inLayers_;
    }
    llvm::SmallVector<WeightLayer> getWeightLayers() const {
        return wtLayer_;
    }
    llvm::SmallVector<OutputLayer> getOutputLayers() const {
        return outLayers_;
    }
    DMAparams getDMAparams() const {
        return DMAparams_;
    }
    ConvLayer getConvLayer() const {
        return convLayer_;
    }
    PoolLayer getPoolLayer() const {
        return poolLayer_;
    }
    ActivationLayer getActivationLayer() const {
        return activationLayer_;
    }
    RaceConditionParams getRaceConditionParams() const {
        return raceConditionParams_;
    }
    DPUTaskParams getDPUTaskParams() const {
        return DPUTaskParams_;
    }
    MultiClusterDPUParams getMultiClusterDPUParams() const {
        return multiClusterDPUParams_;
    }
    HaloParams getHaloParams() const {
        return haloParams_;
    }
    CaseType getCaseType() const {
        return caseType_;
    }
    llvm::StringRef getKernelFilename() const {
        return kernelFilename_;
    }
    std::string getCaseStr() const {
        return caseTypeStr_;
    }
    vpux::VPU::PPEMode getPPELayerType() const {
        return ppeLayerType_;
    }
    MVCNN::Permutation getODUPermutation() const {
        return odu_permutation_;
    }
    std::size_t getIterationCount() const {
        return iterationCount_;
    }
    std::size_t getClusterNumber() const {
        return clusterNumber_;
    }
    SwizzlingKey getWeightsSwizzlingKey() const {
        return weightsSwizzlingKey_;
    }
    SwizzlingKey getActivationSwizzlingKey() const {
        return activationSwizzlingKey_;
    }
    SETablePattern getSETablePattern() const {
        return seTablePattern_;
    }
    std::shared_ptr<TestCaseJsonDescriptor> getUnderlyingOp() const {
        return underlyingOp_;
    }

    vpux::VPU::ArchKind getArchitecture() const {
        return architecture_;
    }

    CompilerBackend getCompilerBackend() const {
        return compilerBackend_;
    }

private:
    llvm::SmallVector<InputLayer> loadInputLayer(llvm::json::Object* jsonObj);
    llvm::SmallVector<WeightLayer> loadWeightLayer(llvm::json::Object* jsonObj);
    llvm::SmallVector<OutputLayer> loadOutputLayer(llvm::json::Object* jsonObj);
    DMAparams loadDMAParams(llvm::json::Object* jsonObj);
    ConvLayer loadConvLayer(llvm::json::Object* jsonObj);
    PoolLayer loadPoolLayer(llvm::json::Object* jsonObj);
    ActivationLayer loadActivationLayer(llvm::json::Object* jsonObj);
    CaseType loadCaseType(llvm::json::Object* jsonObj);
    QuantParams loadQuantizationParams(llvm::json::Object* obj);
    RaceConditionParams loadRaceConditionParams(llvm::json::Object* obj);
    DPUTaskParams loadDPUTaskParams(llvm::json::Object* obj);
    MultiClusterDPUParams loadMultiClusterDPUParams(llvm::json::Object* obj);
    HaloParams loadHaloTaskParams(llvm::json::Object* jsonObj);
    std::size_t loadIterationCount(llvm::json::Object* obj);
    std::size_t loadClusterNumber(llvm::json::Object* obj);
    SETablePattern loadSETablePattern(llvm::json::Object* obj);
    SwizzlingKey loadSwizzlingKey(llvm::json::Object* obj, std::string keyType);

    CaseType caseType_ = CaseType::Unknown;
    DMAparams DMAparams_;
    ConvLayer convLayer_;
    PoolLayer poolLayer_;
    llvm::SmallVector<InputLayer> inLayers_;
    llvm::SmallVector<WeightLayer> wtLayer_;
    llvm::SmallVector<OutputLayer> outLayers_;
    ActivationLayer activationLayer_;
    bool hasActivationLayer_ = false;
    std::string kernelFilename_;
    std::string caseTypeStr_;
    vpux::VPU::PPEMode ppeLayerType_ = vpux::VPU::PPEMode::ADD;
    MVCNN::Permutation odu_permutation_ = MVCNN::Permutation::Permutation_ZXY;
    std::size_t iterationCount_ = 0;
    std::size_t clusterNumber_ = 0;
    std::shared_ptr<TestCaseJsonDescriptor> underlyingOp_;
    SwizzlingKey weightsSwizzlingKey_ = SwizzlingKey::key0;
    SwizzlingKey activationSwizzlingKey_ = SwizzlingKey::key0;
    RaceConditionParams raceConditionParams_;
    DPUTaskParams DPUTaskParams_;
    MultiClusterDPUParams multiClusterDPUParams_;
    HaloParams haloParams_;
    SETablePattern seTablePattern_ = SETablePattern::SwitchLines;
    vpux::VPU::ArchKind architecture_ = vpux::VPU::ArchKind::UNKNOWN;
    CompilerBackend compilerBackend_ = CompilerBackend::Flatbuffer;
};

}  // namespace nb
