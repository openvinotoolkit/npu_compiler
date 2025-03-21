//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

namespace vpux::VPUIP {

enum TargetDevice {
    TargetDevice_NONE = 0,
    TargetDevice_MIN = TargetDevice_NONE,
    TargetDevice_VPUX30XX = 1,
    TargetDevice_VPUX37XX = 2,
    TargetDevice_VPUX311X = 3,
    TargetDevice_VPUX40XX = 4,
    TargetDevice_MAX
};

enum TargetDeviceRevision {
    TargetDeviceRevision_NONE = 0,
    TargetDeviceRevision_A0 = 1,
    TargetDeviceRevision_B0 = 2,
    TargetDeviceRevision_MIN = TargetDeviceRevision_NONE,
    TargetDeviceRevision_MAX = TargetDeviceRevision_B0
};

enum DType {
    DType_NOT_SET = 0,
    DType_FP64 = 1,
    DType_FP32 = 2,
    DType_FP16 = 3,
    DType_FP8 = 4,
    DType_U64 = 5,
    DType_U32 = 6,
    DType_U16 = 7,
    DType_U8 = 8,
    DType_I64 = 9,
    DType_I32 = 10,
    DType_I16 = 11,
    DType_I8 = 12,
    DType_I4 = 13,
    DType_I2 = 14,
    DType_I4X = 15,
    DType_BIN = 16,
    DType_LOG = 17,
    DType_I2X = 18,
    DType_BFP16 = 19,
    DType_U4 = 20,
    DType_BF8 = 21,
    DType_HF8 = 22,
    DType_MIN = DType_NOT_SET,
    DType_MAX = DType_HF8
};

enum MemoryLocation {
    MemoryLocation_NULL = 0,
    MemoryLocation_ProgrammableInput = 1,
    MemoryLocation_ProgrammableOutput = 2,
    MemoryLocation_VPU_DDR_Heap = 3,
    MemoryLocation_GraphFile = 4,
    MemoryLocation_VPU_CMX_NN = 5,
    MemoryLocation_VPU_DDR_BSS = 6,
    MemoryLocation_VPU_CSRAM = 7,
    MemoryLocation_AbsoluteAddr = 8,
    MemoryLocation_MAC_Accumulators = 9,
    MemoryLocation_ProfilingOutput = 10,
    MemoryLocation_GFEmbeddedKernel = 11,
    MemoryLocation_KernelsBuffer = 12,
    MemoryLocation_ExternalFile = 13,
    MemoryLocation_MIN = MemoryLocation_NULL,
    MemoryLocation_MAX = MemoryLocation_ExternalFile
};

enum OVNodeType {
    OVNodeType_UNDEFINED = 0,
    OVNodeType_DYNAMIC = 1,
    OVNodeType_BOOLEAN = 2,
    OVNodeType_BF16 = 3,
    OVNodeType_F16 = 4,
    OVNodeType_F32 = 5,
    OVNodeType_F64 = 6,
    OVNodeType_I4 = 7,
    OVNodeType_I8 = 8,
    OVNodeType_I16 = 9,
    OVNodeType_I32 = 10,
    OVNodeType_I64 = 11,
    OVNodeType_U1 = 12,
    OVNodeType_U4 = 13,
    OVNodeType_U8 = 14,
    OVNodeType_U16 = 15,
    OVNodeType_U32 = 16,
    OVNodeType_U64 = 17,
    OVNodeType_F8E4M3 = 18,
    OVNodeType_F8E5M2 = 19,
    OVNodeType_NF4 = 20,
    OVNodeType_MIN = OVNodeType_UNDEFINED,
    OVNodeType_MAX = OVNodeType_NF4,
};

enum DepthToSpaceMode {
    DepthToSpaceMode_BLOCKS_FIRST = 0,
    DepthToSpaceMode_DEPTH_FIRST = 1,
    DepthToSpaceMode_MIN = DepthToSpaceMode_BLOCKS_FIRST,
    DepthToSpaceMode_MAX = DepthToSpaceMode_DEPTH_FIRST
};

enum ROIAlignMethod {
    ROIAlignMethod_roi_align_avg = 0,
    ROIAlignMethod_roi_align_max = 1,
    ROIAlignMethod_MIN = ROIAlignMethod_roi_align_avg,
    ROIAlignMethod_MAX = ROIAlignMethod_roi_align_max
};

enum SpaceToDepthMode {
    SpaceToDepthMode_BLOCKS_FIRST = 0,
    SpaceToDepthMode_DEPTH_FIRST = 1,
    SpaceToDepthMode_MIN = SpaceToDepthMode_BLOCKS_FIRST,
    SpaceToDepthMode_MAX = SpaceToDepthMode_DEPTH_FIRST
};

enum PadMode {
    PadMode_Constant = 0,
    PadMode_Edge = 1,
    PadMode_Reflect = 2,
    PadMode_Symmetric = 3,
    PadMode_MIN = PadMode_Constant,
    PadMode_MAX = PadMode_Symmetric
};

enum RoundMode {
    RoundMode_HALF_TO_EVEN = 0,
    RoundMode_HALF_AWAY_FROM_ZERO = 1,
    RoundMode_MIN = RoundMode_HALF_TO_EVEN,
    RoundMode_MAX = RoundMode_HALF_AWAY_FROM_ZERO
};

enum PSROIPoolingMode {
    PSROIPoolingMode_AVERAGE = 0,
    PSROIPoolingMode_BILINEAR = 1,
    PSROIPoolingMode_BILINEAR_DEFORMABLE = 2,
    PSROIPoolingMode_MIN = PSROIPoolingMode_AVERAGE,
    PSROIPoolingMode_MAX = PSROIPoolingMode_BILINEAR_DEFORMABLE
};

enum M2IFormat {
    M2IFormat_PL_YUV444_8 = 0,
    M2IFormat_PL_YUV420_8 = 1,
    M2IFormat_PL_RGB24 = 2,
    M2IFormat_PL_RGB30 = 3,
    M2IFormat_PL_GRAY8 = 4,
    M2IFormat_PL_FP16_RGB = 5,
    M2IFormat_PL_FP16_YUV = 6,
    M2IFormat_PL_YUV422_8 = 7,
    M2IFormat_SP_NV12_8 = 8,
    M2IFormat_SP_NV12_10 = 9,
    M2IFormat_SP_P010 = 10,
    M2IFormat_IL_YUV422_8 = 24,
    M2IFormat_IL_RGB8888 = 25,
    M2IFormat_IL_RGB888 = 26,
    M2IFormat_IL_RGB30 = 27,
    M2IFormat_MIN = M2IFormat_PL_YUV444_8,
    M2IFormat_MAX = M2IFormat_IL_RGB30
};

enum M2IInterp {
    M2IInterp_NEAREST = 0,
    M2IInterp_BILINEAR = 1,
    M2IInterp_MIN = M2IInterp_NEAREST,
    M2IInterp_MAX = M2IInterp_BILINEAR
};

enum DeformablePSROIPoolingMode {
    DeformablePSROIPoolingMode_AVERAGE = 0,
    DeformablePSROIPoolingMode_BILINEAR = 1,
    DeformablePSROIPoolingMode_BILINEAR_DEFORMABLE = 2,
    DeformablePSROIPoolingMode_MIN = DeformablePSROIPoolingMode_AVERAGE,
    DeformablePSROIPoolingMode_MAX = DeformablePSROIPoolingMode_BILINEAR_DEFORMABLE
};

enum InterpolationMethod {
    InterpolationMethod_NEAREST = 0,
    InterpolationMethod_BILINEAR = 1,
    InterpolationMethod_BICUBIC = 2,
    InterpolationMethod_LINEARONNX = 3,
    InterpolationMethod_MIN = InterpolationMethod_NEAREST,
    InterpolationMethod_MAX = InterpolationMethod_LINEARONNX
};

enum InterpolationNearestMode {
    InterpolationNearestMode_ROUND_PREFER_FLOOR = 0,
    InterpolationNearestMode_ROUND_PREFER_CEIL = 1,
    InterpolationNearestMode_FLOOR = 2,
    InterpolationNearestMode_CEIL = 3,
    InterpolationNearestMode_SIMPLE = 4,
    InterpolationNearestMode_MIN = InterpolationNearestMode_ROUND_PREFER_FLOOR,
    InterpolationNearestMode_MAX = InterpolationNearestMode_SIMPLE
};

enum InterpolationCoordTransMode {
    InterpolationCoordTransMode_HALF_PIXEL = 0,
    InterpolationCoordTransMode_PYTORCH_HALF_PIXEL = 1,
    InterpolationCoordTransMode_ASYMMETRIC = 2,
    InterpolationCoordTransMode_TF_HALF_PIXEL_FOR_NN = 3,
    InterpolationCoordTransMode_ALIGN_CORNERS = 4,
    InterpolationCoordTransMode_MIN = InterpolationCoordTransMode_HALF_PIXEL,
    InterpolationCoordTransMode_MAX = InterpolationCoordTransMode_ALIGN_CORNERS
};

enum PhysicalProcessor {
    PhysicalProcessor_NULL = 0,
    PhysicalProcessor_LEON_RT = 1,
    PhysicalProcessor_LEON_NN = 2,
    PhysicalProcessor_NN_SHV = 3,
    PhysicalProcessor_ARM = 4,
    PhysicalProcessor_NCE_Cluster = 5,
    PhysicalProcessor_NCE_PerClusterDPU = 6,
    PhysicalProcessor_MIN = PhysicalProcessor_NULL,
    PhysicalProcessor_MAX = PhysicalProcessor_NCE_PerClusterDPU
};

enum PhysicalMem {
    PhysicalMem_NULL = 0,
    PhysicalMem_DDR = 1,
    PhysicalMem_NN_CMX = 2,
    PhysicalMem_CSRAM = 3,
    PhysicalMem_MIN = PhysicalMem_NULL,
    PhysicalMem_MAX = PhysicalMem_CSRAM
};

enum MPE_Mode {
    MPE_Mode_VECTOR = 0,
    MPE_Mode_MATRIX = 1,
    MPE_Mode_VECTOR_FP16 = 2,
    MPE_Mode_CUBOID_16x16 = 3,
    MPE_Mode_CUBOID_8x16 = 4,
    MPE_Mode_CUBOID_4x16 = 5,
    MPE_Mode_NOP = 6,
    MPE_Mode_MIN = MPE_Mode_VECTOR,
    MPE_Mode_MAX = MPE_Mode_NOP
};

enum Permutation {
    Permutation_ZXY = 0,
    Permutation_ZYX = 1,
    Permutation_YZX = 2,
    Permutation_YXZ = 3,
    Permutation_XZY = 4,
    Permutation_XYZ = 5,
    Permutation_MIN = Permutation_ZXY,
    Permutation_MAX = Permutation_XYZ
};

}  // namespace vpux::VPUIP
