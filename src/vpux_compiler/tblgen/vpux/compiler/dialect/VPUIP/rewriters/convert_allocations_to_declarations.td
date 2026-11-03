//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_REWRITERS_CONVERT_ALLOCATIONS_TO_DECLARATIONS
#define VPUX_COMPILER_DIALECT_VPUIP_REWRITERS_CONVERT_ALLOCATIONS_TO_DECLARATIONS

include "vpux/compiler/dialect/VPUIP/ops.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

//
// VPUIP.StaticAlloc -> VPUIP.DeclareTensor
//

def createDeclareDDRTensorOp :
        NativeCodeCall<[{
            vpux::VPURT::DeclareBufferOp::create($_builder,
                $_loc, mlir::cast<mlir::MemRefType>($0.getType()),
                vpux::VPURT::BufferSection::DDR,
                $1.getInt()
            )
        }]>;

def createDeclareCMXTensorOp :
        NativeCodeCall<[{
            vpux::VPURT::DeclareBufferOp::create($_builder,
                $_loc, mlir::cast<mlir::MemRefType>($0.getType()),
                vpux::VPURT::BufferSection::CMX_NN, mlir::cast<vpux::NDTypeInterface>($0.getType()).getMemSpace().getIndex().value(),
                $1.getInt()
            )
        }]>;

class HasMemSpace<string physicalMemory> :
        Constraint<CPred<[{
            mlir::cast<vpux::NDTypeInterface>($0.getType()).getMemoryKind() == vpux::VPU::MemoryKind::}] # physicalMemory
        >>;

def HasIndex:
        Constraint<CPred<"mlir::cast<vpux::NDTypeInterface>($0.getType()).getMemSpace().getIndex().has_value()">>;

def RewriteStaticAllocDDR :
        Pat<
            (VPUIP_StaticAllocOp:$memref $offset),
            (createDeclareDDRTensorOp $memref, $offset),
            [
                (HasMemSpace<"DDR"> $memref)
            ]
        >;

def RewriteStaticAllocNNCMX :
        Pat<
            (VPUIP_StaticAllocOp:$memref $offset),
            (createDeclareCMXTensorOp $memref, $offset),
            [
                (HasMemSpace<"CMX_NN"> $memref),
                (HasIndex $memref)
            ]
        >;

#endif
