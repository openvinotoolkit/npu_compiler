# Dynamic shapes bounds handling in NPU plugin

## Brief 
The NPU plugin implements a multi-layered bounds inference system to handle dynamic shapes in neural networks. When OpenVINO's shape inference cannot determine upper bounds for dynamic operations, the system provides fallback mechanisms at both the compiler frontend (pattern-based detection) and MLIR dialect level (operation-specific inference). This gives dynamic shapes the bounded dimensions required for efficient memory allocation and hardware execution on NPU devices. The HostCompile pipelines also compile a dimension without an upper bound, which is never given an invented bound; see "Error indicators" in the frontend section and "Strides of an unbounded shape".

## Problem Statement 
The NPU compiler requires bounded dynamic shapes to:
- Allocate memory efficiently during compilation
- Perform optimizations on operations with known upper bounds
- Execute models reliably on hardware with memory constraints

When receiving OpenVINO nGraph representation in the compiler frontend, nodes with dynamic shapes sometimes lack bounds information on their outputs. While OpenVINO's shape inference should ideally handle this, there are cases where bounds cannot be determined due to:
- Missing implementation in OpenVINO's shape inference
- Insufficient information to calculate output shapes from inputs
- Complex graph patterns where bounds propagation fails

Examples of problematic operations:
- **Range operation**: Generates sequences based on `start`, `stop`, and `step` values. When these are not constants, output size cannot be statically determined

## Solution Architecture

The solution is implemented as a multi-layered approach that handles bounds inference at different levels of the compilation pipeline. The architecture follows clean separation of concerns and leverages MLIR's type system to propagate bounds information throughout the compilation process.

### 1. Multi-Level Bounds Inference

The bounds detection system operates at two distinct levels:

#### MLIR Compiler Frontend Level
At the nGraph import stage, the system analyzes the OpenVINO graph representation to:
- Detect operations with missing bounds information, which we expect user to provide
- Detect operations, which in Dynamic case should be represented by different layers
- Update meta-information about network `NetworkInfo` according to changes done in ops on MLIR level

#### MLIR Dialect Level
Within individual operation implementations (`inferReturnTypeComponents`):
- Operations define their own bounds inference logic
- Type components carry bounds information through the compilation pipeline
- Bounds are computed based on operation semantics and input constraints
- Conservative bounds are applied when precise computation is not possible

### 2. Bounds Inference Strategies

#### Semantic-Based Inference
Operations implement bounds logic based on their mathematical semantics:
- Element-wise operations: bounds derived from input bounds and broadcast rules
- Reduction operations: bounds computed from reduction dimensions
- Shape-manipulating operations: bounds transformed according to operation parameters

#### Conservative Estimation
When precise bounds cannot be determined:
- Apply reasonable upper limits based on operation characteristics
- Use configurable defaults for unbounded scenarios
- Prioritize compilation success over optimal memory usage

## Shape Representation Rules

A dynamic dimension is stored as `ShapedType::kDynamic` in the shape itself. Its upper bound is stored
beside the shape, in the type's encoding attribute (`vpux::TensorAttr` for a tensor, `MemRefAttr` for a
memref). The rules below say which combinations of the two are legal.

### Allowed

| Shape | Encoding | Meaning |
|---|---|---|
| has `?` dimensions | `bounds`, one entry per dimension | the bounded dynamic form, and the one the hardware path needs |
| has `?` dimensions | no `bounds` | legal in the IE dialect. A pass can run before anything has bounds to propagate |
| fully static | `dynamic_dims_mask` | the other dynamic representation. It pairs a static shape with a mask and carries no bounds |
| fully static | neither | the ordinary case |

Avoid `dynamic_dims_mask` in new code; it is expected to be deprecated. Prefer dynamic shapes with
`bounds`.

A `bounds` vector holds exactly one entry per dimension, and no entry is negative.

Examples of each row, in the same order:

```mlir
tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
tensor<1x?x3x3xf32, {order = #NCHW}>
tensor<1x18x3x3xf32, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 1, 0, 0]> : tensor<4xsi64>, order = #NCHW}>
tensor<1x18x3x3xf32>
```

### Not allowed

Rejected when the type is parsed or constructed:

- A `bounds` vector whose length differs from the rank.
- A negative bound.

```mlir
// three bounds for four dimensions
tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3]> : tensor<3xsi64>}>
// error: TensorType bounds '[1, 18, 3]' do not match to shape '[1, -9223372036854775808, 3, 3]'

// a negative bound
tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, -1, 3, 3]> : tensor<4xsi64>}>
// error: TensorType bounds '[1, -1, 3, 3]' has negative value '-1' at dim '1'
```

A dynamic dimension prints as `-9223372036854775808` in these messages. That is
`ShapedType::kDynamic`, and not a corrupt value.

## SCF Tiling Bounds and Shape Refresh

SCF tiling can create dynamic `tensor.extract_slice` results even when the source
tensor is static, for example when a static extent is split into uneven tiles.
`VPU::generateTile` keeps or reconstructs a bounded result type using the following
order:

1. Preserve bounds already present on the inferred tile type. If the tiling
	information provides explicit `SCFTileInfo::bounds`, use those bounds for the
	dynamic tile dimensions.
2. For each remaining dynamic tile dimension, use a positive constant from the
	tile size or a positive upper bound computed by MLIR's
	`ValueBoundsConstraintSet` (`getConstantOrUpperBound`). If the corresponding
	source dimension is static, clamp this value to the source extent.
3. If the previous step cannot resolve the value, evaluate the tile-size
	expression with `OpChainAnalysis::MODE::MAX_VALUE`. Positive results are
	accepted and are likewise clamped to a static source extent.
4. If no positive bound can be derived, keep the dimension dynamic. A statically
	known source extent is used as the initial conservative fallback before the
	refinements above; an unknown source extent remains unbounded.

Non-positive constants and upper bounds are not usable tensor bounds and are
treated as unresolved. Bounds are only attached when every dynamic dimension has
an available bound; otherwise the inferred type is retained without a synthesized
bounded encoding.

After SCF tile-and-fuse, result shapes are refreshed in the loop's program order,
rather than in the order of the selected operation collection. This refresh runs
before deferred skip-connection slice replacement so later replacements see the
actual tiled result types. The replacement step then refreshes the transitive users
of the changed slices, restricted to the affected loops and again ordered by the
loop walk. Both refreshes re-infer only operations implementing
`InferTypeOpInterface`; utility operations such as `tensor.extract_slice`,
`tensor.insert_slice`, `tensor.cast`, and `tensor.pad` are skipped.

### Tolerated today, but not intended

These forms are still accepted because existing compiler transformations produce them, but they are
not intended representations.

**Bounds on a fully static shape** (`E#233281`). A fully static shape already specifies every
dimension exactly, so a separate bounds vector is redundant and should be omitted.

```mlir
tensor<1x18x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
```

**A bound that does not equal a static dimension** (`E#233282`). A static dimension has an exact size,
so its bounds entry should match it. A different value leaves the shape and bounds inconsistent.

```mlir
// Dim 2 is 3, but its bound is 6.
tensor<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 6, 3]> : tensor<4xsi64>, order = #NCHW}>
```

## Implementation Details

### SCF Tiling
**Location**: `src/vpux_compiler/src/dialect/VPU/utils/scf/scf_utils.cpp` and
`src/vpux_compiler/src/dialect/VPU/utils/tiling_algorithm/scf_tiling/scf_tiling.cpp`

**Purpose**:
- Preserve bounds while constructing dynamic SCF tiles.
- Provide conservative per-dimension bounds when tile sizes are dynamic.
- Re-infer tiled and fused result shapes after unrolling and after deferred
	skip-connection replacements.

The fallback order and refresh points are defined in
[SCF Tiling Bounds and Shape Refresh](#scf-tiling-bounds-and-shape-refresh).

### MLIR Compiler Frontend Level 
**Location**: `src/frontend/IE.{hpp,cpp}`

**Purpose**:
- `saveBoundsInfoForInput` is used to convert all parameters to `BoundedTensorType`. A parameter whose shape has an
axis with no upper bound is left as a plain dynamic tensor instead: the maximum value of the dimension type is not a
bound, and encoding it as one describes a buffer nothing could be sized against.
- `parseNode` can contain logic, dedicated to creating different operation, instead of original one in dynamic case.
For example, `DynamicReshapeOp` instead of `ReshapeOp`. 
- `updateModuleInfo` function is used to update information about bounds inside `NetworkInfo`, since it's created initially inside `addNetworkInfoOp` which is happening before we have MLIR representation meaning shape / bounds inference after creating MLIR representation for each node wouldn't be taken into account. 

**When to modify**: Add new pattern detection methods when you encounter graph patterns where bounds cannot be inferred by the OpenVINO framework's built-in shape inference.

**Error indicators**: The frontend does not reject a missing bound - a bound it cannot know is not the frontend's to
invent, and the HostCompile pipelines compile an unbounded axis without ever reaching for one. The failure is
reported where a bound or the strides are actually needed: memory allocation here, and the stride-dependent
consumers described in "Strides of an unbounded shape" below. Bufferization gives an unbounded DDR buffer a
sizeless `VPURT.Alloc` placeholder (an unbounded CMX buffer is rejected right away), and FuseLastCopy replaces it
with the network output buffer when the value is a network result. A placeholder that is still left when the DDR
`StaticAllocation` pass runs is rejected:
```
Cannot allocate a buffer of type 'memref<1x1x1x?xf16>': its shape has a dimension without an upper bound and it was
not replaced by a network output, so there is no size to allocate it with
```
### MLIR Dialect Level
**Location**: `src/dialect/IE/IR/ops/*.cpp` (individual operation files, this is IE dialect example)

**Purpose**: Operations implement semantic-based bounds inference through their `inferReturnTypeComponents()` methods.

**When to modify**: When adding new operations or when existing operations need bounds inference logic based on their mathematical semantics.

**Example**: Range operation sets bounds using `setBounds(Bounds{RANGEBOUND})` when output size cannot be statically determined.

### Type Verification Level
**Location**: `src/dialect/core/IR/dialect.cpp`

**Purpose**: `TensorEncodingVerifier` checks tensor bounds against the
[Shape Representation Rules](#shape-representation-rules) when types are parsed or constructed.
`MemRefAttr` verifies bounds for buffers.

**When to modify**: When a new field is added to `TensorAttr` or its validation rules change.

**Error indicators**: A type that breaks a rule fails at parse time with a `TensorType ...` message,
for example `TensorType bounds '[1, 18, 3]' do not match to shape '[1, -9223372036854775808, 3, 3]'`.

### Strides of an unbounded shape
**Location**: `src/dialect/core/interfaces/type_interfaces.cpp`, `src/utils/types.cpp`

**Purpose**: The strides of a buffer whose shape has a dimension without an upper bound are defined only at runtime
by the actual shape. MLIR represents them as partially known: `memref<1x?x16x16xf16>` has the strides
`[?, 256, 16, 1]` elements, and a consumer of `getStridesAndOffset` has to check for `?` before it computes with a
stride. Most of the compiler's consumers predate unbounded shapes and compute offsets, sizes and layout compatibility
from `Bit` strides without such a check, so a `kDynamic` entry would turn into a wrong offset rather than an error.

Therefore an unbounded buffer has only the identity layout, which in MLIR means dense in row-major order for
whatever the runtime shape is, and the compiler never computes, stores or compares its strides.
`NDTypeInterface::getStrides` and `getMemStrides` throw for such a type, and `getMemRefType` rejects explicit strides,
a non-identity order and any layout extras for it. A consumer that rebuilds a type with another component, such as
`changeMemSpace`, `changeTypeComponents`, `setSparsityCompressionAttr` or bufferization, takes the strides from
`getStridesUnlessUnbounded`, which returns none for an unbounded type and rejects one with an explicit `strided<>`
layout, because its unknown strides cannot be rebuilt. Any other consumer reaches `getStrides` and fails instead of
computing a wrong offset.

No hardware DMA accesses an unbounded buffer: only SwIoDma kernels move its data, from SHAVE, and the
`DMATypeOpInterface` verifier rejects a DMA on one. `AddCopyBetweenSWKernelsAndNetworkIO` rejects an unbounded
network IO used by any other SW kernel, since it would have to copy it. A SwIoDma kernel derives compact strides
from the runtime extents, so `LegalizeStridedDmas` rejects an unbounded network IO marked with dynamic strides.

**When to modify**: When a new consumer needs strides of an unbounded type. Use `getStridesUnlessUnbounded` only if
the consumer rebuilds the same type with another component; otherwise reject the type with an explicit message, as
GenericReshape and `isCompatibleForInplaceOp` do.

**Error indicators**: `Strides of 'memref<1x?x16x16xf16>' are unknown at compile time: a dimension has no upper
bound. Unbounded dynamic shapes are not supported here`, or a consumer-specific rejection such as
`GenericReshape does not support a shape without an upper bound: ...`.
