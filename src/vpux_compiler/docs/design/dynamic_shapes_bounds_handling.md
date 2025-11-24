# Dynamic shapes bounds handling in NPU plugin

## Brief 
The NPU plugin implements a multi-layered bounds inference system to handle dynamic shapes in neural networks. When OpenVINO's shape inference cannot determine upper bounds for dynamic operations, the system provides fallback mechanisms at both the compiler frontend (pattern-based detection) and MLIR dialect level (operation-specific inference). This ensures all dynamic shapes have bounded dimensions required for efficient memory allocation and hardware execution on NPU devices.

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

## Implementation Details

### MLIR Compiler Frontend Level 
**Location**: `src/frontend/IE.{hpp,cpp}`

**Purpose**:
- `saveBoundsInfoForInput` is used to convert all parameters to `BoundedTensorType`
- `parseNode` can contain logic, dedicated to creating different operation, instead of original one in dynamic case.
For example, `DynamicReshapeOp` instead of `ReshapeOp`. 
- `isUpperBoundsMissing` dedicated for validation of created nodes, that all converted nodes, converted from ngraph to 
MLIR representation has valid upper bounds (from ngraph or calculated during node creation)
- `updateModuleInfo` function is used to update information about bounds inside `NetworkInfo`, since it's created initially inside `addNetworkInfoOp` which is happening before we have MLIR representation meaning shape / bounds inference after creating MLIR representation for each node wouldn't be taken into account. 

**When to modify**: Add new pattern detection methods when you encounter graph patterns where bounds cannot be inferred by the OpenVINO framework's built-in shape inference.

**Error indicators**: When a pattern is not handled by either OpenVINO shape propagation or MLIR frontend, compilation will fail with an error similar to:
```
Upper bounds are not specified for node 'Relu_70' (type 'Relu'): input '0' bounds are '[1, 9223372036854775807, 3]'
```
### MLIR Dialect Level
**Location**: `src/dialect/IE/IR/ops/*.cpp` (individual operation files, this is IE dialect example)

**Purpose**: Operations implement semantic-based bounds inference through their `inferReturnTypeComponents()` methods.

**When to modify**: When adding new operations or when existing operations need bounds inference logic based on their mathematical semantics.

**Example**: Range operation sets bounds using `setBounds(Bounds{RANGEBOUND})` when output size cannot be statically determined.
