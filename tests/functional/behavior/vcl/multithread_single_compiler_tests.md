# VCL Functional Multithread Test Guide (Single Compiler, Dummy Models)

This guide explains how to add a new multithread functional test that:
- creates one compiler handle,
- runs compilation-related work in multiple threads,
- validates a VCL API flow,
- reuses dummy models instead of JSON input files.

The tests in this folder share one root base and then split into two common shapes:
- `VCLFunctionalTestsCommon` provides dummy-model setup and parameter handling.
- `VCLAllocatedExecutableTestsBase` adds the single-compiler allocator helper used by allocated-executable tests.

Current concrete fixtures are:
- `VCLParallelCompilationBehaviorTest` for parallel executable creation/get-blob flows.
- `VCLQueryNetworkParallelBehaviorTest` for parallel query-network flows.
- `VCLAllocatedExecutableParallelCompilationBehaviorTest` for single-compiler allocated-executable flows.

Class hierarchy:

```text
VCLTestsCommon
├── VCLFunctionalTestsCommon (dummy-model SetUp)
│   ├── VCLParallelCompilationBehaviorTest
│   └── VCLQueryNetworkParallelBehaviorTest
└── VCLAllocatedExecutableTestsBase (single-compiler allocator helper)
    └── VCLAllocatedExecutableParallelCompilationBehaviorTest
```

Both `VCLFunctionalTestsCommon` and `VCLAllocatedExecutableTestsBase` inherit virtually from
`VCLTestsCommon`, so any fixture that combines them still shares one `VCLTestsCommon` subobject.

## 1. Where to add a new test source

Add a new `.cpp` under this folder:

- `tests/functional/behavior/vcl`

Examples in this folder:
- `parallel_compilation.cpp`
- `parallel_compilation_allocators.cpp`
- `parallel_query_network.cpp`

`tests/functional/CMakeLists.txt` (`npuFuncTests`) currently uses `ov_add_test_target` for this directory, so new `.cpp` files in `tests/functional/behavior/vcl` are picked up automatically.

## 2. Recommended test structure

### Adding to an existing fixture (recommended): `parallel_compilation_allocators.cpp`

`runParallelWithAllocator2` creates one compiler, sets up a `vcl_allocator2_t`, then
dispatches `numCompilationThreads` threads each running the caller-supplied lambda,
and returns the first failure or the compiler teardown result.

When adding a new API that is closely related to the existing allocator fixture (same concern, same setup), add a new `TEST_P` to `VCLAllocatedExecutableParallelCompilationBehaviorTest` rather than defining a new class.

Example:

```cpp
TEST_P(VCLAllocatedExecutableParallelCompilationBehaviorTest, ParallelCompilationWithAllocatorN) {
    setThreadCount(8);
    const auto ret = runParallelWithAllocator2(
            getNetOptions(),
            [](vcl_compiler_handle_t compiler, vcl_executable_desc_t exeDesc, vcl_allocator2_t& allocator,
               int /*i*/) -> vcl_result_t {
                // Call the new API under test, e.g. vclAllocatedExecutableCreateN(...)
                // Validate outputs.
                // Clean up allocated memory via allocator.deallocate(&allocator, ptr).
                return VCL_RESULT_SUCCESS;
            });
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run parallel vclAllocatedExecutableCreateN test! Result:0x"
                                       << std::hex << uint64_t(ret) << std::dec << std::endl;
}
```


#### Existing tests

| Test name | API exercised |
|---|---|
| `ParallelCompilationWithAllocator2` | `vclAllocatedExecutableCreate2` |
| `ParallelCompilationWithAllocator4` | `vclAllocatedExecutableCreate4` + `vclExecutableGetCompatibilityString` |
| `AllocatedExecutableCreateWSOneShot` | `vclAllocatedExecutableCreateWSOneShot` |

### Defining a new fixture

When the new API represents a genuinely different concern, define a new `BehaviorTest` class instead of adding another `TEST_P` to the existing fixture.

```cpp
class VCLMyApiParallelBehaviorTest : public VCLFunctionalTestsCommon {};

TEST_P(VCLMyApiParallelBehaviorTest, ParallelMyApiFlow) {
    setThreadCount(8);
    // test body here
}

INSTANTIATE_TEST_SUITE_P(ParallelMyApiTest, VCLMyApiParallelBehaviorTest, getVCLFunctionalTestParams(),
                         VCLMyApiParallelBehaviorTest::getTestCaseName);
```

Naming requirement:
- The new API test fixture class name must end with `BehaviorTest` to match the test filter.
- Example: `VCLAllocatedExecutableParallelCompilationBehaviorTest`.


## 3. Where to add models

Dummy functional models are defined in:

- `tests/functional/behavior/vcl/common.hpp`

Update `createDummyModel(const std::string& network)` to add a new model branch.

Then add its model key to the `networks` list in `buildDummyIRInfos()`.

Current model keys include:
- `split_concat`
- `matmul_bias`
- `conv_pool_relu`

## 4. Where to add devices

Device list for these functional multithread tests is also in:

- `tests/functional/behavior/vcl/common.hpp`

Edit the `devices` vector in `buildDummyIRInfos()`.

Current device IDs:
- `3720`
- `4000`
- `5010`
- `5020`

Each `(network, device)` pair becomes one parameterized test instance.

## 5. Parameter map contract

`VCLFunctionalTestsCommon::SetUp()` expects at least these keys in each param map:
- `network`
- `device`
- `info`


## 6. Thread-safety and cleanup checklist

- Keep one compiler shared across worker threads (single-compiler scenario).
- The test intentionally exercises concurrent VCL calls on the same compiler handle.
- Build executable/query descriptors before launching threads and do not mutate the
  model IR while threads are running; descriptors keep raw pointers into the model IR.
- Ensure per-thread outputs/handles are thread-local or protected.
- Always destroy executable/query handles in all paths (including error paths).
- Destroy compiler exactly once after all threads complete.
- Return first failing thread result for easier triage.
