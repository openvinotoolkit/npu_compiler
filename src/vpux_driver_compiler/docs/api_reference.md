# Intel® NPU Driver Compiler API Guide

This guide provides a comprehensive introduction to the NPU Driver Compiler API covering both current and deprecated functions, along with usage patterns to help you get started quickly.


## Table of Contents

- [API Overview](#1-api-overview)
- [Main Data Structures](#2-main-data-structures)
- [Basic Workflow](#3-basic-workflow)
  - [`vclExecutableCreate` Full Procedure](#31-vclexecutablecreate-full-procedure)
  - [`vclAllocatedExecutableCreate2` Full Procedure](#32-vclallocatedexecutablecreate2-full-procedure)
  - [`vclAllocatedExecutableCreate` Full Procedure (Deprecated)](#33-vclallocatedexecutablecreate-full-procedure-deprecated)
  - [Network Querying Workflow](#34-network-querying-workflow)
  - [Error Handling Workflow](#35-error-handling-workflow)
- [Detailed API Reference](#4-detailed-api-reference)
  - [Version and Properties Information](#41-version-and-properties-information)
  - [Compiler Lifecycle](#42-compiler-lifecycle)
  - [Network Capability Query](#43-network-capability-query)
  - [Executable Creation and Management](#44-executable-creation-and-management)
  - [Profiling Operations](#45-profiling-operations)
  - [Logging Functions](#46-logging-functions)
  - [Configuration Management](#47-configuration-management)
- [Frequently Asked Questions](#frequently-asked-questions)


## 1. API Overview

The NPU Driver Compiler API provides interfaces for compiling neural network models to Intel® Neural Processing Unit (NPU) devices. It consists of functions for Driver Compiler creation and management, network compilation, profiling, error logging and configuration management.


## 2. Main Data Structures

### Handle Types

- `vcl_compiler_handle_t` — Compiler object handle
- `vcl_executable_handle_t` — Executable object handle
- `vcl_profiling_handle_t` — Profiling object handle
- `vcl_query_handle_t` — Query network object handle
- `vcl_log_handle_t` — Log object handle

### Structs

- `vcl_version_info_t` — Version information
- `vcl_compiler_properties_t` — Compiler properties
- `vcl_profiling_properties_t` — Profiling properties
- `vcl_device_desc_t` — Device description
- `vcl_compiler_desc_t` — Compiler description
- `vcl_executable_desc_t` — Executable description
- `vcl_query_desc_t` — Query description
- `vcl_profiling_input_t`, `*p_vcl_profiling_input_t` — Profiling input
- `vcl_profiling_output_t`, `*p_vcl_profiling_output_t` — Profiling output
- `vcl_allocator_t`: Allocator V1 (deprecated)
- `vcl_allocator2_t`: Allocator V2 (recommended)

### VCL API Return Types

All vcl API functions return a `vcl_result_t` status code:
| Value                                | Description                       |
|--------------------------------------|-----------------------------------|
| VCL_RESULT_SUCCESS                   | Success                           |
| VCL_RESULT_ERROR_OUT_OF_MEMORY       | Insufficient memory               |
| VCL_RESULT_ERROR_UNSUPPORTED_FEATURE | Unsupported feature               |
| VCL_RESULT_ERROR_INVALID_ARGUMENT    | Invalid argument                  |
| VCL_RESULT_ERROR_INVALID_NULL_HANDLE | Invalid handle                    |
| VCL_RESULT_ERROR_IO                  | IO error                          |
| VCL_RESULT_ERROR_INVALID_IR          | Invalid IR                        |
| VCL_RESULT_ERROR_UNKNOWN             | Unknown/internal error            |


## 3. Basic Workflow

A typical workflow consists of the following steps:

1. **Get API version** using `vclGetVersion()`
2. **Create a compiler instance** using `vclCompilerCreate()`
3. **Perform network operations**:
   - (Optional) Query network capabilities with `vclQueryNetworkCreate()` and `vclQueryNetwork()`
   - Compile network:
        - Compile network with Executable:
            - Create executable with `vclExecutableCreate()`
            - Get compiled blob with `vclExecutableGetSerializableBlob()`
        - Compile network with AllocatedExecutable:
            - Create executable with `vclAllocatedExecutableCreate2()` or `vclAllocatedExecutableCreate() (Deprecated)`
4. **Profile execution** (Optional) using the profiling functions
5. **Handle errors** by retrieving logs with `vclLogHandleGetString()`
6. **Clean up resources** by destroying handles with the appropriate destroy functions

>Note: All objects created by `vclCompilerCreate`, `vclExecutableCreate`, and similar functions must be destroyed by their respective `Destroy` functions to avoid memory leaks.

### 3.1 `vclExecutableCreate` Full Procedure

```c
// 1. Create compiler and device description and instantiate compiler
vcl_compiler_desc_t compilerDesc = { ... };
vcl_device_desc_t deviceDesc = { ... };
vcl_compiler_handle_t compiler;
vcl_log_handle_t log;
vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, &log);

// 2. Prepare model IR (e.g. xml+weights) and optional parameters
vcl_executable_desc_t execDesc = {
    .modelIRData = ...,   // pointer to IR data
    .modelIRSize = ...,   // IR data size
    .options = ...,       // compiler options (optional)
    .optionsSize = ...,   // options size
};

// 3. Create executable object
vcl_executable_handle_t exec;
vcl_result_t ret = vclExecutableCreate(compiler, execDesc, &exec);

// 4. Export blob (if needed)
uint64_t blobSize;
vclExecutableGetSerializableBlob(exec, NULL, &blobSize);
uint8_t* blob = malloc(blobSize);
vclExecutableGetSerializableBlob(exec, blob, &blobSize);
// After use, free(blob);

// 5. Destroy executable object
vclExecutableDestroy(exec);

// 6. Destroy compiler object
vclCompilerDestroy(compiler);
```
>Note: For the configuration of the `options` field, please refer to the detailed content of [`vclAllocatedExecutableCreate2` API](#vclallocatedexecutablecreate2-recommended).

### 3.2 `vclAllocatedExecutableCreate2` Full Procedure

```c
// 1. Create compiler and device description and instantiate compiler
vcl_compiler_desc_t compilerDesc = { ... };
vcl_device_desc_t deviceDesc = { ... };
vcl_compiler_handle_t compiler;
vcl_log_handle_t log;
vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, &log);

// 2. Prepare model IR (e.g. xml+weights) and optional parameters
vcl_executable_desc_t execDesc = {
    .modelIRData = ...,   // pointer to IR data
    .modelIRSize = ...,   // IR data size
    .options = ...,       // compiler options (optional)
    .optionsSize = ...,   // options size
};

// 3. Create allocate and deallocate functions
uint8_t* my_allocate(vcl_allocator2_t* self, uint64_t size) { return (uint8_t*)malloc(size); }
void my_deallocate(vcl_allocator2_t* self, uint8_t* ptr) { free(ptr); }
vcl_allocator2_t allocator2 = { my_allocate, my_deallocate };

// 4. Compile and export blob
uint8_t* blob = NULL;
uint64_t blobSize = 0;
vclAllocatedExecutableCreate2(compiler, execDesc, &allocator2, &blob, &blobSize);

// 5. Use the allocated blob ...

// 6. Free blob
allocator2.deallocate(&allocator2, blob);

// 7. Destroy compiler object
vclCompilerDestroy(compiler);
```
>Note: For the configuration of the `options` field, please refer to the detailed content of [`vclAllocatedExecutableCreate2` API](#vclallocatedexecutablecreate2-recommended).

### 3.3 `vclAllocatedExecutableCreate` Full Procedure (Deprecated)

```c
// 1. Create compiler and device description and instantiate compiler
vcl_compiler_desc_t compilerDesc = { ... };
vcl_device_desc_t deviceDesc = { ... };
vcl_compiler_handle_t compiler;
vcl_log_handle_t log;
vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, &log);

// 2. Prepare model IR (e.g. xml+weights) and optional parameters
vcl_executable_desc_t execDesc = {
    .modelIRData = ...,   // pointer to IR data
    .modelIRSize = ...,   // IR data size
    .options = ...,       // compiler options (optional)
    .optionsSize = ...,   // options size
};

// 3. Create allocate and deallocate functions
uint8_t* my_allocate(uint64_t size) { return (uint8_t*)malloc(size); }
void my_deallocate(uint8_t* ptr) { free(ptr); }
vcl_allocator_t allocator = { my_allocate, my_deallocate };

// 4. Compile, export and free blob (if needed)
uint8_t* blob = NULL;
uint64_t blobSize = 0;
vclAllocatedExecutableCreate(compiler, execDesc, &allocator, &blob, &blobSize);

// 5. Use the allocated blob ...

// 6. Free blob
allocator.deallocate(blob);

// 7. Destroy compiler object
vclCompilerDestroy(compiler);
```
>Note: For the configuration of the `options` field, please refer to the detailed content of [`vclAllocatedExecutableCreate2` API](#vclallocatedexecutablecreate2-recommended).

### 3.4 Network Querying Workflow

```c
// 1. Create query
vcl_query_desc_t queryDesc = { /* initialize with IR data */ };
vcl_query_handle_t query;
vclQueryNetworkCreate(compiler, queryDesc, &query);

// 2. Get query result
uint64_t querySize = 0;
vclQueryNetwork(query, NULL, &querySize);
uint8_t* queryBuffer = (uint8_t*)malloc(querySize);
vclQueryNetwork(query, queryBuffer, &querySize);

// 3. Process query result - the format depends on implementation ...

// 4. Clean up resources
free(queryBuffer);
vclQueryNetworkDestroy(query);
```

### 3.5 Error Handling Workflow

```c
vcl_result_t result = vclSomeFunction(/* params */);
if (result != VCL_RESULT_SUCCESS) {
    // 1. Get log message if we have a log handle
    if (logHandle != NULL) {
        // Get size of log message
        size_t logSize = 0;
        vclLogHandleGetString(logHandle, &logSize, NULL);
        
        // Get content of log message
        char* logBuffer = (char*)malloc(logSize);
        vclLogHandleGetString(logHandle, &logSize, logBuffer);
        
        fprintf(stderr, "Error: %s\n", logBuffer);
        free(logBuffer);
    }
    
    // 2. Handle the error based on error code
    switch (result) {
        case VCL_RESULT_ERROR_OUT_OF_MEMORY:
            fprintf(stderr, "Out of memory\n");
            break;
        case VCL_RESULT_ERROR_INVALID_ARGUMENT:
            fprintf(stderr, "Invalid argument\n");
            break;
        // ... other error cases ...
        default:
            fprintf(stderr, "Unknown error: 0x%x\n", result);
            break;
    }
}
```

## 4. Detailed API Reference

### 4.1 Version and Properties Information

#### vclGetVersion

**Function**:
```c
vcl_result_t vclGetVersion(vcl_version_info_t* compilerVersion, vcl_version_info_t* profilingVersion);
```

**Purpose**: Retrieves the VCL API version.

**Parameters**:
| Parameter        | Type                | Direction | Description                           |
|------------------|---------------------|-----------|---------------------------------------|
| compilerVersion  | vcl_version_info_t* | [out]     | Returns the vcl API version           |
| profilingVersion | vcl_version_info_t* | [out]     | Returns the vcl API profiling version |


**Usage Example**:
```c
vcl_version_info_t compilerVersion, profilingVersion;
vcl_result_t result = vclGetVersion(&compilerVersion, &profilingVersion);
if (result == VCL_RESULT_SUCCESS) {
    printf("Compiler version: %d.%d\n", compilerVersion.major, compilerVersion.minor);
    printf("Profiling version: %d.%d\n", profilingVersion.major, profilingVersion.minor);
}
```

#### `vclCompilerGetProperties`

**Function**:
```c
vcl_result_t vclCompilerGetProperties(vcl_compiler_handle_t compiler, vcl_compiler_properties_t* properties);
```

**Purpose**: Retrieves the MLIR compiler version.

**Parameters**:
| Parameter  | Type                       | Direction | Description                 |
|------------|----------------------------|-----------|-----------------------------|
| compiler   | vcl_compiler_handle_t      | [in]      | The compiler handle         |
| properties | vcl_compiler_properties_t* | [out]     | Returns the MLIR properties |

**Usage Example**:
```c
vcl_compiler_properties_t properties;
result = vclCompilerGetProperties(compiler, &properties);
if (result == VCL_RESULT_SUCCESS) {
    printf("Compiler ID: %s\n", properties.id);
    printf("Supported opsets: 0x%x\n", properties.supportedOpsets);
}
```

#### `vclProfilingGetProperties`

**Function**:
```c
vcl_result_t vclProfilingGetProperties(vcl_profiling_handle_t profilingHandle, vcl_profiling_properties_t* properties);
```

**Purpose**: Retrieves properties of the profiling module.

**Parameters**:
| Parameter       | Type                        | Direction | Description            |
|-----------------|-----------------------------|-----------|------------------------|
| profilingHandle | vcl_profiling_handle_t      | [in]      | The profiling handle   |
| properties      | vcl_profiling_properties_t* | [out]     | Returns the properties |

**Usage Example**:
```c
vcl_profiling_properties_t profProps;
result = vclProfilingGetProperties(profHandle, &profProps);
if (result == VCL_RESULT_SUCCESS) {
    printf("Profiling version: %d.%d\n", 
           profProps.version.major, 
           profProps.version.minor);
}
```

### 4.2 Compiler Lifecycle

#### `vclCompilerCreate`

**Function**:
```c
vcl_result_t vclCompilerCreate(vcl_compiler_desc_t* compilerDesc, vcl_device_desc_t* deviceDesc, vcl_compiler_handle_t* compiler, vcl_log_handle_t* logHandle);
```

**Purpose**: Creates a compiler instance for a specific device.

**Parameters**:
| Parameter    | Type                   | Direction | Description                    |
|--------------|------------------------|-----------|--------------------------------|
| compilerDesc | vcl_compiler_desc_t*   | [in]      | Pointer to compiler descriptor |
| deviceDesc   | vcl_device_desc_t*     | [in]      | Pointer to device descriptor   |
| compiler     | vcl_compiler_handle_t* | [out]     | Returns the compiler handle    |
| logHandle    | vcl_log_handle_t*      | [out]     | Returns the log handle         |

**Usage Example**:
```c
vcl_compiler_desc_t compilerDesc = {
    .version = {7, 4},               // API version
    .debugLevel = VCL_LOG_INFO       // Debug level
};

vcl_device_desc_t deviceDesc = {
    .size = sizeof(vcl_device_desc_t),
    .deviceID = 0x1234,              // PCI Device ID in lower 16 bits
    .revision = 0,                   // NPU Revision (0 for first stepping)
    .tileCount = 1                   // Number of slices/tiles
};

vcl_compiler_handle_t compiler;
vcl_log_handle_t logHandle;
vcl_result_t result = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, &logHandle);
```

`vcl_log_level_t` type for `debugLevel` field of `vcl_compiler_desc_t` struct:
| Value           | Description            |
|-----------------|------------------------|
| VCL_LOG_NONE    | Logging disabled       |
| VCL_LOG_ERROR   | Error events           |
| VCL_LOG_WARNING | Warning events         |
| VCL_LOG_INFO    | Informational messages |
| VCL_LOG_DEBUG   | Debug messages         |
| VCL_LOG_TRACE   | Trace-level messages   |

#### `vclCompilerDestroy`

**Function**:
```c
vcl_result_t vclCompilerDestroy(vcl_compiler_handle_t compiler);
```

**Purpose**: Releases all resources associated with a compiler instance.

**Parameters**:
| Parameter    | Type                  | Direction | Description                                   |
|--------------|-----------------------|-----------|-----------------------------------------------|
| compiler     | vcl_compiler_handle_t | [in]      | Handle to the compiler object to be destroyed |

**Usage Example**:
```c
result = vclCompilerDestroy(compiler);
```

### 4.3 Network Capability Query

#### `vclQueryNetworkCreate`

**Function**:
```c
vcl_result_t vclQueryNetworkCreate(vcl_compiler_handle_t compiler, vcl_query_desc_t desc, vcl_query_handle_t* query);
```

**Purpose**: Creates a query to check what operations in a network can be executed on the NPU.

**Parameters**:
| Parameter | Type                  | Direction | Description                                           |
|-----------|-----------------------|-----------|-------------------------------------------------------|
| compiler  | vcl_compiler_handle_t | [in]      | The compiler handle                                   |
| desc      | vcl_query_desc_t      | [in]      | Query description including model IR data and options |
| query     | vcl_query_handle_t*   | [out]     | Returns the query handle                              |

**Usage Example**:
```c
vcl_query_desc_t queryDesc = {
    .modelIRData = irData,        // IR model data buffer
    .modelIRSize = irSize,        // Size of the IR data
    .options = options,           // Compiler options string
    .optionsSize = optionsLength  // Length of options string
};

vcl_query_handle_t query;
result = vclQueryNetworkCreate(compiler, queryDesc, &query);
```

#### `vclQueryNetwork`

**Function**:
```c
vcl_result_t vclQueryNetwork(vcl_query_handle_t query, uint8_t* queryResult, uint64_t* size);
```

**Purpose**: Retrieves the result of a network query, showing what operations are supported.

**Parameters**:
| Parameter   | Type               | Direction | Description                             |
|-------------|--------------------|-----------|-----------------------------------------|
| query       | vcl_query_handle_t | [in]      | The query handle                        |
| queryResult | uint8_t*           | [in]      | Buffer to receive the query result data |
| size        | uint64_t*          | [in,out]  | Pointer to size variable                |

**Usage Example**:
```c
// First call: get the required buffer size
uint64_t querySize = 0;
result = vclQueryNetwork(query, NULL, &querySize);

// Allocate buffer
uint8_t* queryBuffer = (uint8_t*)malloc(querySize);

// Second call: get the actual data
result = vclQueryNetwork(query, queryBuffer, &querySize);

// Process the query results ...

free(queryBuffer);
```

#### `vclQueryNetworkDestroy`

**Function**:
```c
vcl_result_t vclQueryNetworkDestroy(vcl_query_handle_t query);
```

**Purpose**: Destroys a query handle and releases associated resources.

**Parameters**:
| Parameter    | Type               | Direction | Description                                |
|--------------|--------------------|-----------|--------------------------------------------|
| query        | vcl_query_handle_t | [in]      | Handle to the query object to be destroyed |

**Usage Example**:
```c
result = vclQueryNetworkDestroy(query);
```

### 4.4 Executable Creation and Management

#### `vclExecutableCreate`

**Function**:
```c
vcl_result_t vclExecutableCreate(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc, vcl_executable_handle_t* executable);
```

**Purpose**:  Create an executable object. Compiles IR (such as OpenVINO IR xml and weights) into a NPU-executable blob, managed by internal cache.

**Parameters**:
| Parameter  | Type                     | Direction | Description                                                |
|------------|--------------------------|-----------|------------------------------------------------------------|
| compiler   | vcl_compiler_handle_t    | [in]      | The compiler handle                                        |
| desc       | vcl_executable_desc_t    | [in]      | Executable description including model IR data and options |
| executable | vcl_executable_handle_t* | [out]     | Returns the executable handle                              |

**Usage Example**:
```c
// Prepare IR Data: Arrange model IR (such as xml and weights) in memory ...

// Create Description:
    vcl_executable_desc_t execDesc = {
        .modelIRData = ...,   // Pointer to IR data buffer
        .modelIRSize = ...,   // IR data size
        .options = ...,       // Optional compiler options (NULL if not needed)
        .optionsSize = ...,   // Options size
    };

// Call the API:
    vcl_executable_handle_t exec;
    vcl_result_t ret = vclExecutableCreate(compiler, execDesc, &exec);
    if (ret != VCL_RESULT_SUCCESS) {
        // Error handling, e.g. use log API
    }
```

#### `vclExecutableGetSerializableBlob`

**Function**:
```c
vcl_result_t vclExecutableGetSerializableBlob(vcl_executable_handle_t executable, uint8_t* blobBuffer, uint64_t* blobSize);
```

**Purpose**: Retrieves the compiled blob from an executable.

**Parameters**:
| Parameter  | Type                    | Direction | Description                     |
|------------|-------------------------|-----------|---------------------------------|
| executable | vcl_executable_handle_t | [in]      | The executable handle           |
| blobBuffer | uint8_t*                | [in]      | Buffer to receive the blob data |
| blobSize   | uint64_t*               | [in,out]  | Pointer to size variable        |

**Usage Example**:
```c
// First call: get the required buffer size
uint64_t blobSize = 0;
result = vclExecutableGetSerializableBlob(executable, NULL, &blobSize);

// Allocate buffer
uint8_t* blobBuffer = (uint8_t*)malloc(blobSize);

// Second call: get the actual blob data
result = vclExecutableGetSerializableBlob(executable, blobBuffer, &blobSize);

// Process the blob ...

// Free the blob buffer
free(blobBuffer);
```

#### `vclExecutableDestroy`

**Function**:
```c
vcl_result_t vclExecutableDestroy(vcl_executable_handle_t executable);
```

**Purpose**: Destroys an executable and releases associated resources.

**Parameters**:
| Parameter    | Type                    | Direction | Description                                     |
|--------------|-------------------------|-----------|-------------------------------------------------|
| executable   | vcl_executable_handle_t | [in]      | Handle to the executable object to be destroyed |

**Usage Example**:
```c
result = vclExecutableDestroy(executable);
```

#### `vclAllocatedExecutableCreate2` (Recommended)

**Function**:
```c
vcl_result_t vclAllocatedExecutableCreate2(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc, vcl_allocator2_t* allocator, uint8_t** blobBuffer, uint64_t* blobSize);
```

**Purpose**: Creates an executable with a custom allocator for the blob buffer.

**Parameters**:
| Parameter  | Type                  | Direction | Description                                                |
|------------|-----------------------|-----------|------------------------------------------------------------|
| compiler   | vcl_compiler_handle_t | [in]      | The compiler handle                                        |
| desc       | vcl_executable_desc_t | [in]      | Executable description including model IR data and options |
| allocator  | vcl_allocator2_t*     | [in]      | Custom memory allocator and deallocator functions          |
| blobBuffer | uint8_t**             | [out]     | Pointer to receive the blob buffer pointer                 |
| blobSize   | uint64_t*             | [out]     | Pointer to receive the blob size                           |

**Usage Example**:
```c
// Prepare IR Data: Arrange model IR (such as xml and weights) in memory ...

// Create Description
    vcl_executable_desc_t execDesc = {
        .modelIRData = ...,   // Pointer to IR data buffer
        .modelIRSize = ...,   // IR data size
        .options = ...,       // Optional compiler options (NULL if not needed)
        .optionsSize = ...,   // Options size
    };

// Custom allocator and deallocator implementation
uint8_t* customAllocate(vcl_allocator2_t* allocator, uint64_t size) {
    return (uint8_t*)malloc(size);
}

void customDeallocate(vcl_allocator2_t* allocator, uint8_t* ptr) {
    free(ptr);
}

vcl_allocator2_t allocator = {
    .allocate = customAllocate,
    .deallocate = customDeallocate
};

// Call the API:
uint8_t* blobBuffer;
uint64_t blobSize;
result = vclAllocatedExecutableCreate2(
    compiler, execDesc, &allocator, &blobBuffer, &blobSize
);

// Use the allocated blob ...

// Free the buffer using the custom deallocate function
allocator.deallocate(&allocator, blobBuffer);
```

#### `vclAllocatedExecutableCreate` (Deprecated)

**Deprecated In Favor Of**: `vclAllocatedExecutableCreate2`
**Function**:
```c
vcl_result_t vclAllocatedExecutableCreate(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc, vcl_allocator_t const* allocator, uint8_t** blobBuffer, uint64_t* blobSize);
```

**Purpose**: Creates an executable with a custom allocator for the blob buffer. 
>Note: Avoid using this function in new code. The deprecated `vcl_allocator_t` structure doesn't include context information for the allocator functions, making it less flexible than the newer `vcl_allocator2_t` structure.

**Parameters**:
| Parameter  | Type                  | Direction | Description                                                |
|------------|-----------------------|-----------|------------------------------------------------------------|
| compiler   | vcl_compiler_handle_t | [in]      | The compiler handle                                        |
| desc       | vcl_executable_desc_t | [in]      | Executable description including model IR data and options |
| allocator  | vcl_allocator_t       | [in]      | Custom memory allocator and deallocator functions          |
| blobBuffer | uint8_t**             | [out]     | Pointer to receive the blob buffer pointer                 |
| blobSize   | uint64_t*             | [out]     | Pointer to receive the blob size                           |

**Usage Example**:

```c
// Prepare IR Data: Arrange model IR (such as xml and weights) in memory ...
// Create Description
    vcl_executable_desc_t execDesc = {
        .modelIRData = ...,   // Pointer to IR data buffer
        .modelIRSize = ...,   // IR data size
        .options = ...,       // Optional compiler options (NULL if not needed)
        .optionsSize = ...,   // Options size
    };

// Custom allocator and deallocator implementation
uint8_t* my_allocate(uint64_t size) { return (uint8_t*)malloc(size); }
void my_deallocate(uint8_t* ptr) { free(ptr); }
vcl_allocator_t allocator = { my_allocate, my_deallocate };

// Call the API
uint8_t* blob = NULL;
uint64_t blobSize = 0;
vclAllocatedExecutableCreate(compiler, execDesc, &allocator, &blob, &blobSize);
// Use the allocated blob ...

// Free the buffer using the custom deallocate function
allocator.deallocate(blob);
```

### 4.5 Profiling Operations

#### `vclProfilingCreate`

**Function**:
```c
vcl_result_t vclProfilingCreate(p_vcl_profiling_input_t profilingInput, vcl_profiling_handle_t* profilingHandle, vcl_log_handle_t* logHandle);
```

**Purpose**: Creates a profiling handle to analyze execution performance data.

**Parameters**:
| Parameter       | Type                    | Direction | Description                                  |
|-----------------|-------------------------|-----------|----------------------------------------------|
| profilingInput  | p_vcl_profiling_input_t | [in]      | Input data including blob and profiling data |
| profilingHandle | vcl_profiling_handle_t* | [out]     | Pointer to receive the profiling handle      |
| logHandle       | vcl_log_handle_t*       | [out]     | Pointer to receive the log handle            |

**Usage Example**:
```c
vcl_profiling_input_t profilingInput = {
    .blobData = blobBuffer,   // Compiled blob data
    .blobSize = blobSize,     // Size of blob data
    .profData = profRawData,  // Raw profiling data from execution
    .profSize = profRawSize   // Size of raw profiling data
};

vcl_profiling_handle_t profHandle;
vcl_log_handle_t profLogHandle;
result = vclProfilingCreate(&profilingInput, &profHandle, &profLogHandle);
```

#### `vclGetDecodedProfilingBuffer`

**Function**:
```c
vcl_result_t vclGetDecodedProfilingBuffer(vcl_profiling_handle_t profilingHandle, vcl_profiling_request_type_t requestType, p_vcl_profiling_output_t profilingOutput);
```

**Purpose**: Retrieves decoded profiling information for the requested detail level.

**Parameters**:
| Parameter       | Type                         | Direction | Description                                              |
|-----------------|------------------------------|-----------|----------------------------------------------------------|
| profilingHandle | vcl_profiling_handle_t       | [in]      | The profiling handle                                     |
| requestType     | vcl_profiling_request_type_t | [in]      | Type of profiling data to retrieve (layer, task, or raw) |
| profilingOutput | p_vcl_profiling_output_t     | [out]     | Pointer to receive the output data                       |

*vcl_profiling_request_type_t* type:
| Value                     | Description           |
|---------------------------|-----------------------|
| VCL_PROFILING_LAYER_LEVEL | Layer-level profiling |
| VCL_PROFILING_TASK_LEVEL  | Task-level profiling  |
| VCL_PROFILING_RAW         | Raw profiling data    |

**Usage Example**:
```c
vcl_profiling_output_t profOutput;
result = vclGetDecodedProfilingBuffer(
    profHandle, VCL_PROFILING_LAYER_LEVEL, &profOutput
);
if (result == VCL_RESULT_SUCCESS) {
    // Process layer-level profiling data
    // profOutput.data contains the decoded information
    // profOutput.size is the size of the data
}
```

#### `vclProfilingDestroy`

**Function**:
```c
vcl_result_t vclProfilingDestroy(vcl_profiling_handle_t profilingHandle);
```

**Purpose**: Destroys a profiling handle and releases associated resources.

**Parameters**:
| Parameter       | Type                   | Direction | Description                                    |
|-----------------|------------------------|-----------|------------------------------------------------|
| profilingHandle | vcl_profiling_handle_t | [in]      | Handle to the profiling object to be destroyed |

**Usage Example**:
```c
result = vclProfilingDestroy(profHandle);
```

### 4.6 Logging Functions

#### `vclLogHandleGetString`

**Function**:
```c
vcl_result_t vclLogHandleGetString(vcl_log_handle_t logHandle, size_t* logSize, char* log);
```

**Purpose**: Retrieves error/debug messages from a log handle.

**Parameters**:
| Parameter | Type             | Direction | Description                      |
|-----------|------------------|-----------|----------------------------------|
| logHandle | vcl_log_handle_t | [in]      | The log handle                   |
| logSize   | size_t*          | [in,out]  | Pointer to size variable         |
| log       | char*            | [out]     | Buffer to receive the log string |

**Usage Example**:
```c
// First call: get the required buffer size
size_t logSize = 0;
result = vclLogHandleGetString(logHandle, &logSize, NULL);

// Allocate buffer
char* logBuffer = (char*)malloc(logSize);

// Second call: get the actual log data
result = vclLogHandleGetString(logHandle, &logSize, logBuffer);

// Process the log ...
printf("Log message: %s\n", logBuffer);

free(logBuffer);
```

### 4.7 Configuration Management

#### `vclGetCompilerSupportedOptions`

**Function**:
```c
vcl_result_t vclGetCompilerSupportedOptions(vcl_compiler_handle_t compiler, char* result, uint64_t* size);
```

**Purpose**: Retrieves the list of compiler options supported by this version of the compiler.

**Parameters**:
| Parameter | Type                  | Direction | Description                    |
|-----------|-----------------------|-----------|--------------------------------|
| compiler  | vcl_compiler_handle_t | [in]      | The compiler handle            |
| result    | char*                 | [out]     | Buffer to receive options data |
| size      | uint64_t*             | [in,out]  | Pointer to size variable       |

**Usage Example**:
```c
// First call: get the required buffer size
uint64_t optionsSize = 0;
result = vclGetCompilerSupportedOptions(compiler, NULL, &optionsSize);

// Allocate buffer
char* optionsBuffer = (char*)malloc(optionsSize);

// Second call: get the actual options data
result = vclGetCompilerSupportedOptions(compiler, optionsBuffer, &optionsSize);

// Process the options data ...
printf("Supported options: %s\n", optionsBuffer);

free(optionsBuffer);
```

#### `vclGetCompilerIsOptionSupported`

**Function**:
```c
vcl_result_t vclGetCompilerIsOptionSupported(vcl_compiler_handle_t compiler, const char* option, const char* value);
```

**Purpose**: Checks if a given config option (or option-value pair) is supported by the compiler.

**Parameters**:
| Parameter | Type                  | Direction | Description                         |
|-----------|-----------------------|-----------|-------------------------------------|
| compiler  | vcl_compiler_handle_t | [in]      | The compiler handle                 |
| option    | const char*           | [in]      | Option name to check                |
| value     | const char*           | [in]      | Option value to check (can be NULL) |

**Usage Example**:
```c
result = vclGetCompilerIsOptionSupported(compiler, "NPU_PLATFORM", "4000");
if (result == VCL_RESULT_SUCCESS) {
    printf("NPU_PLATFORM=4000 is supported\n");
} else {
    printf("NPU_PLATFORM=4000 is not supported\n");
}
```


## Frequently Asked Questions

- **Q: What is the difference between `vclExecutableCreate` and `vclAllocatedExecutableCreate2`?**  
  A: vclExecutableCreate manages memory internally while vclAllocatedExecutableCreate2 lets you control memory allocation through custom allocators.
  
- **Q: How do I choose between layer-level and task-level profiling?**  
  A: Use layer-level for high-level performance analysis and task-level for detailed optimization.

## References

- Header file: [npu_driver_compiler.h](../include/npu_driver_compiler.h)
- For detailed structure and parameter descriptions, refer to the header file comments.

---

This guide covers the main APIs of the NPU Driver Compiler. If you need more detailed parameter explanations or code samples, please refer to the header file or contact the development support team.