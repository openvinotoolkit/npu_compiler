//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <variant>
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "level_zero_wrapper/level_zero_wrapper.h"
#include "npu_interpreter_runtime/npu_vm_runtime.hpp"
#include "openvino/util/file_util.hpp"
#include "vpux/compiler/network_metadata.hpp"
#include "vpux/utils/logger/logger.hpp"

#if defined(_WIN32)
#pragma warning(push)
#pragma warning(disable : 4244 4267 4146 4996)
#endif
#include <llvm/Support/Error.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#if defined(_WIN32)
#pragma warning(pop)
#endif

using namespace intel_npu;

#if defined(_WIN32)
#define MLIR_ZERO_WRAPPER_FILE_NAME "openvino_intel_npu_level_zero_wrapper.dll"
#else
#define MLIR_ZERO_WRAPPER_FILE_NAME "libopenvino_intel_npu_level_zero_wrapper.so"
#endif

class DebugTrace {
public:
    DebugTrace(const std::string& funcName, vpux::Logger logger = vpux::Logger::global())
            : _funcName(funcName), _logger(logger) {
        _logger.trace("{0} start", _funcName);
    }
    ~DebugTrace() {
        _logger.trace("{0} end", _funcName);
    }
    DebugTrace(const DebugTrace&) = delete;
    DebugTrace& operator=(const DebugTrace&) = delete;

private:
    std::string _funcName;
    vpux::Logger _logger;
};

// Same with the one in level_zero_wrapper.h
struct MemRefNDRef {
    constexpr static size_t headerSize = 3;  // allocatedPtr, alignedPtr, offset

    int64_t* bufferPtr;
    int64_t dimCount;
    MemRefNDRef(int64_t* buffer, int64_t dim_count): bufferPtr(buffer), dimCount(dim_count) {
    }

    void setAllocated(const void* buf) {
        bufferPtr[0] = reinterpret_cast<int64_t>(buf);
    }

    void setAligned(const void* buf) {
        bufferPtr[1] = reinterpret_cast<int64_t>(buf);
    }

    void setOffset(int64_t offset) {
        bufferPtr[2] = static_cast<int64_t>(offset);
    }

    template <typename T>
    void setSizes(T* size, int64_t dimCount) {
        int64_t* ptr = bufferPtr + headerSize;
        for (int64_t i = 0; i < dimCount; ++i) {
            ptr[i] = static_cast<int64_t>(size[i]);
        }
    }

    template <typename T>
    void setStrides(T* strides, int64_t dimCount) {
        int64_t* ptr = bufferPtr + headerSize + dimCount;
        for (int64_t i = 0; i < dimCount; ++i) {
            ptr[i] = static_cast<int64_t>(strides[i]);
        }
    }

    void* getAllocated() {
        return reinterpret_cast<void*>(bufferPtr[0]);
    }

    void* getAligned() {
        return reinterpret_cast<void*>(bufferPtr[1]);
    }

    int64_t getOffset() {
        return bufferPtr[2];
    }

    int64_t* getSizes() {
        return reinterpret_cast<int64_t*>(bufferPtr + headerSize);
    }

    int64_t* getStrides() {
        return reinterpret_cast<int64_t*>(bufferPtr + headerSize + dimCount);
    }
};

struct MemRefHandle {
    int64_t* memRefBufferPtr;
    int64_t dimCount;

    MemRefHandle(int64_t dim_count): memRefBufferPtr(nullptr), dimCount(dim_count) {
        int64_t numElements = MemRefNDRef::headerSize + dimCount * 2;
        memRefBufferPtr = new int64_t[numElements];
        for (int64_t i = 0; i < numElements; ++i) {
            memRefBufferPtr[i] = 0;
        }
    }
    ~MemRefHandle() {
        if (memRefBufferPtr != nullptr) {
            delete[] memRefBufferPtr;
            memRefBufferPtr = nullptr;
        }
    }

    MemRefHandle(const MemRefHandle&) = delete;
    MemRefHandle& operator=(const MemRefHandle&) = delete;

    int64_t getMemRefBufferNumElements() {
        return MemRefNDRef::headerSize + dimCount * 2;
    }

    int64_t getMemRefBufferByteSize() {
        return getMemRefBufferNumElements() * sizeof(uint64_t);
    }

    void* getAllocated() {
        MemRefNDRef ref(memRefBufferPtr, dimCount);
        return ref.getAllocated();
    }

    void parseMemRef(const void** pBasePtr, const void** pData, int64_t* pOffset, int64_t* pSizes, int64_t* pStrides,
                     int64_t* pDimsCount) {
        MemRefNDRef ref(memRefBufferPtr, dimCount);
        *pBasePtr = ref.getAllocated();
        *pData = ref.getAligned();
        *pOffset = static_cast<int64_t>(ref.getOffset());
        int64_t* sizes = ref.getSizes();
        int64_t* strides = ref.getStrides();
        for (int64_t i = 0; i < dimCount; ++i) {
            pSizes[i] = sizes[i];
            pStrides[i] = strides[i];
        }
        *pDimsCount = dimCount;
    }

    std::string toString() {
        std::string result = "MemRefHandle(dimCount=" + std::to_string(dimCount) + ", buffer=[";
        for (int64_t i = 0; i < getMemRefBufferNumElements(); ++i) {
            result += std::to_string(memRefBufferPtr[i]);
            if (i < getMemRefBufferNumElements() - 1) {
                result += ", ";
            }
        }
        result += "])";
        return result;
    }
};

// Internal implementation helpers use the shared VM runtime interface types.
class NPUMLIRRuntime {
public:
    NPUMLIRRuntime(const npu_vm_runtime_blob_desc_t* desc, npu_vm_runtime_properties_t* pProperties);
    ~NPUMLIRRuntime();

    NPUMLIRRuntime(const NPUMLIRRuntime&) = delete;
    NPUMLIRRuntime& operator=(const NPUMLIRRuntime&) = delete;

    void createExecutionEngine(const npu_vm_runtime_blob_desc_t* blob);

    void parseMetadata();

    void getArgumentProperties(uint32_t argIndex, ze_graph_argument_properties_3_t* pGraphArgumentProperties,
                               ze_graph_argument_metadata_t* pGraphArgumentMetadata);

    void execute(npu_vm_runtime_execute_params_t* pParams);

    void predictOutputShape(npu_vm_runtime_predict_output_shape_params_t* pParams);

    void createExecutionContext(npu_vm_runtime_execution_context_handle_t* phExecutionContextHandle);
    void destroyExecutionContext(npu_vm_runtime_execution_context_handle_t hExecutionContextHandle);
    void updateMutableCommandList(npu_vm_runtime_execute_params_t* pParams, uint64_t* argIndexArray,
                                  uint64_t argIndexArraySize);
    void executeMutableCommandList(npu_vm_runtime_execute_params_t* pParams, uint64_t* argIndexArray,
                                   uint64_t argIndexArraySize);

private:
    std::unique_ptr<mlir::MLIRContext> _context;
    mlir::DialectRegistry _registry;
    std::unique_ptr<mlir::ExecutionEngine> _engine;

    vpux::NetworkMetadata _metadata;
    // use ze_graph_argument_properties_3_t instead of ArgumentDescriptor in metadata function in the future
    std::vector<ArgumentDescriptor> _inputs;
    std::vector<ArgumentDescriptor> _outputs;
    uint64_t _numOfSubgraphs = 0;
    uint64_t _numOfNetworkArgs = 0;
    vpux::Logger _logger = vpux::Logger("NPUMLIRRuntime", vpux::Logger::global().level());
};

class ExecutionContext {
public:
    NPUMLIRRuntime* _runtime;
    void* _executionContextHandle = nullptr;

public:
    ExecutionContext(NPUMLIRRuntime* runtime): _runtime(runtime) {
    }
};

void NPUMLIRRuntime::createExecutionEngine(const npu_vm_runtime_blob_desc_t* desc) {
    DebugTrace dt("createExecutionEngine", _logger);
    _logger.debug("Creating execution engine from blob at {0} of size {1}", desc->pInput, desc->inputSize);
    const std::string adapterPrefix = std::string("_mlir_ciface_");
    const std::string entryName = "main";
    const std::string adapterName = adapterPrefix + entryName;

    auto blobPtr = desc->pInput;
    auto blobSize = desc->inputSize;

    // Metadata<METADATA_VERSION_X_X> is stored after LLVM code in CompiledModel::export_model
    // So, the file size needs to be adjusted to avoid compilation error
    auto getLLVMIRSize = [](const uint8_t* llvmIR, size_t size) {
        if (size == 0 || llvmIR == nullptr) {
            return 0ULL;
        }
        for (size_t index = size; index-- > 0;) {
            if (llvmIR[index] == static_cast<uint8_t>('}')) {
                return index + 1ULL;
            }
        }
        return 0ULL;
    };

    llvm::StringRef content(reinterpret_cast<const char*>(blobPtr), getLLVMIRSize(blobPtr, blobSize));
    auto llvmBlob = llvm::MemoryBuffer::getMemBufferCopy(content, "LLVMBlob");
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(llvmBlob), llvm::SMLoc());
    mlir::OwningOpRef<mlir::Operation*> module = mlir::parseSourceFile<mlir::ModuleOp>(*sourceMgr, _context.get());

    if (!module) {
        OPENVINO_THROW("Failed to parse LLVM IR");
    }

    _logger.debug("Creating JITTargetMachineBuilder");
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
        OPENVINO_THROW("Failed to detect host");
    }
    _logger.debug("Creating TargetMachine for {0}", tmBuilderOrError->getCPU());
    _logger.debug("Target triple {0}", tmBuilderOrError->getTargetTriple().normalize());

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
        OPENVINO_THROW("Failed to create TargetMachine");
    }
    _logger.debug("TargetMachine created");

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;

    llvm::SmallVector<mlir::StringRef, 4> sharedLibs;
    static std::string libpath = (ov::util::get_ov_lib_path() / MLIR_ZERO_WRAPPER_FILE_NAME).string();
    sharedLibs.push_back(libpath);
    engineOptions.sharedLibPaths = sharedLibs;
    _logger.debug("Creating engine");
    auto expectedEngine = mlir::ExecutionEngine::create(*module, engineOptions, std::move(tmOrError.get()));
    if (!expectedEngine) {
        OPENVINO_THROW("Failed to create ExecutionEngine");
    }
    _logger.debug("Engine created");
    _engine = std::move(*expectedEngine);
    auto expectedFPtr = _engine->lookupPacked(entryName);

    if (!expectedFPtr) {
        OPENVINO_THROW("Failed to lookup main function");
    }
}

void NPUMLIRRuntime::parseMetadata() {
    DebugTrace dt("parseMetadata", _logger);
    std::string getNetworkMetadataFuncName = "get_network_metadata";

    // Get metadata and number of graph
    auto error = _engine->invoke(getNetworkMetadataFuncName, &_metadata, &_numOfSubgraphs, &_inputs, &_outputs);
    if (error) {
        OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
    }
    _logger.debug("num of subgraphs: {0} inputs: {1} outputs: {2}", _numOfSubgraphs, _inputs.size(), _outputs.size());
    _metadata.bindRelatedDescriptors();
    _numOfNetworkArgs = static_cast<uint32_t>(_inputs.size() + _outputs.size());
    for (size_t i = 0; i < _inputs.size(); i++) {
        _metadata.inputs[i].indexUsedByDriver = _inputs[i].idx;
    }

    for (size_t i = 0; i < _outputs.size(); i++) {
        _metadata.outputs[i].indexUsedByDriver = _outputs[i].idx;
    }
}

NPUMLIRRuntime::NPUMLIRRuntime(const npu_vm_runtime_blob_desc_t* desc, npu_vm_runtime_properties_t* pProperties) {
    DebugTrace dt("Constructor", _logger);
    // Initialize MLIR context and register necessary dialects
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    mlir::registerAllToLLVMIRTranslations(_registry);

    _context = std::make_unique<mlir::MLIRContext>(_registry);

    createExecutionEngine(desc);

    parseMetadata();

    pProperties->numOfSubGraphs = static_cast<uint32_t>(_numOfSubgraphs);
    pProperties->numOfGraphArgs = static_cast<uint32_t>(_numOfNetworkArgs);
}

NPUMLIRRuntime::~NPUMLIRRuntime() {
    DebugTrace dt("Destructor", _logger);
    _engine.reset();
    _context.reset();
}

void NPUMLIRRuntime::getArgumentProperties(uint32_t argIndex,
                                           ze_graph_argument_properties_3_t* pGraphArgumentProperties,
                                           ze_graph_argument_metadata_t* pGraphArgumentMetadata) {
    DebugTrace dt("getArgumentProperties", _logger);
    _logger.debug("Getting argument properties for index {0}", argIndex);
    if (argIndex >= _numOfNetworkArgs) {
        OPENVINO_THROW("Invalid argument index {0} >= {1}", argIndex, _numOfNetworkArgs);
    }

    const ArgumentDescriptor* argDesc = nullptr;
    vpux::IODescriptor desc;
    if (argIndex < _inputs.size()) {
        argDesc = &_inputs[argIndex];
        desc = _metadata.inputs[argIndex];
    } else {
        argDesc = &_outputs[argIndex - _inputs.size()];
        desc = _metadata.outputs[argIndex - _inputs.size()];
    }

    // Define new struct to hold metadata
    *pGraphArgumentProperties = argDesc->info;

    // Fill in metadata struct
    pGraphArgumentMetadata->stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_METADATA;
    pGraphArgumentMetadata->pNext = nullptr;
    pGraphArgumentMetadata->type = argDesc->info.type;
    std::strncpy(pGraphArgumentMetadata->friendly_name, argDesc->info.name, ZE_MAX_GRAPH_ARGUMENT_NAME - 1);
    pGraphArgumentMetadata->data_type = ZE_GRAPH_METADATA_TYPE_UNDEFINED;
    pGraphArgumentMetadata->friendly_name[ZE_MAX_GRAPH_ARGUMENT_NAME - 1] = '\0';

    if (desc.shapeFromIRModel.has_value()) {
        // Only care about shape, this is shapeFromIRModel
        for (size_t i = 0; i < desc.shapeFromIRModel->size() && i < ZE_MAX_GRAPH_TENSOR_REF_DIMS; ++i) {
            auto val = desc.shapeFromIRModel.value()[i];
            pGraphArgumentMetadata->shape[i] =
                    val.is_dynamic() ? std::numeric_limits<uint64_t>::max() : val.get_length();
        }
    } else {
        // Use shapeFromCompiler
        std::copy(std::begin(desc.shapeFromCompiler.get_shape()), std::end(desc.shapeFromCompiler.get_shape()),
                  std::begin(pGraphArgumentMetadata->shape));
    }
    pGraphArgumentMetadata->shape_size = argDesc->info.dims_count;
    pGraphArgumentMetadata->tensor_names_count = 0;  // Not used
    std::strncpy(pGraphArgumentMetadata->input_name, argDesc->info.name, ZE_MAX_GRAPH_ARGUMENT_NAME - 1);
    pGraphArgumentMetadata->input_name[ZE_MAX_GRAPH_ARGUMENT_NAME - 1] = '\0';

    // Dump argDesc info
    _logger.debug("Argument Descriptor Info:");
    _logger.debug("  Name: {0}", argDesc->info.name);
    _logger.debug("  Type: {0}", argDesc->info.type);
    _logger.debug("  Dimensions: {0}", argDesc->info.dims_count);
    for (size_t i = 0; i < argDesc->info.dims_count; ++i) {
        _logger.debug("    Dim[{0}]: {1}", i, argDesc->info.dims[i]);
    }

    // Dump metadata info
    _logger.debug("Graph Argument Metadata Info:");
    _logger.debug("  Input Name: {0}", pGraphArgumentMetadata->input_name);
    _logger.debug("  Shape Size: {0}", pGraphArgumentMetadata->shape_size);
    for (size_t i = 0; i < pGraphArgumentMetadata->shape_size; ++i) {
        _logger.debug("    Shape[{0}]: {1}", i, pGraphArgumentMetadata->shape[i]);
    }
}

void NPUMLIRRuntime::execute(npu_vm_runtime_execute_params_t* pParams) {
    if (pParams == nullptr) {
        OPENVINO_THROW("Invalid execute parameters");
    }
    DebugTrace dt("execute", _logger);
    _logger.debug("Executing with {0} inputs and {1} outputs", pParams->numOfInputs, pParams->numOfOutputs);

    // reset execution context if provided
    if (pParams->executionContext != nullptr) {
        _logger.debug("Resetting execution context");
        ExecutionContext* execCtx = reinterpret_cast<ExecutionContext*>(pParams->executionContext);
        mlir::SmallVector<void*> packedResetArgs;
        mlir::ExecutionEngine::Argument<void*>::pack(packedResetArgs, execCtx->_executionContextHandle);
        mlir::ExecutionEngine::Argument<ze_command_list_handle_t*>::pack(packedResetArgs, pParams->commandLists);
        mlir::ExecutionEngine::Argument<uint64_t>::pack(packedResetArgs, pParams->numCommandLists);
        const std::string resetFuncName = "_mlir_ciface_reset_execution_context";
        auto error = _engine->invokePacked(resetFuncName, packedResetArgs);
        if (error) {
            OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
        }
    }

    mlir::SmallVector<void*> packedArgs;
    for (uint32_t i = 0; i < pParams->numOfInputs; ++i) {
        auto handle = reinterpret_cast<MemRefHandle*>(pParams->pInputs[i]);
        mlir::ExecutionEngine::Argument<int64_t*>::pack(packedArgs, handle->memRefBufferPtr);
        _logger.debug("Input : {0}, info: {1}", i, handle->toString().c_str());
    }
    for (uint32_t i = 0; i < pParams->numOfOutputs; ++i) {
        auto handle = reinterpret_cast<MemRefHandle*>(pParams->pOutputs[i]);
        mlir::ExecutionEngine::Argument<int64_t*>::pack(packedArgs, handle->memRefBufferPtr);
        _logger.debug("Output : {0}, info: {1}", i, handle->toString().c_str());
    }

    // execution context is not used now, pass a dummy nullptr
    // This is reserved for future use to support mutable command list
    ExecutionContext* execCtx = reinterpret_cast<ExecutionContext*>(pParams->executionContext);
    mlir::ExecutionEngine::Argument<ze_context_handle_t>::pack(packedArgs, pParams->ctx);
    mlir::ExecutionEngine::Argument<ze_device_handle_t>::pack(packedArgs, pParams->device);
    mlir::ExecutionEngine::Argument<ze_graph_dditable_ext_t*>::pack(packedArgs, pParams->graphDdiTableExt);
    mlir::ExecutionEngine::Argument<ze_command_list_handle_t*>::pack(packedArgs, pParams->commandLists);
    mlir::ExecutionEngine::Argument<uint64_t>::pack(packedArgs, pParams->numCommandLists);
    mlir::ExecutionEngine::Argument<ze_command_queue_handle_t>::pack(packedArgs, pParams->commandQueue);
    mlir::ExecutionEngine::Argument<ze_fence_handle_t>::pack(packedArgs, pParams->inferenceFence);
    mlir::ExecutionEngine::Argument<ze_event_handle_t>::pack(packedArgs, pParams->event);
    void* dummyExecutionContextHandle = nullptr;
    if (pParams->executionContext != nullptr) {
        mlir::ExecutionEngine::Argument<void*>::pack(packedArgs, execCtx->_executionContextHandle);
    } else {
        mlir::ExecutionEngine::Argument<void*>::pack(packedArgs, dummyExecutionContextHandle);
    }

    const std::string adapterName = "_mlir_ciface_main";
    auto error = _engine->invokePacked(adapterName, packedArgs);
    if (error) {
        OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
    }
}

void NPUMLIRRuntime::predictOutputShape(npu_vm_runtime_predict_output_shape_params_t* params) {
    DebugTrace dt("predictOutputShape", _logger);

    for (uint32_t i = 0; i < params->numOfInputs; i++) {
        MemRefHandle* input = reinterpret_cast<MemRefHandle*>(params->pInputs[i]);
        _logger.debug("Input : {0}, info: {1}", i, input->toString());
    }

    const std::string predictFuncName = "_mlir_ciface_output_shape";
    auto expectedFPtr = _engine->lookupPacked(predictFuncName);
    if (!expectedFPtr) {
        _logger.debug("Can not find predict func ptr, remain original value of output");

        for (uint32_t i = 0; i < params->numOfOutputs; i++) {
            MemRefHandle* output = reinterpret_cast<MemRefHandle*>(params->pOutputs[i]);
            _logger.debug("Fake Output : {0}, info: {1}", i, output->toString());
        }
        return;
    } else {
        mlir::SmallVector<void*> packedArgs;

        for (uint32_t i = 0; i < params->numOfInputs; ++i) {
            MemRefHandle* input = reinterpret_cast<MemRefHandle*>(params->pInputs[i]);
            mlir::ExecutionEngine::Argument<int64_t*>::pack(packedArgs, input->memRefBufferPtr);
        }

        // Prepare container to save predict result
        std::vector<std::shared_ptr<MemRefHandle>> predictedOutputs;
        predictedOutputs.reserve(params->numOfOutputs);
        std::vector<std::vector<uint64_t>> outputShapes;
        outputShapes.reserve(params->numOfOutputs);
        for (uint32_t i = 0; i < params->numOfOutputs; ++i) {
            MemRefHandle* output = reinterpret_cast<MemRefHandle*>(params->pOutputs[i]);
            // Create a local MemRefHandle containing a tensor with a single dimension (dimCount).
            // The tensor size is set to dimCount for the target tensor.
            // The final prediction results are stored in the handle's basePtr.
            std::shared_ptr<MemRefHandle> handleForPredict = std::make_shared<MemRefHandle>(1);
            // Create reference of the buffer inside handleForPredict and update it with right value
            MemRefNDRef refForHandle(handleForPredict->memRefBufferPtr, 1);
            outputShapes.push_back(std::vector<uint64_t>(output->dimCount, 0));
            void* basePtr = outputShapes[i].data();
            refForHandle.setAllocated(basePtr);
            refForHandle.setAligned(basePtr);
            refForHandle.setOffset(0);
            refForHandle.setSizes(&(output->dimCount), 1);
            // Stride is always 1 now since we always use dense tensor now
            int64_t stride = 1;
            refForHandle.setStrides(&stride, 1);
            predictedOutputs.push_back(handleForPredict);
            mlir::ExecutionEngine::Argument<int64_t*>::pack(packedArgs, handleForPredict->memRefBufferPtr);
        }

        auto error = _engine->invokePacked(predictFuncName, packedArgs);
        if (error) {
            OPENVINO_THROW("Error invoking output_shape: " + llvm::toString(std::move(error)));
        }

        // Update original output to have right info
        for (uint32_t i = 0; i < params->numOfOutputs; i++) {
            MemRefHandle* output = reinterpret_cast<MemRefHandle*>(params->pOutputs[i]);
            MemRefNDRef ref(output->memRefBufferPtr, output->dimCount);
            auto& predictedShape = outputShapes[i];
            if (predictedShape.size() != output->dimCount) {
                OPENVINO_THROW("Predicted dim count is not the same as the original one!");
            }
            ref.setSizes(predictedShape.data(), predictedShape.size());
            // Only size matters, calc stride by hand
            std::vector<int64_t> strides;
            strides.resize(predictedShape.size());
            int64_t stride = 1;
            for (int64_t i = predictedShape.size() - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= predictedShape[i];
            }
            ref.setStrides(strides.data(), strides.size());
            _logger.debug("Output : {0}, info: {1}", i, output->toString());
        }
    }
}

void NPUMLIRRuntime::createExecutionContext(npu_vm_runtime_execution_context_handle_t* phExecutionContextHandle) {
    if (!phExecutionContextHandle) {
        OPENVINO_THROW("phExecutionContextHandle is null");
    }
    if (!_engine) {
        OPENVINO_THROW("MLIR ExecutionEngine is not initialized");
    }

    auto pContext = std::make_unique<ExecutionContext>(this);
    const std::string funcName = "_mlir_ciface_create_execution_context";
    auto expectedFPtr = _engine->lookupPacked(funcName);
    if (!expectedFPtr) {
        OPENVINO_THROW("Function " + funcName + " not found in MLIR module");
    }

    mlir::SmallVector<void*> packedArgs;
    auto pThis = this;
    mlir::ExecutionEngine::Argument<NPUMLIRRuntime*>::pack(packedArgs, pThis);
    void* executionContextHandlePtr = &(pContext->_executionContextHandle);
    mlir::ExecutionEngine::Argument<uint64_t>::pack(packedArgs, this->_numOfSubgraphs);
    mlir::ExecutionEngine::Argument<uint64_t>::pack(packedArgs, this->_numOfNetworkArgs);
    mlir::ExecutionEngine::Argument<void*>::pack(packedArgs, executionContextHandlePtr);
    auto error = _engine->invokePacked(funcName, packedArgs);
    if (error) {
        OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
    }

    *phExecutionContextHandle = reinterpret_cast<npu_vm_runtime_execution_context_handle_t>(pContext.release());
}

void NPUMLIRRuntime::destroyExecutionContext(npu_vm_runtime_execution_context_handle_t hExecutionContextHandle) {
    ExecutionContext* pContext = reinterpret_cast<ExecutionContext*>(hExecutionContextHandle);

    const std::string funcName = "_mlir_ciface_destroy_execution_context";
    auto expectedFPtr = _engine->lookupPacked(funcName);
    if (!expectedFPtr) {
        OPENVINO_THROW("Function " + funcName + " not found in MLIR module");
    }

    mlir::SmallVector<void*> packedArgs;
    mlir::ExecutionEngine::Argument<void*>::pack(packedArgs, pContext->_executionContextHandle);
    auto error = _engine->invokePacked(funcName, packedArgs);
    if (error) {
        OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
    }
    pContext->_executionContextHandle = nullptr;
}

void NPUMLIRRuntime::updateMutableCommandList(npu_vm_runtime_execute_params_t* pParams, uint64_t* argIndexArray,
                                              uint64_t argIndexArraySize) {
    ExecutionContext* execCtx = reinterpret_cast<ExecutionContext*>(pParams->executionContext);
    if (execCtx == nullptr || execCtx->_executionContextHandle == nullptr) {
        OPENVINO_THROW("null execution context");
    }

    const std::string funcName = "_mlir_ciface_update_mutable_command_list";
    auto expectedFPtr = _engine->lookupPacked(funcName);
    if (!expectedFPtr) {
        OPENVINO_THROW("Function " + funcName + " not found in MLIR module");
    }
    mlir::SmallVector<void*> packedArgs;
    uint64_t numArgs = pParams->numOfInputs + pParams->numOfOutputs;
    std::vector<uint64_t> networkArgArray(numArgs);
    networkArgArray.reserve(numArgs);
    for (uint64_t i = 0; i < pParams->numOfInputs; i++) {
        auto memRefHandle = reinterpret_cast<MemRefHandle*>(pParams->pInputs[i]);
        const uint64_t allocatedPtr = reinterpret_cast<uint64_t>(memRefHandle->getAllocated());
        networkArgArray[i] = allocatedPtr;
    }
    for (uint64_t i = 0; i < pParams->numOfOutputs; i++) {
        auto memRefHandle = reinterpret_cast<MemRefHandle*>(pParams->pOutputs[i]);
        const uint64_t allocatedPtr = reinterpret_cast<uint64_t>(memRefHandle->getAllocated());
        networkArgArray[i + pParams->numOfInputs] = allocatedPtr;
    }

    mlir::ExecutionEngine::Argument<void*>::pack(packedArgs, execCtx->_executionContextHandle);
    auto networkArgArrayPtr = networkArgArray.data();
    mlir::ExecutionEngine::Argument<uint64_t*>::pack(packedArgs, networkArgArrayPtr);
    mlir::ExecutionEngine::Argument<uint64_t>::pack(packedArgs, numArgs);
    mlir::ExecutionEngine::Argument<uint64_t*>::pack(packedArgs, argIndexArray);
    mlir::ExecutionEngine::Argument<uint64_t>::pack(packedArgs, argIndexArraySize);

    auto error = _engine->invokePacked(funcName, packedArgs);
    if (error) {
        OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
    }
}

void NPUMLIRRuntime::executeMutableCommandList(npu_vm_runtime_execute_params_t* pParams, uint64_t* argIndexArray,
                                               uint64_t argIndexArraySize) {
    ExecutionContext* execCtx = reinterpret_cast<ExecutionContext*>(pParams->executionContext);
    if (execCtx == nullptr || execCtx->_executionContextHandle == nullptr) {
        OPENVINO_THROW("null execution context");
    }

    const std::string funcName = "_mlir_ciface_execute_mutable_command_list";
    auto expectedFPtr = _engine->lookupPacked(funcName);
    if (!expectedFPtr) {
        OPENVINO_THROW("Function " + funcName + " not found in MLIR module");
    }
    mlir::SmallVector<void*> packedArgs;
    uint64_t numArgs = pParams->numOfInputs + pParams->numOfOutputs;
    std::vector<uint64_t> networkArgArray(numArgs);
    networkArgArray.reserve(numArgs);
    for (uint64_t i = 0; i < pParams->numOfInputs; i++) {
        auto memRefHandle = reinterpret_cast<MemRefHandle*>(pParams->pInputs[i]);
        const uint64_t allocatedPtr = reinterpret_cast<uint64_t>(memRefHandle->getAllocated());
        networkArgArray[i] = allocatedPtr;
    }
    for (uint64_t i = 0; i < pParams->numOfOutputs; i++) {
        auto memRefHandle = reinterpret_cast<MemRefHandle*>(pParams->pOutputs[i]);
        const uint64_t allocatedPtr = reinterpret_cast<uint64_t>(memRefHandle->getAllocated());
        networkArgArray[i + pParams->numOfInputs] = allocatedPtr;
    }

    mlir::ExecutionEngine::Argument<void*>::pack(packedArgs, execCtx->_executionContextHandle);
    auto networkArgArrayPtr = networkArgArray.data();
    mlir::ExecutionEngine::Argument<uint64_t*>::pack(packedArgs, networkArgArrayPtr);
    mlir::ExecutionEngine::Argument<uint64_t>::pack(packedArgs, numArgs);
    mlir::ExecutionEngine::Argument<uint64_t*>::pack(packedArgs, argIndexArray);
    mlir::ExecutionEngine::Argument<uint64_t>::pack(packedArgs, argIndexArraySize);
    mlir::ExecutionEngine::Argument<ze_command_list_handle_t*>::pack(packedArgs, pParams->commandLists);
    mlir::ExecutionEngine::Argument<uint64_t>::pack(packedArgs, pParams->numCommandLists);
    mlir::ExecutionEngine::Argument<ze_command_queue_handle_t>::pack(packedArgs, pParams->commandQueue);
    mlir::ExecutionEngine::Argument<ze_fence_handle_t>::pack(packedArgs, pParams->inferenceFence);

    auto error = _engine->invokePacked(funcName, packedArgs);
    if (error) {
        OPENVINO_THROW("Error invoking main: " + llvm::toString(std::move(error)));
    }
}

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__((visibility("default")))
#endif

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeGetAPIVersion(npu_vm_runtime_version_t* pVersion) {
    DebugTrace dt("npuVMRuntimeGetAPIVersion");
    if (pVersion == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    *pVersion = NPU_VM_RUNTIME_VERSION_CURRENT;
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeCreate(const npu_vm_runtime_blob_desc_t* desc,
                                                                            npu_vm_runtime_handle_t* phRuntime,
                                                                            npu_vm_runtime_properties_t* pProperties) {
    DebugTrace dt("npuVMRuntimeCreate");
    if (phRuntime == nullptr || desc == nullptr || pProperties == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = new NPUMLIRRuntime(desc, pProperties);
        *phRuntime = reinterpret_cast<npu_vm_runtime_handle_t>(runtime);
    } catch (const std::exception& e) {
        vpux::Logger::global().error("npuVMRuntimeCreate - Error creating MLIR runtime: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeDestroy(npu_vm_runtime_handle_t hRuntime) {
    DebugTrace dt("npuVMRuntimeDestroy");
    if (hRuntime == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        delete runtime;
    } catch (const std::exception& e) {
        vpux::Logger::global().error("npuVMRuntimeDestroy - Error destroying MLIR runtime: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeGetMetadata(
        npu_vm_runtime_handle_t hRuntime, uint32_t argIndex, ze_graph_argument_properties_3_t* pGraphArgumentProperties,
        ze_graph_argument_metadata_t* pGraphArgumentMetadata, [[maybe_unused]] int64_t* upperBound) {
    DebugTrace dt("npuVMRuntimeGetMetadata");
    if (hRuntime == nullptr || pGraphArgumentProperties == nullptr || pGraphArgumentMetadata == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        runtime->getArgumentProperties(argIndex, pGraphArgumentProperties, pGraphArgumentMetadata);
    } catch (const std::exception& e) {
        vpux::Logger::global().error("Error getting argument properties: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeExecute(npu_vm_runtime_handle_t hRuntime,
                                                                             npu_vm_runtime_execute_params_t* pParams) {
    DebugTrace dt("npuVMRuntimeExecute");
    if (hRuntime == nullptr || pParams == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        runtime->execute(pParams);
    } catch (const std::exception& e) {
        vpux::Logger::global().error("npuVMRuntimeExecute - Error executing MLIR runtime: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimePredictOutputShape(
        npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_predict_output_shape_params_t* pParams) {
    DebugTrace dt("npuVMRuntimePredictOutputShape");
    if (hRuntime == nullptr || pParams == nullptr || pParams->pInputs == nullptr || pParams->pOutputs == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        runtime->predictOutputShape(pParams);
    } catch (const std::exception& e) {
        vpux::Logger::global().error("npuVMRuntimePredictOutputShape - Error executing MLIR runtime: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeCreateMemRef(int64_t dimsCount, npu_vm_runtime_mem_ref_handle_t* phMemRef) {
    DebugTrace dt("npuVMRuntimeCreateMemRef");
    if (phMemRef == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (dimsCount <= 0 || dimsCount > ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) {
        return NPU_VM_RUNTIME_RESULT_ERROR_UNSUPPORTED_DIM_COUNT;
    }

    try {
        MemRefHandle* memRef = new MemRefHandle(dimsCount);
        *phMemRef = reinterpret_cast<npu_vm_runtime_mem_ref_handle_t>(memRef);
    } catch (const std::exception& e) {
        vpux::Logger::global().error("npuVMRuntimeCreateMemRef - Error creating MemRef: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeDestroyMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef) {
    DebugTrace dt("npuVMRuntimeDestroyMemRef");
    if (hMemRef == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    try {
        MemRefHandle* memRef = reinterpret_cast<MemRefHandle*>(hMemRef);
        delete memRef;
    } catch (const std::exception& e) {
        vpux::Logger::global().error("npuVMRuntimeDestroyMemRef - Error destroying MemRef: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeSetMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef,
                                                                               const void* basePtr, const void* data,
                                                                               int64_t offset, int64_t* pSizes,
                                                                               int64_t* pStrides, int64_t dimsCount) {
    DebugTrace dt("npuVMRuntimeSetMemRef");
    if (hMemRef == nullptr || pSizes == nullptr || pStrides == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    try {
        MemRefHandle* memRef = reinterpret_cast<MemRefHandle*>(hMemRef);
        if (dimsCount <= 0 || dimsCount != memRef->dimCount || dimsCount > ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) {
            return NPU_VM_RUNTIME_RESULT_ERROR_UNSUPPORTED_DIM_COUNT;
        }
        MemRefNDRef ref(memRef->memRefBufferPtr, memRef->dimCount);
        ref.setAllocated(basePtr);
        ref.setAligned(data);
        ref.setOffset(offset);
        ref.setSizes(pSizes, memRef->dimCount);
        ref.setStrides(pStrides, memRef->dimCount);
    } catch (const std::exception& e) {
        vpux::Logger::global().error("npuVMRuntimeSetMemRef - Error setting MemRef: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeParseMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef, const void** pBasePtr, const void** pData,
                        int64_t* pOffset, int64_t* pSizes, int64_t* pStrides, int64_t* pDimsCount) {
    DebugTrace dt("npuVMRuntimeParseMemRef");
    if (hMemRef == nullptr || pBasePtr == nullptr || pData == nullptr || pOffset == nullptr || pSizes == nullptr ||
        pStrides == nullptr || pDimsCount == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    try {
        MemRefHandle* memRef = reinterpret_cast<MemRefHandle*>(hMemRef);
        memRef->parseMemRef(pBasePtr, pData, pOffset, pSizes, pStrides, pDimsCount);
    } catch (const std::exception& e) {
        vpux::Logger::global().error("npuVMRuntimeParseMemRef - Error parsing MemRef: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeCreateExecutionContext(
        npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_execution_context_handle_t* phExecutionHandle) {
    DebugTrace dt("npuVMRuntimeCreateExecutionContext");
    if (hRuntime == nullptr || phExecutionHandle == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        runtime->createExecutionContext(phExecutionHandle);
    } catch (const std::exception& e) {
        vpux::Logger::global().error("Error creating an execution context: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeDestroyExecutionContext(npu_vm_runtime_execution_context_handle_t phExecutionHandle) {
    DebugTrace dt("npuVMRuntimeDestroyExecutionContext");
    if (phExecutionHandle == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    ExecutionContext* pContext = reinterpret_cast<ExecutionContext*>(phExecutionHandle);
    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(pContext->_runtime);
        if (runtime) {
            runtime->destroyExecutionContext(phExecutionHandle);
        }
    } catch (const std::exception& e) {
        delete pContext;
        vpux::Logger::global().error("Error destroying an execution context: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    delete pContext;
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeUpdateMutableCommandList(npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_execute_params_t* pParams,
                                     uint64_t* argIndexArray, uint64_t argIndexArraySize) {
    DebugTrace dt("npuVMRuntimeUpdateMutableCommandList");
    if (hRuntime == nullptr || pParams == nullptr || argIndexArray == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        runtime->executeMutableCommandList(pParams, argIndexArray, argIndexArraySize);
    } catch (const std::exception& e) {
        vpux::Logger::global().error("Error updating mutable command list: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimePredictOutputShape2(
        npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_predict_output_shape_params_t2* pParams) {
    DebugTrace dt("npuVMRuntimePredictOutputShape2");
    if (hRuntime == nullptr || pParams == nullptr || pParams->pInputs == nullptr || pParams->pOutputs == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        npu_vm_runtime_predict_output_shape_params_t params{};
        params.pInputs = pParams->pInputs;
        params.numOfInputs = pParams->numOfInputs;
        params.pOutputs = pParams->pOutputs;
        params.numOfOutputs = pParams->numOfOutputs;

        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        runtime->predictOutputShape(&params);
    } catch (const std::exception& e) {
        vpux::Logger::global().error("npuVMRuntimePredictOutputShape2 - Error executing MLIR runtime: {0}", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
