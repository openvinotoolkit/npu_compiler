//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <vector>
// clang-format off
#include <cstdint>
// clang-format on
#include <memory>
#include <map>
#include <set>

#include <ie_blob.h>
#include <ie_common.h>
#include <ie_remote_context.hpp>
#include <ie_icnn_network.hpp>

#include "vpux/utils/IE/config.hpp"

#include "vpux_compiler.hpp"

namespace vpux {

bool isBlobAllocatedByAllocator(const InferenceEngine::Blob::Ptr& blob,
                                const std::shared_ptr<InferenceEngine::IAllocator>& allocator);

std::string getLibFilePath(const std::string& baseName);

//------------------------------------------------------------------------------
class IDevice;
class Device;

class IEngineBackend : public std::enable_shared_from_this<IEngineBackend> {
public:
    /** @brief Get device, which can be used for inference. Backend responsible for selection. */
    virtual const std::shared_ptr<IDevice> getDevice() const;
    /** @brief Search for a specific device by name */
    virtual const std::shared_ptr<IDevice> getDevice(const std::string& specificDeviceName) const;
    /** @brief Get device, which is configured/suitable for provided params */
    virtual const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& paramMap) const;
    /** @brief Provide a list of names of all devices, with which user can work directly */
    virtual const std::vector<std::string> getDeviceNames() const;
    /** @brief Get name of backend */
    virtual const std::string getName() const = 0;
    /** @brief Register backend-specific options */
    virtual void registerOptions(OptionsDesc& options) const;

#ifndef OPENVINO_STATIC_LIBRARY
protected:
#endif
    ~IEngineBackend() = default;
};

class EngineBackend final {
public:
    EngineBackend() = default;

#ifdef OPENVINO_STATIC_LIBRARY
    EngineBackend(std::shared_ptr<IEngineBackend> impl);
#endif

#ifndef OPENVINO_STATIC_LIBRARY
    EngineBackend(const std::string& pathToLib);
#endif

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~EngineBackend() {
        _impl = {};
    }

    const std::shared_ptr<Device> getDevice() const;
    const std::shared_ptr<Device> getDevice(const std::string& specificDeviceName) const;
    const std::shared_ptr<Device> getDevice(const InferenceEngine::ParamMap& paramMap) const;
    const std::vector<std::string> getDeviceNames() const {
        return _impl->getDeviceNames();
    }
    const std::string getName() const {
        return _impl->getName();
    }
    void registerOptions(OptionsDesc& options) const {
        _impl->registerOptions(options);
        options.addSharedObject(_so);
    }

private:
    std::shared_ptr<IEngineBackend> _impl;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::shared_ptr<void> _so;
};

//------------------------------------------------------------------------------
class Allocator : public InferenceEngine::IAllocator {
public:
    using Ptr = std::shared_ptr<Allocator>;
    using CPtr = std::shared_ptr<const Allocator>;

    /** @brief Wrap remote memory. Backend should get all required data from paramMap */
    virtual void* wrapRemoteMemory(const InferenceEngine::ParamMap& paramMap) noexcept;
    // TODO remove these methods
    // [Track number: E#23679]
    // TODO: need update methods to remove Kmb from parameters
    /** @deprecated These functions below should not be used */
    virtual void* wrapRemoteMemoryHandle(const int& remoteMemoryFd, const size_t size, void* memHandle) noexcept = 0;
    virtual void* wrapRemoteMemoryOffset(const int& remoteMemoryFd, const size_t size,
                                         const size_t& memOffset) noexcept = 0;

    // FIXME: temporary exposed to allow executor to use vpux::Allocator
    virtual unsigned long getPhysicalAddress(void* handle) noexcept = 0;
};

//------------------------------------------------------------------------------
class AllocatorWrapper : public Allocator {
public:
    AllocatorWrapper(const std::shared_ptr<Allocator>& impl, const std::shared_ptr<void>& so): _impl(impl), _so(so) {
    }

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~AllocatorWrapper() {
        _impl = {};
    }

    virtual void* lock(void* handle, InferenceEngine::LockOp op = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        return _impl->lock(handle, op);
    }
    virtual void unlock(void* handle) noexcept override {
        return _impl->unlock(handle);
    }
    virtual void* alloc(size_t size) noexcept override {
        return _impl->alloc(size);
    }
    virtual bool free(void* handle) noexcept override {
        return _impl->free(handle);
    }

    virtual void* wrapRemoteMemory(const InferenceEngine::ParamMap& paramMap) noexcept override {
        return _impl->wrapRemoteMemory(paramMap);
    }
    virtual void* wrapRemoteMemoryHandle(const int& remoteMemoryFd, const size_t size,
                                         void* memHandle) noexcept override {
        return _impl->wrapRemoteMemoryHandle(remoteMemoryFd, size, memHandle);
    }
    virtual void* wrapRemoteMemoryOffset(const int& remoteMemoryFd, const size_t size,
                                         const size_t& memOffset) noexcept override {
        return _impl->wrapRemoteMemoryOffset(remoteMemoryFd, size, memOffset);
    }
    virtual unsigned long getPhysicalAddress(void* handle) noexcept override {
        return _impl->getPhysicalAddress(handle);
    }

private:
    std::shared_ptr<Allocator> _impl;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::shared_ptr<void> _so;
};

//------------------------------------------------------------------------------
class Executor;

class IDevice : public std::enable_shared_from_this<IDevice> {
public:
    virtual std::shared_ptr<Allocator> getAllocator() const = 0;
    /** @brief Get allocator, which is configured/suitable for provided params
     * @example Each backend may have many allocators, each of which suitable for different RemoteMemory param */
    virtual std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& paramMap) const;

    virtual std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& networkDescription,
                                                     const Config& config) = 0;

    virtual std::string getName() const = 0;

protected:
    ~IDevice() = default;
};

class Device final {
public:
    using Ptr = std::shared_ptr<Device>;
    using CPtr = std::shared_ptr<const Device>;

    Device(const std::shared_ptr<IDevice>& device, const std::shared_ptr<void>& so): _impl(device), _so(so) {
        if (_impl->getAllocator()) {
            _allocatorWrapper = std::make_shared<AllocatorWrapper>(_impl->getAllocator(), _so);
        }
    }

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~Device() {
        _impl = {};
        _allocatorWrapper = {};
    }

    std::shared_ptr<Allocator> getAllocator() const {
        return _allocatorWrapper;
    }
    std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& paramMap) {
        return std::make_shared<AllocatorWrapper>(_impl->getAllocator(paramMap), _so);
    }

    std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& networkDescription, const Config& config) {
        return _impl->createExecutor(networkDescription, config);
    }

    std::string getName() const {
        return _impl->getName();
    }

private:
    std::shared_ptr<IDevice> _impl;
    std::shared_ptr<AllocatorWrapper> _allocatorWrapper;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::shared_ptr<void> _so;
};

//------------------------------------------------------------------------------
using PreprocMap = std::map<std::string, const InferenceEngine::PreProcessInfo>;
class Executor {
public:
    using Ptr = std::shared_ptr<Executor>;
    using CPtr = std::shared_ptr<const Executor>;

    virtual void setup(const InferenceEngine::ParamMap& params) = 0;
    virtual Executor::Ptr clone() const {
        IE_THROW() << "Not implemented";
    }

    virtual void push(const InferenceEngine::BlobMap& inputs) = 0;
    virtual void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) = 0;

    virtual void pull(InferenceEngine::BlobMap& outputs) = 0;

    virtual bool isPreProcessingSupported(const PreprocMap& preProcMap) const = 0;
    virtual std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() = 0;
    virtual InferenceEngine::Parameter getParameter(const std::string& paramName) const = 0;

    virtual ~Executor() = default;
};

}  // namespace vpux
