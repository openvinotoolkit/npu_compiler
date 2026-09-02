//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/frontend/xml_deserializer.hpp"

#include <openvino/util/shared_object.hpp>

#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

using ExtensionsMap = std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>;
using CreateExtensionsFn = void(std::vector<ov::Extension::Ptr>&);

class SharedObjectExtension final : public ov::Extension {
public:
    SharedObjectExtension(const ov::Extension::Ptr& extension, const std::shared_ptr<void>& sharedObject)
            : _extension(extension), _sharedObject(sharedObject) {
    }

    const ov::Extension::Ptr& extension() const {
        return _extension;
    }

private:
    ov::Extension::Ptr _extension;
    std::shared_ptr<void> _sharedObject;
};

std::filesystem::path resolveExtensionPath(const std::string& path) {
    try {
        const auto absolutePath = std::filesystem::absolute(std::filesystem::weakly_canonical(path));
        return std::filesystem::exists(absolutePath) ? absolutePath : std::filesystem::path(path);
    } catch (const std::exception& ex) {
        throw std::runtime_error("Path could not be resolved: " + path + ". Error: " + ex.what());
    }
}

std::shared_ptr<void> loadExtensionSharedObject(const std::filesystem::path& resolvedPath,
                                                const std::string& originalPath) {
    try {
        return ov::util::load_shared_object(resolvedPath);
    } catch (const std::exception& ex) {
        throw std::runtime_error("Failed to load extension library '" + originalPath + "': " + ex.what());
    }
}

std::vector<ov::Extension::Ptr> loadExtensionsFromSharedObject(const std::shared_ptr<void>& sharedObject) {
    if (!sharedObject) {
        throw std::runtime_error("Shared object is null");
    }
    std::vector<ov::Extension::Ptr> loadedExtensions;
    auto createExtensionsSym = ov::util::get_symbol(sharedObject, "create_extensions");
    if (createExtensionsSym == nullptr) {
        throw std::runtime_error("Extension library does not export required symbol 'create_extensions'");
    }
    reinterpret_cast<CreateExtensionsFn*>(createExtensionsSym)(loadedExtensions);
    return loadedExtensions;
}

ov::BaseOpExtension::Ptr insertBaseOpExtension(const ov::Extension::Ptr& extension, ExtensionsMap& extensionsMap) {
    if (auto opBaseExt = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension)) {
        extensionsMap.insert({opBaseExt->get_type_info(), opBaseExt});
        return opBaseExt;
    }

    return ov::BaseOpExtension::Ptr{};
}

void insertAttachedExtensions(const ov::BaseOpExtension::Ptr& baseOpExtension, ExtensionsMap& extensionsMap) {
    for (const auto& attachedExtension : baseOpExtension->get_attached_extensions()) {
        insertBaseOpExtension(attachedExtension, extensionsMap);
    }
}

void mergeLoadedExtensionsIntoMap(const std::vector<ov::Extension::Ptr>& loadedExtensions,
                                  ExtensionsMap& extensionsMap) {
    // Mirror add_extensions_unsafe traversal: add loaded extension first, then attached ones.
    for (const auto& extension : loadedExtensions) {
        if (auto baseOpExtension = insertBaseOpExtension(extension, extensionsMap)) {
            insertAttachedExtensions(baseOpExtension, extensionsMap);
        }
    }
}

}  // namespace

namespace vpux {

void loadBaseOpExtensionsForMap(const std::string& path,
                                std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensionsMap,
                                std::vector<ov::Extension::Ptr>& sharedObjectExtensions) {
    static std::mutex loadBaseOpExtensionsMutex;
    const std::lock_guard<std::mutex> lock(loadBaseOpExtensionsMutex);

    const auto resolvedPath = resolveExtensionPath(path);
    const auto sharedObject = loadExtensionSharedObject(resolvedPath, path);
    const auto loadedExtensions = loadExtensionsFromSharedObject(sharedObject);
    for (const auto& extension : loadedExtensions) {
        sharedObjectExtensions.push_back(std::make_shared<SharedObjectExtension>(extension, sharedObject));
    }
    mergeLoadedExtensionsIntoMap(loadedExtensions, extensionsMap);
}

}  // namespace vpux
