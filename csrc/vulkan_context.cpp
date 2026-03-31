#include "vulkan_context.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace torch_vulkan {

VulkanContext& VulkanContext::instance() {
  static VulkanContext ctx;
  return ctx;
}

VulkanContext::VulkanContext() {
  // Kompute manages Vulkan instance + device selection automatically.
  // Device 0 = first Vulkan-capable GPU found.
  mgr_ = std::make_unique<kp::Manager>(0);

  // Default shader directory: alongside the shared library
  shader_dir_ = "";
}

kp::Manager& VulkanContext::manager() {
  return *mgr_;
}

void VulkanContext::set_shader_dir(const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  shader_dir_ = path;
  if (!shader_dir_.empty() && shader_dir_.back() != '/') {
    shader_dir_ += '/';
  }
}

std::vector<uint32_t> VulkanContext::load_shader(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = shader_cache_.find(name);
  if (it != shader_cache_.end()) {
    return it->second;
  }

  std::string path = shader_dir_ + name + ".spv";
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to load shader: " + path);
  }

  size_t size = file.tellg();
  file.seekg(0);

  std::vector<uint32_t> spirv(size / sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(spirv.data()), size);

  shader_cache_[name] = spirv;
  return spirv;
}

std::string VulkanContext::device_name() const {
  // Kompute exposes device properties through the manager
  return "Vulkan Device 0";
}

} // namespace torch_vulkan
