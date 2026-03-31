#pragma once

#include <kompute/Kompute.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch_vulkan {

// Singleton managing the Vulkan instance, device, and shader cache.
// Wraps Kompute to hide Vulkan boilerplate.
class VulkanContext {
public:
  static VulkanContext& instance();

  kp::Manager& manager();

  // Load a pre-compiled SPIR-V shader from the shaders/ directory.
  // Returns the raw SPIR-V bytecode. Caches after first load.
  std::vector<uint32_t> load_shader(const std::string& name);

  // Set the directory where .spv files live.
  void set_shader_dir(const std::string& path);

  std::string device_name() const;

private:
  VulkanContext();
  ~VulkanContext() = default;
  VulkanContext(const VulkanContext&) = delete;
  VulkanContext& operator=(const VulkanContext&) = delete;

  std::unique_ptr<kp::Manager> mgr_;
  std::string shader_dir_;
  std::unordered_map<std::string, std::vector<uint32_t>> shader_cache_;
  std::mutex mutex_;
};

} // namespace torch_vulkan
