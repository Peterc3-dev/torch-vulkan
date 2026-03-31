#pragma once

#include <cstring>
#include <c10/core/Allocator.h>
#include <kompute/Kompute.hpp>
#include <mutex>
#include <unordered_map>

namespace torch_vulkan {

class VulkanAllocator final : public c10::Allocator {
public:
  static VulkanAllocator& instance();

  c10::DataPtr allocate(size_t nbytes) override;
  c10::DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;

  std::shared_ptr<kp::TensorT<float>> get_kompute_tensor(void* ptr);

private:
  VulkanAllocator() = default;

  static void deleter(void* ptr);

  std::mutex mutex_;
  std::unordered_map<void*, std::shared_ptr<kp::TensorT<float>>> tensor_map_;
};

} // namespace torch_vulkan
