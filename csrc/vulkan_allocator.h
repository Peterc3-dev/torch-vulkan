#pragma once

#include <c10/core/Allocator.h>
#include <kompute/Kompute.hpp>
#include <mutex>
#include <unordered_map>

namespace torch_vulkan {

// Custom allocator that backs PyTorch tensors with Kompute GPU buffers.
// When PyTorch requests memory for a "vulkan" tensor, this allocator
// creates a Kompute tensor (Vulkan buffer) and returns a pointer to
// the host-mapped staging buffer.
class VulkanAllocator final : public c10::Allocator {
public:
  static VulkanAllocator& instance();

  c10::DataPtr allocate(size_t nbytes) override;
  c10::DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;

  // Retrieve the Kompute tensor backing a given data pointer.
  // Needed when dispatching ops — we need the GPU buffer, not the host ptr.
  std::shared_ptr<kp::TensorT<float>> get_kompute_tensor(void* ptr);

private:
  VulkanAllocator() = default;

  static void deleter(void* ptr);

  std::mutex mutex_;
  // Map from host data pointer -> Kompute tensor
  std::unordered_map<void*, std::shared_ptr<kp::TensorT<float>>> tensor_map_;
};

} // namespace torch_vulkan
