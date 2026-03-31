#include "vulkan_allocator.h"
#include "vulkan_context.h"
#include <c10/core/Device.h>
#include <cstring>

namespace torch_vulkan {

VulkanAllocator& VulkanAllocator::instance() {
  static VulkanAllocator alloc;
  return alloc;
}

c10::DataPtr VulkanAllocator::allocate(size_t nbytes) {
  if (nbytes == 0) {
    return c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0));
  }

  auto& mgr = VulkanContext::instance().manager();

  // Allocate as float buffer (round up)
  size_t num_floats = (nbytes + sizeof(float) - 1) / sizeof(float);
  std::vector<float> zeros(num_floats, 0.0f);

  auto tensor = mgr.tensor(zeros);

  // The data pointer for PyTorch is the host-side staging buffer.
  // When we dispatch ops, we sync to GPU, run shader, sync back.
  void* ptr = tensor->data();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor_map_[ptr] = tensor;
  }

  return c10::DataPtr(
      ptr,
      ptr,
      &VulkanAllocator::deleter,
      c10::Device(c10::DeviceType::PrivateUse1, 0));
}

c10::DeleterFnPtr VulkanAllocator::raw_deleter() const {
  return &VulkanAllocator::deleter;
}

void VulkanAllocator::deleter(void* ptr) {
  if (!ptr) return;
  auto& alloc = VulkanAllocator::instance();
  std::lock_guard<std::mutex> lock(alloc.mutex_);
  alloc.tensor_map_.erase(ptr);
  // Kompute tensor destructor handles Vulkan buffer cleanup
}

void VulkanAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  std::memcpy(dest, src, count);
}

std::shared_ptr<kp::TensorT<float>> VulkanAllocator::get_kompute_tensor(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = tensor_map_.find(ptr);
  if (it == tensor_map_.end()) {
    throw std::runtime_error("No Kompute tensor found for data pointer");
  }
  return it->second;
}

} // namespace torch_vulkan
