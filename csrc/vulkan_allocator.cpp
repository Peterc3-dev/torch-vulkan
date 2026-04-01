#include "vulkan_allocator.h"
#include "vulkan_context.h"
#include <c10/core/Device.h>
#include <cstring>
#include <cstdio>

namespace torch_vulkan {

VulkanAllocator& VulkanAllocator::instance() {
  static VulkanAllocator alloc;
  return alloc;
}

c10::DataPtr VulkanAllocator::allocate(size_t nbytes) {
  if (nbytes == 0) {
    return c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0));
  }

  size_t num_floats = (nbytes + sizeof(float) - 1) / sizeof(float);

  std::shared_ptr<kp::TensorT<float>> tensor;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Phase 4: Check free list for a reusable buffer of the same size.
    auto it = free_list_.find(num_floats);
    if (it != free_list_.end() && !it->second.empty()) {
      tensor = std::move(it->second.back());
      it->second.pop_back();
      // Zero out the reused buffer for safety
      std::memset(tensor->data(), 0, num_floats * sizeof(float));
    }
  }

  if (!tensor) {
    // No reusable buffer found — create a new Kompute tensor.
    auto& mgr = VulkanContext::instance().manager();
    std::vector<float> zeros(num_floats, 0.0f);
    tensor = mgr.tensor(zeros);
  }

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

  auto it = alloc.tensor_map_.find(ptr);
  if (it == alloc.tensor_map_.end()) return;

  // Phase 4: Instead of destroying the Kompute tensor, move it to the free
  // list keyed by size. The next allocation of the same size will reuse this
  // buffer, getting the same data pointer, which means the algorithm cache
  // in VulkanContext will match and reuse the compiled VkPipeline.
  auto tensor = std::move(it->second);
  size_t num_floats = tensor->size();
  alloc.free_list_[num_floats].push_back(std::move(tensor));
  alloc.tensor_map_.erase(it);
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
