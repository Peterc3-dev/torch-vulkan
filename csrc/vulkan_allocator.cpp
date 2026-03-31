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
  fprintf(stderr, "ALLOC: nbytes=%zu\n", nbytes);
  if (nbytes == 0) {
    return c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0));
  }

  fprintf(stderr, "ALLOC: getting manager\n");
  auto& mgr = VulkanContext::instance().manager();
  fprintf(stderr, "ALLOC: got manager\n");

  size_t num_floats = (nbytes + sizeof(float) - 1) / sizeof(float);
  std::vector<float> zeros(num_floats, 0.0f);

  fprintf(stderr, "ALLOC: creating kompute tensor, num_floats=%zu\n", num_floats);
  auto tensor = mgr.tensor(zeros);
  fprintf(stderr, "ALLOC: tensor created\n");

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
