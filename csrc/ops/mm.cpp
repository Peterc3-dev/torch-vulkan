#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include "../vulkan_allocator.h"
#include "../vulkan_context.h"
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>
#include <cstring>

namespace torch_vulkan {
namespace ops {

// aten::mm — matrix multiplication via matmul.spv
at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.dim() == 2, "mm: self must be 2D");
  TORCH_CHECK(mat2.dim() == 2, "mm: mat2 must be 2D");
  TORCH_CHECK(self.size(1) == mat2.size(0), "mm: dimension mismatch");
  TORCH_CHECK(self.dtype() == at::kFloat, "mm: only float32 for now");

  int64_t M = self.size(0), K = self.size(1), N = mat2.size(1);
  int64_t dim = std::max({M, K, N});

  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();
  auto self_c = self.contiguous();
  auto mat2_c = mat2.contiguous();

  std::vector<float> a_data(dim * dim, 0.0f);
  std::vector<float> b_data(dim * dim, 0.0f);
  std::vector<float> c_data(dim * dim, 0.0f);

  const float* a_ptr = self_c.data_ptr<float>();
  const float* b_ptr = mat2_c.data_ptr<float>();
  for (int64_t i = 0; i < M; i++)
    for (int64_t j = 0; j < K; j++)
      a_data[i * dim + j] = a_ptr[i * K + j];
  for (int64_t i = 0; i < K; i++)
    for (int64_t j = 0; j < N; j++)
      b_data[i * dim + j] = b_ptr[i * N + j];

  auto tensor_a = mgr.tensor(a_data);
  auto tensor_b = mgr.tensor(b_data);
  auto tensor_c = mgr.tensor(c_data);

  auto spirv = ctx.load_shader("matmul");
  uint32_t n_val = static_cast<uint32_t>(dim);
  uint32_t wg_x = (dim + 15) / 16, wg_y = (dim + 15) / 16;

  auto algorithm = mgr.algorithm(
      {tensor_a, tensor_b, tensor_c}, spirv,
      kp::Workgroup({wg_x, wg_y, 1}), {},
      std::vector<uint32_t>{n_val});

  auto seq = mgr.sequence();
  seq->record<kp::OpTensorSyncDevice>({tensor_a, tensor_b});
  seq->record<kp::OpAlgoDispatch>(algorithm);
  seq->record<kp::OpTensorSyncLocal>({tensor_c});
  seq->eval();

  auto output = at::empty({M, N}, self.options());
  float* out_ptr = output.data_ptr<float>();
  const float* c_ptr = tensor_c->data();
  for (int64_t i = 0; i < M; i++)
    for (int64_t j = 0; j < N; j++)
      out_ptr[i * N + j] = c_ptr[i * dim + j];

  return output;
}

// _to_copy — this is what .to(device) actually calls
at::Tensor vulkan_to_copy(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<at::MemoryFormat> memory_format) {

  auto target_device = device.value_or(self.device());
  auto target_dtype = dtype.value_or(self.scalar_type());

  if (target_device.type() == c10::DeviceType::PrivateUse1) {
    // Something -> Vulkan
    auto self_cpu = self.device().type() == c10::DeviceType::CPU
        ? self.contiguous()
        : self.to(at::kCPU).contiguous();
    auto self_float = self_cpu.to(target_dtype).contiguous();

    auto result = at::empty(self_float.sizes(),
        self_float.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
    std::memcpy(result.data_ptr(), self_float.data_ptr(),
        self_float.numel() * self_float.element_size());
    return result;

  } else if (target_device.type() == c10::DeviceType::CPU) {
    // Vulkan -> CPU
    auto result = at::empty(self.sizes(),
        at::TensorOptions().dtype(target_dtype).device(at::kCPU));
    std::memcpy(result.data_ptr(), self.data_ptr(),
        self.numel() * self.element_size());
    return result;

  } else {
    TORCH_CHECK(false, "_to_copy: unsupported target device ", target_device);
  }
}

// copy_ — in-place copy between devices
at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  auto src_c = src.contiguous();
  TORCH_CHECK(self.numel() == src_c.numel(), "copy_: size mismatch");

  if (self.device().type() == c10::DeviceType::PrivateUse1 &&
      src_c.device().type() == c10::DeviceType::CPU) {
    std::memcpy(self.data_ptr(), src_c.data_ptr(),
        src_c.numel() * src_c.element_size());
  } else if (self.device().type() == c10::DeviceType::CPU &&
             src_c.device().type() == c10::DeviceType::PrivateUse1) {
    std::memcpy(self.data_ptr(), src_c.data_ptr(),
        src_c.numel() * src_c.element_size());
  } else if (self.device().type() == c10::DeviceType::PrivateUse1 &&
             src_c.device().type() == c10::DeviceType::PrivateUse1) {
    std::memcpy(self.data_ptr(), src_c.data_ptr(),
        src_c.numel() * src_c.element_size());
  } else {
    TORCH_CHECK(false, "copy_: unsupported device combination");
  }
  return self;
}

// aten::empty.memory_format
at::Tensor empty_memory_format(
    at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> memory_format) {
  auto actual_dtype = dtype.value_or(at::kFloat);
  auto allocator = &VulkanAllocator::instance();

  int64_t nelements = 1;
  for (auto s : size) nelements *= s;
  size_t nbytes = nelements * at::elementSize(actual_dtype);

  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(), nbytes,
      allocator->allocate(nbytes), allocator, true);

  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      at::scalarTypeToTypeMeta(actual_dtype));

  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  return tensor;
}

// CPU fallback — but NOT for copy/transfer ops
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mm", &mm);
  m.impl("empty.memory_format", &empty_memory_format);
  m.impl("_to_copy", &vulkan_to_copy);
  m.impl("copy_", &vulkan_copy_);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace ops
} // namespace torch_vulkan
