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

// Helper: run a 1D elementwise shader (3 buffers: a, b, out)
static at::Tensor elementwise_binary(
    const at::Tensor& self, const at::Tensor& other,
    const std::string& shader_name,
    std::vector<uint32_t> push_constants) {

  auto self_c = self.contiguous();
  auto other_c = other.contiguous();
  int64_t n = self_c.numel();
  TORCH_CHECK(n == other_c.numel(), shader_name, ": size mismatch");

  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();

  std::vector<float> a_data(self_c.data_ptr<float>(), self_c.data_ptr<float>() + n);
  std::vector<float> b_data(other_c.data_ptr<float>(), other_c.data_ptr<float>() + n);
  std::vector<float> c_data(n, 0.0f);

  auto tensor_a = mgr.tensor(a_data);
  auto tensor_b = mgr.tensor(b_data);
  auto tensor_c = mgr.tensor(c_data);

  auto spirv = ctx.load_shader(shader_name);
  uint32_t wg = (n + 255) / 256;

  auto algorithm = mgr.algorithm(
      {tensor_a, tensor_b, tensor_c}, spirv,
      kp::Workgroup({wg, 1, 1}), {}, push_constants);

  auto seq = mgr.sequence();
  seq->record<kp::OpTensorSyncDevice>({tensor_a, tensor_b});
  seq->record<kp::OpAlgoDispatch>(algorithm);
  seq->record<kp::OpTensorSyncLocal>({tensor_c});
  seq->eval();

  auto output = at::empty(self.sizes(), self.options());
  std::memcpy(output.data_ptr(), tensor_c->data(), n * sizeof(float));
  return output;
}

// Helper: run a 1D elementwise shader (2 buffers: in, out)
static at::Tensor elementwise_unary(
    const at::Tensor& self, const std::string& shader_name,
    std::vector<uint32_t> push_constants) {

  auto self_c = self.contiguous();
  int64_t n = self_c.numel();

  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();

  std::vector<float> in_data(self_c.data_ptr<float>(), self_c.data_ptr<float>() + n);
  std::vector<float> out_data(n, 0.0f);

  auto tensor_in = mgr.tensor(in_data);
  auto tensor_out = mgr.tensor(out_data);

  auto spirv = ctx.load_shader(shader_name);
  uint32_t wg = (n + 255) / 256;

  auto algorithm = mgr.algorithm(
      {tensor_in, tensor_out}, spirv,
      kp::Workgroup({wg, 1, 1}), {}, push_constants);

  auto seq = mgr.sequence();
  seq->record<kp::OpTensorSyncDevice>({tensor_in});
  seq->record<kp::OpAlgoDispatch>(algorithm);
  seq->record<kp::OpTensorSyncLocal>({tensor_out});
  seq->eval();

  auto output = at::empty(self.sizes(), self.options());
  std::memcpy(output.data_ptr(), tensor_out->data(), n * sizeof(float));
  return output;
}

// ---- Operators ----

// aten::mm — matrix multiplication via matmul.spv
at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2, "mm: inputs must be 2D");
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

// aten::add.Tensor — elementwise add with alpha scaling
at::Tensor add_tensor(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  uint32_t n = static_cast<uint32_t>(self.numel());
  float alpha_f = alpha.toFloat();
  uint32_t alpha_bits;
  std::memcpy(&alpha_bits, &alpha_f, sizeof(uint32_t));
  return elementwise_binary(self, other, "add", {n, alpha_bits});
}

// aten::mul.Tensor — elementwise multiply
at::Tensor mul_tensor(const at::Tensor& self, const at::Tensor& other) {
  uint32_t n = static_cast<uint32_t>(self.numel());
  return elementwise_binary(self, other, "mul", {n});
}

// aten::relu — rectified linear
at::Tensor relu(const at::Tensor& self) {
  uint32_t n = static_cast<uint32_t>(self.numel());
  return elementwise_unary(self, "relu", {n});
}

// _to_copy — .to(device) implementation
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
    auto self_cpu = self.device().type() == c10::DeviceType::CPU
        ? self.contiguous() : self.to(at::kCPU).contiguous();
    auto self_cast = self_cpu.to(target_dtype).contiguous();
    auto result = at::empty(self_cast.sizes(),
        self_cast.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
    std::memcpy(result.data_ptr(), self_cast.data_ptr(),
        self_cast.numel() * self_cast.element_size());
    return result;
  } else if (target_device.type() == c10::DeviceType::CPU) {
    auto result = at::empty(self.sizes(),
        at::TensorOptions().dtype(target_dtype).device(at::kCPU));
    std::memcpy(result.data_ptr(), self.data_ptr(),
        self.numel() * self.element_size());
    return result;
  }
  TORCH_CHECK(false, "_to_copy: unsupported target device");
}

// copy_ — in-place copy
at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  auto src_c = src.contiguous();
  size_t bytes = src_c.numel() * src_c.element_size();
  std::memcpy(self.data_ptr(), src_c.data_ptr(), bytes);
  return self;
}

// empty.memory_format
at::Tensor empty_memory_format(
    at::IntArrayRef size, std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout, std::optional<at::Device> device,
    std::optional<bool> pin_memory, std::optional<at::MemoryFormat> memory_format) {
  auto actual_dtype = dtype.value_or(at::kFloat);
  auto allocator = &VulkanAllocator::instance();
  int64_t nel = 1;
  for (auto s : size) nel *= s;
  size_t nbytes = nel * at::elementSize(actual_dtype);

  auto storage = c10::Storage(c10::Storage::use_byte_size_t(), nbytes,
      allocator->allocate(nbytes), allocator, true);
  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      at::scalarTypeToTypeMeta(actual_dtype));
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  return tensor;
}

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mm", &mm);
  m.impl("add.Tensor", &add_tensor);
  m.impl("mul.Tensor", &mul_tensor);
  m.impl("relu", &relu);
  m.impl("empty.memory_format", &empty_memory_format);
  m.impl("_to_copy", &vulkan_to_copy);
  m.impl("copy_", &vulkan_copy_);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace ops
} // namespace torch_vulkan
