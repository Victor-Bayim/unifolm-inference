#include <torch/extension.h>

void launch_gelu_and_mul(torch::Tensor &out, torch::Tensor &input);

void gelu_and_mul(torch::Tensor out, torch::Tensor input)
{
    launch_gelu_and_mul(out, input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gelu_and_mul", &gelu_and_mul, "GELU and Mul activation function");
}