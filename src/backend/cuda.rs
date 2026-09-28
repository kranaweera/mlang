//! Experimental CUDA source generator: GPU matrix multiplication via nvcc/CUDA runtime.
//! Other operations currently execute on CPU. CUDA is not available on macOS.
use crate::ir::Module;

pub fn emit_cuda(module: &Module) -> String {
    super::cpu::emit_cuda_c(module)
}

pub fn status() -> &'static str {
    "CUDA requires nvcc, an NVIDIA GPU, and CUDA runtime on Linux/Windows; macOS CUDA execution is unsupported"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir,lexer,parser};
    #[test]
    fn generates_gpu_matmul_kernel() {
        let source = include_str!("../../examples/matmul.ml");
        let ast=parser::Parser::new(lexer::lex(source).unwrap()).program().unwrap();
        let code=emit_cuda(&ir::optimize(ir::lower(&ast).unwrap()));
        assert!(code.contains("__global__ void ml_kernel_m2_2_m2_2"));
        assert!(code.contains("cudaMemcpyDeviceToHost"));
        assert!(code.contains("<<<"));
    }
}
