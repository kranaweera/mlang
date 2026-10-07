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
        assert!(code.contains("ml_alloc_m2_2_m2_2"));
        assert!(code.contains("atexit(ml_release_m2_2_m2_2)"));
        assert!(!code.contains("ML_CUDA_CHECK(cudaFree"));
        assert!(code.contains("<<<"));
    }
    #[test]
    fn generates_fused_relu_matmul() {
        let source="fn main() { let a=[[-1.0,2.0],[3.0,-4.0]]; let b=[[2.0,0.0],[0.0,2.0]]; print(relu(a @ b)); }";
        let ast=parser::Parser::new(lexer::lex(source).unwrap()).program().unwrap();
        let code=emit_cuda(&ir::optimize(ir::lower(&ast).unwrap()));
        assert!(code.contains("ml_matmul_relu_m2_2_m2_2(ml_a, ml_b)"));
        assert!(code.contains("int use_relu"));
    }
}
