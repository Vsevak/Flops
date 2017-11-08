/* arch_Aarch64.h
 *
 * Author           : Vsevolod Nikolskii
 * Date Created     : 10/04/2017
 * Last Modified    : 10/04/2017
 *
 */

//  128-bit
#include "../bench_128/bench_f64v1_aarch64.h"
#include "../bench_128/bench_f32v2_aarch64.h"


namespace Flops{

    void run_benchmark(largeint_t iterations, size_t threads){
        cout << "Running Benchmarks for ARMv8..." << endl;
        cout << endl;

        //bench_add_mul_compiler_chains12().run(iterations, threads);
        //  128-bit
        // bench_add_f32v2_aarch64_chains4().run(iterations, threads);
        // bench_add_f32v2_aarch64_chains8().run(iterations, threads);
        // bench_mul_f32v2_aarch64_chains8().run(iterations, threads);
        // bench_mul_f32v2_aarch64_chains12().run(iterations, threads);
        // bench_mac_f32v2_aarch64_chains12().run(iterations, threads);
        // bench_fma_linear_f32v2_aarch64_chains12().run(iterations, threads);
        //
        // bench_add_f64v1_aarch64_chains4().run(iterations, threads);
        bench_add_f64v1_aarch64_chains8().run(iterations, threads);
        // bench_mul_f64v1_aarch64_chains8().run(iterations, threads);
        bench_mul_f64v1_aarch64_chains12().run(iterations, threads);
        bench_add_mul_f64v1_aarch64_chains12().run(iterations, threads);
        bench_mac_f64v0_aarch64_chains12().run(iterations, threads);
        bench_mac_f64v1_aarch64_chains12().run(iterations, threads);
        bench_fma_linear_f64v1_aarch64_chains12().run(iterations, threads);
        bench_fma_add_f64v1_aarch64_chains12().run(iterations, threads);
    }

}
