/* bench_f32v2_aarch64.h
 *
 * Author           : Vsevolod Nikolskii
 * Date Created     : 10/04/2017
 * Last Modified    : 11/04/2017
 *
 *      f32     =   Single Precision
 *      v2      =   Vectorize by 4
 *      aarch   =   ARMv8 Aarch64 instruction set
 *
 */

 #ifndef _flops_bench_f32v2_aarch64_H
 #define _flops_bench_f32v2_aarch64_H
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 #include <arm_neon.h>
 #include <arm_acle.h>
 #include "../globals.h"
 #include "../macros/macro_reduce.h"
 #include "../macros/macro_add.h"
 #include "../macros/macro_mul.h"
 #include "../macros/macro_mac.h"
 namespace Flops{
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 //  Add
 class bench_add_f32v2_aarch64_chains4 : public benchmark{
     virtual void print_meta() const{
         cout << "Single-Precision - 128-bit Aarch64 - Add/Sub:" << endl;
         cout << "    Dependency Chains  = 4" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         const float32x4_t add0 = {(float)TEST_ADD_ADD, (float)TEST_ADD_ADD,\
              (float)TEST_ADD_ADD, (float)TEST_ADD_ADD};
         const float32x4_t sub0 = {(float)TEST_ADD_SUB, (float)TEST_ADD_SUB,\
              (float)TEST_ADD_SUB, (float)TEST_ADD_SUB};

         float32x4_t r0 = {1.0f, 1.0f, 1.0f, 1.0f};
         float32x4_t r1 = {1.1f, 1.1f, 1.1f, 1.1f};
         float32x4_t r2 = {1.2f, 1.2f, 1.2f, 1.2f};
         float32x4_t r3 = {1.3f, 1.3f, 1.3f, 1.3f};
         for (size_t i = 0; i < iterations; i++){
             flops_add_chains4_unroll2_ops16(
                 vaddq_f32, vsubq_f32,
                 add0, sub0,
                 r0, r1, r2, r3
             );
         }
         flops_reduce_chains4(
             vaddq_f32,
             r0, r1, r2, r3
         );
         result = vaddvq_f32(r0);

         //  (4 ops / vector) * (16 ops / macro)
         return iterations * 4 * 16;
     }
 };
 class bench_add_f32v2_aarch64_chains8 : public benchmark{
     virtual void print_meta() const{
         cout << "Single-Precision - 128-bit Aarch64 - Add/Sub:" << endl;
         cout << "    Dependency Chains  = 8" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         const float32x4_t add0 = {(float)TEST_ADD_ADD, (float)TEST_ADD_ADD,\
              (float)TEST_ADD_ADD, (float)TEST_ADD_ADD};
         const float32x4_t sub0 = {(float)TEST_ADD_SUB, (float)TEST_ADD_SUB,\
              (float)TEST_ADD_SUB, (float)TEST_ADD_SUB};

         float32x4_t r0 = {1.0f, 1.0f, 1.0f, 1.0f};
         float32x4_t r1 = {1.1f, 1.1f, 1.1f, 1.1f};
         float32x4_t r2 = {1.2f, 1.2f, 1.2f, 1.2f};
         float32x4_t r3 = {1.3f, 1.3f, 1.3f, 1.3f};
         float32x4_t r4 = {1.4f, 1.4f, 1.4f, 1.4f};
         float32x4_t r5 = {1.5f, 1.5f, 1.5f, 1.5f};
         float32x4_t r6 = {1.6f, 1.6f, 1.6f, 1.6f};
         float32x4_t r7 = {1.7f, 1.7f, 1.7f, 1.7f};
         for (size_t i = 0; i < iterations; i++){
             flops_add_chains8_unroll2_ops32(
                 vaddq_f32, vsubq_f32,
                 add0, sub0,
                 r0, r1, r2, r3, r4, r5, r6, r7
             );
         }
         flops_reduce_chains8(
             vaddq_f32,
             r0, r1, r2, r3, r4, r5, r6, r7
         );
         result = vaddvq_f32(r0);

         //  (4 ops / vector) * (32 ops / macro)
         return iterations * 4 * 32;
     }
 };
 ////////////////////////////////////////////////////////////////////////////////
 //  Multiply
 class bench_mul_f32v2_aarch64_chains8 : public benchmark{
     virtual void print_meta() const{
         cout << "Single-Precision - 128-bit Aarch64 - Multiply:" << endl;
         cout << "    Dependency Chains = 8" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         const float32x4_t mul0 = {(float)TEST_MUL_MUL, (float)TEST_MUL_MUL,\
              (float)TEST_MUL_MUL, (float)TEST_MUL_MUL};
         const float32x4_t mul1 = {(float)TEST_MUL_DIV, (float)TEST_MUL_DIV,\
              (float)TEST_MUL_DIV, (float)TEST_MUL_DIV};

         float32x4_t r0 = {1.0f, 1.0f, 1.0f, 1.0f};
         float32x4_t r1 = {1.1f, 1.1f, 1.1f, 1.1f};
         float32x4_t r2 = {1.2f, 1.2f, 1.2f, 1.2f};
         float32x4_t r3 = {1.3f, 1.3f, 1.3f, 1.3f};
         float32x4_t r4 = {1.4f, 1.4f, 1.4f, 1.4f};
         float32x4_t r5 = {1.5f, 1.5f, 1.5f, 1.5f};
         float32x4_t r6 = {1.6f, 1.6f, 1.6f, 1.6f};
         float32x4_t r7 = {1.7f, 1.7f, 1.7f, 1.7f};
         for (size_t i = 0; i < iterations; i++){
             flops_mul_chains8_unroll2_ops32(
                 vmulq_f32,
                 mul0, mul1,
                 r0, r1, r2, r3, r4, r5, r6, r7
             );
         }
         flops_reduce_chains8(
             vaddq_f32,
             r0, r1, r2, r3, r4, r5, r6, r7
         );
         result = vaddvq_f32(r0);

         //  (4 ops / vector) * (32 ops / macro)
         return iterations * 4 * 32;
     }
 };
 class bench_mul_f32v2_aarch64_chains12 : public benchmark{
     virtual void print_meta() const{
         cout << "Single-Precision - 128-bit Aarch64 - Multiply:" << endl;
         cout << "    Dependency Chains = 12" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         const float32x4_t mul0 = {(float)TEST_MUL_MUL, (float)TEST_MUL_MUL,\
              (float)TEST_MUL_MUL, (float)TEST_MUL_MUL};
         const float32x4_t mul1 = {(float)TEST_MUL_DIV, (float)TEST_MUL_DIV,\
              (float)TEST_MUL_DIV, (float)TEST_MUL_DIV};

         float32x4_t r0 = {1.0f, 1.0f, 1.0f, 1.0f};
         float32x4_t r1 = {1.1f, 1.1f, 1.1f, 1.1f};
         float32x4_t r2 = {1.2f, 1.2f, 1.2f, 1.2f};
         float32x4_t r3 = {1.3f, 1.3f, 1.3f, 1.3f};
         float32x4_t r4 = {1.4f, 1.4f, 1.4f, 1.4f};
         float32x4_t r5 = {1.5f, 1.5f, 1.5f, 1.5f};
         float32x4_t r6 = {1.6f, 1.6f, 1.6f, 1.6f};
         float32x4_t r7 = {1.7f, 1.7f, 1.7f, 1.7f};
         float32x4_t r8 = {1.8f, 1.8f, 1.8f, 1.8f};
         float32x4_t r9 = {1.9f, 1.9f, 1.9f, 1.9f};
         float32x4_t rA = {2.0f, 2.0f, 2.0f, 2.0f};
         float32x4_t rB = {2.1f, 2.1f, 2.1f, 2.1f};
         for (size_t i = 0; i < iterations; i++){
             flops_mul_chains12_unroll2_ops48(
                 vmulq_f32,
                 mul0, mul1,
                 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
             );
         }
         flops_reduce_chains12(
             vaddq_f32,
             r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
         );
         result = vaddvq_f32(r0);

         //  (4 ops / vector) * (48 ops / macro)
         return iterations * 4 * 48;
     }
 };
////////////////////////////////////////////////////////////////////////////////
//  Multiply + Add
class bench_mac_f32v2_aarch64_chains12 : public benchmark{
    virtual void print_meta() const{
        cout << "Single-Precision - 128-bit Aarch64 - Multiply + Add:" << endl;
        cout << "    Dependency Chains = 12" << endl;
    }
    virtual largeint_t run_loop(largeint_t iterations, double &result) const{
        const float32x4_t add0 = {(float)TEST_ADD_ADD, (float)TEST_ADD_ADD,\
             (float)TEST_ADD_ADD, (float)TEST_ADD_ADD};
        const float32x4_t sub0 = {(float)TEST_ADD_SUB, (float)TEST_ADD_SUB,\
             (float)TEST_ADD_SUB, (float)TEST_ADD_SUB};
        const float32x4_t mul0 = {(float)TEST_MUL_MUL, (float)TEST_MUL_MUL,\
             (float)TEST_MUL_MUL, (float)TEST_MUL_MUL};
        const float32x4_t mul1 = {(float)TEST_MUL_DIV, (float)TEST_MUL_DIV,\
             (float)TEST_MUL_DIV, (float)TEST_MUL_DIV};

        float32x4_t r0 = {1.0f, 1.0f, 1.0f, 1.0f};
        float32x4_t r1 = {1.1f, 1.1f, 1.1f, 1.1f};
        float32x4_t r2 = {1.2f, 1.2f, 1.2f, 1.2f};
        float32x4_t r3 = {1.3f, 1.3f, 1.3f, 1.3f};
        float32x4_t r4 = {1.4f, 1.4f, 1.4f, 1.4f};
        float32x4_t r5 = {1.5f, 1.5f, 1.5f, 1.5f};
        float32x4_t r6 = {1.6f, 1.6f, 1.6f, 1.6f};
        float32x4_t r7 = {1.7f, 1.7f, 1.7f, 1.7f};
        float32x4_t r8 = {1.8f, 1.8f, 1.8f, 1.8f};
        float32x4_t r9 = {1.9f, 1.9f, 1.9f, 1.9f};
        float32x4_t rA = {2.0f, 2.0f, 2.0f, 2.0f};
        float32x4_t rB = {2.1f, 2.1f, 2.1f, 2.1f};
        for (size_t i = 0; i < iterations; i++){
            flops_muladd_chains12_unroll2_ops48(
                vaddq_f32, vsubq_f32, vmulq_f32,
                add0, sub0, mul0, mul1,
                r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
            );
        }
        flops_reduce_chains12(
            vaddq_f32,
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
        );
        result = vaddvq_f32(r0);

        //  (4 ops / vector) * (48 ops / macro)
        return iterations * 4 * 48;
    }
};
////////////////////////////////////////////////////////////////////////////////
//  Fused Multiply Add
 class bench_fma_linear_f32v2_aarch64_chains12 : public benchmark{
         virtual void print_meta() const{
             cout << "Single-Precision - 128-bit Aarch64 - Fused Multiply Add:" << endl;
             cout << "    Dependency Chains = 12" << endl;
         }
         virtual largeint_t run_loop(largeint_t iterations, double &result) const{
             const float32x4_t mul0 = {(float)TEST_MUL_MUL, (float)TEST_MUL_MUL,\
                  (float)TEST_MUL_MUL, (float)TEST_MUL_MUL};
             const float32x4_t mul1 = {(float)TEST_MUL_DIV, (float)TEST_MUL_DIV,\
                  (float)TEST_MUL_DIV, (float)TEST_MUL_DIV};

             float32x4_t r0 = {1.0f, 1.0f, 1.0f, 1.0f};
             float32x4_t r1 = {1.1f, 1.1f, 1.1f, 1.1f};
             float32x4_t r2 = {1.2f, 1.2f, 1.2f, 1.2f};
             float32x4_t r3 = {1.3f, 1.3f, 1.3f, 1.3f};
             float32x4_t r4 = {1.4f, 1.4f, 1.4f, 1.4f};
             float32x4_t r5 = {1.5f, 1.5f, 1.5f, 1.5f};
             float32x4_t r6 = {1.6f, 1.6f, 1.6f, 1.6f};
             float32x4_t r7 = {1.7f, 1.7f, 1.7f, 1.7f};
             float32x4_t r8 = {1.8f, 1.8f, 1.8f, 1.8f};
             float32x4_t r9 = {1.9f, 1.9f, 1.9f, 1.9f};
             float32x4_t rA = {2.0f, 2.0f, 2.0f, 2.0f};
             float32x4_t rB = {2.1f, 2.1f, 2.1f, 2.1f};
             for (size_t i = 0; i < iterations; i++){
                 flops_fma_linear_chains12_unroll2_ops48(
                     vfmaq_f32, vfmsq_f32,
                     mul0, mul1,
                     r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
                 );
             }
             flops_reduce_chains12(
                   vaddq_f32,
                   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
               );
             result = vaddvq_f32(r0);

             //  (8 ops / vector) * (48 ops / macro)
             return iterations * 8 * 48;
         }
     };
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 }
 #endif
