/* bench_f64v1_aarch64.h
 *
 * Author           : Vsevolod Nikolskiy, Alexander J. Yee
 * Date Created     : 10/04/2017
 * Last Modified    : 11/04/2017
 *
 *      f64     =   Double Precision
 *      v1      =   Vectorize by 2
 *      aarch   =   ARMv8 Aarch64 instruction set
 *
 */

 #ifndef _flops_bench_f64v1_aarch64_H
 #define _flops_bench_f64v1_aarch64_H
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
 #include "../macros/macro_fma.h"
 namespace Flops{
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////

 class bench_add_mul_f64v1_aarch64_chains12 : public benchmark{
     virtual void print_meta() const{
         cout << "Double-Precision - 128-bit Aarch64 - Add/Sub linear interleave Multiply:" << endl;
         cout << "    Dependency Chains  = 12" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         const float64x2_t add0 = {TEST_ADD_ADD, TEST_ADD_ADD};
         const float64x2_t sub0 = {TEST_ADD_SUB, TEST_ADD_SUB};

         const float64x2_t mul0 = {TEST_MUL_MUL, TEST_MUL_MUL};
         const float64x2_t mul1 = {TEST_MUL_DIV, TEST_MUL_DIV};

         float64x2_t r0 = {1.0, 1.01};
         float64x2_t r1 = {1.1, 1.11};
         float64x2_t r2 = {1.2, 1.21};
         float64x2_t r3 = {1.3, 1.31};
         float64x2_t r4 = {1.4, 1.41};
         float64x2_t r5 = {1.5, 1.51};
         float64x2_t r6 = {1.6, 1.61};
         float64x2_t r7 = {1.7, 1.71};
         float64x2_t r8 = {1.7, 1.71};
         float64x2_t r9 = {1.7, 1.71};
         float64x2_t rA = {1.7, 1.71};
         float64x2_t rB = {1.7, 1.71};

         for (size_t i = 0; i < iterations; i++){
              r0 = vaddq_f64(r0, add0);
              r1 = vmulq_f64(r1, mul0);
              r2 = vaddq_f64(r2, add0);
              r3 = vmulq_f64(r3, mul0);
              r4 = vaddq_f64(r4, add0);
              r5 = vmulq_f64(r5, mul0);
              r6 = vaddq_f64(r6, add0);
              r7 = vmulq_f64(r7, mul0);
              r8 = vaddq_f64(r8, add0);
              r9 = vmulq_f64(r9, mul0);
              rA = vaddq_f64(rA, add0);
              rB = vmulq_f64(rB, mul0);

              r0 = vsubq_f64(r0, sub0);
              r1 = vmulq_f64(r1, mul1);
              r2 = vsubq_f64(r2, sub0);
              r3 = vmulq_f64(r3, mul1);
              r4 = vsubq_f64(r4, sub0);
              r5 = vmulq_f64(r5, mul1);
              r6 = vsubq_f64(r6, sub0);
              r7 = vmulq_f64(r7, mul1);
              r8 = vsubq_f64(r8, sub0);
              r9 = vmulq_f64(r9, mul1);
              rA = vsubq_f64(rA, sub0);
              rB = vmulq_f64(rB, mul1);

              r0 = vaddq_f64(r0, add0);
              r1 = vmulq_f64(r1, mul0);
              r2 = vaddq_f64(r2, add0);
              r3 = vmulq_f64(r3, mul0);
              r4 = vaddq_f64(r4, add0);
              r5 = vmulq_f64(r5, mul0);
              r6 = vaddq_f64(r6, add0);
              r7 = vmulq_f64(r7, mul0);
              r8 = vaddq_f64(r8, add0);
              r9 = vmulq_f64(r9, mul0);
              rA = vaddq_f64(rA, add0);
              rB = vmulq_f64(rB, mul0);

              r0 = vsubq_f64(r0, sub0);
              r1 = vmulq_f64(r1, mul1);
              r2 = vsubq_f64(r2, sub0);
              r3 = vmulq_f64(r3, mul1);
              r4 = vsubq_f64(r4, sub0);
              r5 = vmulq_f64(r5, mul1);
              r6 = vsubq_f64(r6, sub0);
              r7 = vmulq_f64(r7, mul1);
              r8 = vsubq_f64(r8, sub0);
              r9 = vmulq_f64(r9, mul1);
              rA = vsubq_f64(rA, sub0);
              rB = vmulq_f64(rB, mul1);
         }
         flops_reduce_chains12(
             vaddq_f64,
             r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
         );
         result = vaddvq_f64(r0);
         return iterations * 2 * 48;
     }
 };



 //  Add
class bench_add_f64v1_aarch64_chains4 : public benchmark{
     virtual void print_meta() const{
         cout << "Double-Precision - 128-bit Aarch64 - Add/Sub:" << endl;
         cout << "    Dependency Chains  = 4" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         const float64x2_t add0 = {TEST_ADD_ADD,TEST_ADD_ADD};
         const float64x2_t sub0 = {TEST_ADD_SUB,TEST_ADD_SUB};

         float64x2_t r0 = {1.0, 1.01};
         float64x2_t r1 = {1.1, 1.11};
         float64x2_t r2 = {1.2, 1.21};
         float64x2_t r3 = {1.3, 1.31};
         for (size_t i = 0; i < iterations; i++){
             flops_add_chains4_unroll2_ops16(
                 vaddq_f64, vsubq_f64,
                 add0, sub0,
                 r0, r1, r2, r3
             );
         }
         flops_reduce_chains4(
             vaddq_f64,
             r0, r1, r2, r3
         );
         result = vaddvq_f64(r0);

         //  (2 ops / vector) * (16 ops / macro)
         return iterations * 2 * 16;
     }
 };
 //  Add
 class bench_add_f64v1_aarch64_chains8 : public benchmark{
     virtual void print_meta() const{
         cout << "Double-Precision - 128-bit Aarch64 - Add/Sub:" << endl;
         cout << "    Dependency Chains  = 8" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         const float64x2_t add0 = {TEST_ADD_ADD,TEST_ADD_ADD};
         const float64x2_t sub0 = {TEST_ADD_SUB,TEST_ADD_SUB};

         float64x2_t r0 = {1.0, 1.01};
         float64x2_t r1 = {1.1, 1.11};
         float64x2_t r2 = {1.2, 1.21};
         float64x2_t r3 = {1.3, 1.31};
         float64x2_t r4 = {1.4, 1.41};
         float64x2_t r5 = {1.5, 1.51};
         float64x2_t r6 = {1.6, 1.61};
         float64x2_t r7 = {1.7, 1.71};
         for (size_t i = 0; i < iterations; i++){
             flops_add_chains8_unroll2_ops32(
                 vaddq_f64, vsubq_f64,
                 add0, sub0,
                 r0, r1, r2, r3, r4, r5, r6, r7
             );
         }
         flops_reduce_chains8(
             vaddq_f64,
             r0, r1, r2, r3, r4, r5, r6, r7
         );
         result = vaddvq_f64(r0);

         //  (2 ops / vector) * (32 ops / macro)
         return iterations * 2 * 32;
     }
 };
 //////////////////////////////////////////////////////////////////////////////
 //  Multiply
 class bench_mul_f64v1_aarch64_chains8 : public benchmark{
     virtual void print_meta() const{
         cout << "Double-Precision - 128-bit Aarch64 - Multiply:" << endl;
         cout << "    Dependency Chains = 8" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         const float64x2_t mul0 = {TEST_MUL_MUL, TEST_MUL_MUL};
         const float64x2_t mul1 = {TEST_MUL_DIV, TEST_MUL_DIV};

         float64x2_t r0 = {1.0, 1.0};
         float64x2_t r1 = {1.1, 1.1};
         float64x2_t r2 = {1.2, 1.2};
         float64x2_t r3 = {1.3, 1.3};
         float64x2_t r4 = {1.4, 1.4};
         float64x2_t r5 = {1.5, 1.5};
         float64x2_t r6 = {1.6, 1.6};
         float64x2_t r7 = {1.7, 1.7};
         for (size_t i = 0; i < iterations; i++){
             flops_mul_chains8_unroll2_ops32(
                 vmulq_f64,
                 mul0, mul1,
                 r0, r1, r2, r3, r4, r5, r6, r7
             );
         }
         flops_reduce_chains8(
             vaddq_f64,
             r0, r1, r2, r3, r4, r5, r6, r7
         );
         result = vaddvq_f64(r0);

         //  (2 ops / vector) * (32 ops / macro)
         return iterations * 2 * 32;
     }
 };
 class bench_mul_f64v1_aarch64_chains12 : public benchmark{
     virtual void print_meta() const{
         cout << "Double-Precision - 128-bit Aarch64 - Multiply:" << endl;
         cout << "    Dependency Chains = 12" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         const float64x2_t mul0 = {TEST_MUL_MUL, TEST_MUL_MUL};
         const float64x2_t mul1 = {TEST_MUL_DIV, TEST_MUL_DIV};

         float64x2_t r0 = {1.0, 1.0};
         float64x2_t r1 = {1.1, 1.1};
         float64x2_t r2 = {1.2, 1.2};
         float64x2_t r3 = {1.3, 1.3};
         float64x2_t r4 = {1.4, 1.4};
         float64x2_t r5 = {1.5, 1.5};
         float64x2_t r6 = {1.6, 1.6};
         float64x2_t r7 = {1.7, 1.7};
         float64x2_t r8 = {1.7, 1.7};
         float64x2_t r9 = {1.7, 1.7};
         float64x2_t rA = {1.7, 1.7};
         float64x2_t rB = {1.7, 1.7};
         for (size_t i = 0; i < iterations; i++){
             flops_mul_chains12_unroll2_ops48(
                 vmulq_f64,
                 mul0, mul1,
                 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
             );
         }
         flops_reduce_chains12(
             vaddq_f64,
             r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
         );
         result = vaddvq_f64(r0);

         //  (2 ops / vector) * (48 ops / macro)
         return iterations * 2 * 48;
     }
 };
 //////////////////////////////////////////////////////////////////////////////
 //  Fused Multiply Add
 class bench_fma_linear_f64v1_aarch64_chains12 : public benchmark{
         virtual void print_meta() const{
             cout << "Double-Precision - 128-bit Aarch64 - Fused Multiply Add:" << endl;
             cout << "    Dependency Chains = 12" << endl;
         }
         virtual largeint_t run_loop(largeint_t iterations, double &result) const{
             const float64x2_t mul0 = {TEST_MUL_MUL,TEST_MUL_MUL};
             const float64x2_t mul1 = {TEST_MUL_DIV,TEST_MUL_DIV};
             const float64x2_t add0 = {TEST_ADD_ADD,TEST_ADD_ADD};

             float64x2_t r0 = {1.0f, 1.01f};
             float64x2_t r1 = {1.1f, 1.11f};
             float64x2_t r2 = {1.2f, 1.21f};
             float64x2_t r3 = {1.3f, 1.31f};
             float64x2_t r4 = {1.4f, 1.41f};
             float64x2_t r5 = {1.5f, 1.51f};
             float64x2_t r6 = {1.6f, 1.61f};
             float64x2_t r7 = {1.7f, 1.71f};
             float64x2_t r8 = {1.8f, 1.81f};
             float64x2_t r9 = {1.9f, 1.91f};
             float64x2_t rA = {2.0f, 2.01f};
             float64x2_t rB = {2.1f, 2.11f};
             for (size_t i = 0; i < iterations; i++){
                 flops_fma_arm_linear_chains12_ops24(
                     vfmaq_f64, vfmsq_f64,
                     add0, mul0, mul1,
                     r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
                 );
             }
             flops_reduce_chains12(
                   vaddq_f64,
                   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
               );
             result = vaddvq_f64(r0);

             //  (4 ops / vector) * (24 ops / macro)
             return iterations * 4 * 24;
         }
     };
 //  Multiply + Add
 class bench_mac_f64v1_aarch64_chains12 : public benchmark{
     virtual void print_meta() const{
         cout << "Double-Precision - 128-bit Aarch64 - Multiply + Add:" << endl;
         cout << "    Dependency Chains = 12" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         float64x2_t add0 = {TEST_ADD_ADD,TEST_ADD_ADD};
         float64x2_t sub0 = {TEST_ADD_SUB+0.000000000000005,TEST_ADD_SUB+0.000000000000005};
         float64x2_t mul0 = {TEST_MUL_MUL,TEST_MUL_MUL};
         float64x2_t mul1 = {TEST_MUL_DIV+0.000000000000005,TEST_MUL_DIV+0.000000000000005};

         float64x2_t r0 = {0.9f, 0.9f};
         float64x2_t r1 = {1.1f, 1.1f};
         float64x2_t r2 = {1.2f, 1.2f};
         float64x2_t r3 = {1.3f, 1.3f};
         float64x2_t r4 = {1.4f, 1.4f};
         float64x2_t r5 = {1.5f, 1.5f};
         float64x2_t r6 = {1.6f, 1.6f};
         float64x2_t r7 = {1.7f, 1.7f};
         float64x2_t r8 = {1.8f, 1.8f};
         float64x2_t r9 = {1.9f, 1.9f};
         float64x2_t rA = {2.0f, 2.0f};
         float64x2_t rB = {2.1f, 2.1f};

         float64x2_t r10 = {0.19f, 0.19f};
         float64x2_t r11 = {1.11f, 1.11f};
         float64x2_t r12 = {1.12f, 1.12f};
         float64x2_t r13 = {1.13f, 1.13f};
         float64x2_t r14 = {1.14f, 1.14f};
         float64x2_t r15 = {1.15f, 1.15f};
         float64x2_t r16 = {1.16f, 1.16f};
         float64x2_t r17 = {1.17f, 1.17f};
         float64x2_t r18 = {1.18f, 1.18f};
         float64x2_t r19 = {1.19f, 1.19f};
         float64x2_t r1A = {2.12f, 2.12f};
         float64x2_t r1B = {2.11f, 2.11f};
         for (size_t i = 0; i < iterations; i++){
             flops_muladd_chains12_ops12(
                 vaddq_f64, vmulq_f64,
                 add0, mul0, mul1,
                 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
             );
             flops_muladd_chains12_ops12(
                 vaddq_f64, vmulq_f64,
                 add0, mul1, mul0,
                 r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r1A, r1B
             );
             flops_muladd_chains12_ops12(
                 vsubq_f64, vmulq_f64,
                 sub0, mul0, mul1,
                 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
             );
             flops_muladd_chains12_ops12(
                 vsubq_f64, vmulq_f64,
                 sub0, mul1, mul0,
                 r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r1A, r1B
             );
         }
         flops_reduce_chains12(
             vaddq_f64,
             r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
         );
         flops_reduce_chains12(
             vaddq_f64,
             r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r1A, r1B
         );
         result = vaddvq_f64(r10)+vaddvq_f64(r0);

         //  (2 ops / vector) * (12 ops / macro)
         return iterations * 2 * 12 * 4;
     }
 };
 //  Multiply + Add
 class bench_mac_f64v0_aarch64_chains12 : public benchmark{
     virtual void print_meta() const{
         cout << "Double-Precision - 64-bit Aarch64 - Multiply + Add:" << endl;
         cout << "    Dependency Chains = 12" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         float64x1_t add0 = {TEST_ADD_ADD};
         float64x1_t sub0 = {TEST_ADD_SUB+0.000000000000001};
         float64x1_t mul0 = {TEST_MUL_MUL};
         float64x1_t mul1 = {TEST_MUL_DIV+0.000000000000001};

         float64x1_t r0 = {0.9f};
         float64x1_t r1 = {1.1f};
         float64x1_t r2 = {1.2f};
         float64x1_t r3 = {1.3f};
         float64x1_t r4 = {1.4f};
         float64x1_t r5 = {1.5f};
         float64x1_t r6 = {1.6f};
         float64x1_t r7 = {1.7f};
         float64x1_t r8 = {1.8f};
         float64x1_t r9 = {1.9f};
         float64x1_t rA = {2.0f};
         float64x1_t rB = {2.1f};

         for (size_t i = 0; i < iterations; i++){
             flops_muladd_chains12_ops12(
                 vadd_f64, vmul_f64,
                 add0, mul0, mul1,
                 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
             );
         }
         flops_reduce_chains12(
             vadd_f64,
             r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
         );
         result = vget_lane_f64(r0, 0);

         //  (2 ops / vector) * (12 ops / macro)
         return iterations * 1 * 12 * 2;
     }
 };

 /// FMA + add
 class bench_fma_add_f64v1_aarch64_chains12 : public benchmark{
     virtual void print_meta() const{
         cout << "Double-Precision - 128-bit Aarch64 - FMA + Add:" << endl;
         cout << "    Dependency Chains = 12" << endl;
     }
     virtual largeint_t run_loop(largeint_t iterations, double &result) const{
         float64x2_t sub0 = {TEST_ADD_SUB+0.000000000000001,TEST_ADD_SUB+0.000000000000001};


         const float64x2_t mul0 = {TEST_MUL_MUL,TEST_MUL_MUL};
         const float64x2_t mul1 = {TEST_MUL_DIV,TEST_MUL_DIV};
         const float64x2_t add0 = {TEST_ADD_ADD,TEST_ADD_ADD};

         float64x2_t r0 = {1.0f, 1.01f};
         float64x2_t r1 = {1.1f, 1.11f};
         float64x2_t r2 = {1.2f, 1.21f};
         float64x2_t r3 = {1.3f, 1.31f};
         float64x2_t r4 = {1.4f, 1.41f};
         float64x2_t r5 = {1.5f, 1.51f};
         float64x2_t r6 = {1.6f, 1.61f};
         float64x2_t r7 = {1.7f, 1.71f};
         float64x2_t r8 = {1.8f, 1.81f};
         float64x2_t r9 = {1.9f, 1.91f};
         float64x2_t rA = {2.0f, 2.01f};
         float64x2_t rB = {2.1f, 2.11f};

         for (size_t i = 0; i < iterations; i++){
                r0 = vfmaq_f64(r0, mul0, add0); r1 = vaddq_f64(r1, add0);
                r2 = vfmaq_f64(r2, mul0, add0); r3 = vaddq_f64(r3, add0);
                r4 = vfmaq_f64(r4, mul0, add0); r5 = vaddq_f64(r5, add0);
                r6 = vfmaq_f64(r6, mul0, add0); r7 = vaddq_f64(r7, add0);
                r8 = vfmaq_f64(r8, mul0, add0); r9 = vaddq_f64(r9, add0);
                rA = vfmaq_f64(rA, mul0, add0); rB = vaddq_f64(rB, add0);

                r0 = vfmsq_f64(r0, mul1, add0); r1 = vsubq_f64(r1, sub0);
                r2 = vfmsq_f64(r2, mul1, add0); r3 = vsubq_f64(r3, sub0);
                r4 = vfmsq_f64(r4, mul1, add0); r5 = vsubq_f64(r5, sub0);
                r6 = vfmsq_f64(r6, mul1, add0); r7 = vsubq_f64(r7, sub0);
                r8 = vfmsq_f64(r8, mul1, add0); r9 = vsubq_f64(r9, sub0);
                rA = vfmsq_f64(rA, mul1, add0); rB = vsubq_f64(rB, sub0);

         }
         flops_reduce_chains12(
               vaddq_f64,
               r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB
           );
         result = vaddvq_f64(r0);


         return iterations * (12*2*2 + 12*2);
     }
 };
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 }
 #endif
