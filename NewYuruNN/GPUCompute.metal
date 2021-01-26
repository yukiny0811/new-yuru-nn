//
//  GPUCompute.metal
//  NewYuruNN
//
//  Created by クワシマ・ユウキ on 2021/01/25.
//

#include <metal_stdlib>
using namespace metal;

kernel void exp_of_matrix (const device float* inputData [[ buffer(0) ]],
                           device float* outputData [[ buffer(1) ]],
                           uint thread_position_in_grid [[thread_position_in_grid]]) {
    outputData[thread_position_in_grid] = exp(inputData[thread_position_in_grid]);
}

kernel void scalar_divided_by_matrix (const device float* inputData [[ buffer(0) ]],
                                      const device float* scalar [[ buffer(1) ]],
                                      device float* outputData [[ buffer(2) ]],
                                      uint thread_position_in_grid [[thread_position_in_grid]]) {
    outputData[thread_position_in_grid] = scalar[0] / inputData[thread_position_in_grid];
}

