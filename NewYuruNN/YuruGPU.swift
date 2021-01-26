//
//  YuruGPU.swift
//  NewYuruNN
//
//  Created by クワシマ・ユウキ on 2021/01/25.
//

import Metal

class YuruGPUCore {
    
    public static var device: MTLDevice!
    public static var library: MTLLibrary!
    public static var commandQueue: MTLCommandQueue!
    
    public static func start() {
        self.device = MTLCreateSystemDefaultDevice()!
        let frameworkBundle = Bundle(for: YuruGPUCore.self)
        library = try! YuruGPUCore.device.makeDefaultLibrary(bundle: frameworkBundle)
        commandQueue = self.device.makeCommandQueue()!
        
        
        pipelineStates["exp_of_matrix"] = try! device.makeComputePipelineState(function: library.makeFunction(name: "exp_of_matrix")!)
        pipelineStates["scalar_divided_by_matrix"] = try! device.makeComputePipelineState(function: library.makeFunction(name: "scalar_divided_by_matrix")!)
        
    }
    
    public static var pipelineStates: [String: MTLComputePipelineState] = [:]
    
    public static func Run_Default(functionName: String, inputDatas: [[Float]], row: Int, col: Int) -> Matrix {
        
        return autoreleasepool {
            //commandqueue and mtlLibrary here
            //setup data
            let outputData: [Float] = autoreleasepool {
                [Float](repeating: 0, count: row * col)
            }

            //metal setup
            let commandBuffer: MTLCommandBuffer! = autoreleasepool {
                commandQueue.makeCommandBuffer()
            }
            let computeCommandEncoder: MTLComputeCommandEncoder! = autoreleasepool {
                commandBuffer.makeComputeCommandEncoder()
            }
            
            autoreleasepool {
                computeCommandEncoder.setComputePipelineState(pipelineStates[functionName]!)
            }
            
    //
    //        //create buffer
            let inputBuffers: [MTLBuffer] = autoreleasepool {
                var inputBuffers: [MTLBuffer] = []
                for i in 0..<inputDatas.count {
                    let tempt: MTLBuffer = autoreleasepool {
                        YuruGPUCore.device.makeBuffer(bytes: inputDatas[i], length: MemoryLayout<Float>.size * inputDatas[i].count, options: [])!
                    }
                    inputBuffers.append(tempt)
                }
                return inputBuffers
            }
            
            let outputBuffer = autoreleasepool {
                return YuruGPUCore.device.makeBuffer(bytes: outputData, length: MemoryLayout<Float>.size * outputData.count, options: [])
            }
    //
    //        //set buffer
            autoreleasepool {
                for i in 0..<inputBuffers.count {
                    computeCommandEncoder.setBuffer(inputBuffers[i], offset: 0, index: i)
                }
                computeCommandEncoder.setBuffer(outputBuffer, offset: 0, index: inputBuffers.count)
            }
            
    //
    //        //calculate threads
            let width = autoreleasepool {
                pipelineStates[functionName]!.threadExecutionWidth
            }
            let threadGroupsPerGrid = autoreleasepool {
                MTLSize(width: (outputData.count + width - 1) / width, height: 1, depth: 1)
            }
            let threadsPerThreadGroup = autoreleasepool {
                MTLSize(width: width, height: 1, depth: 1)
            }
            autoreleasepool {
                computeCommandEncoder.dispatchThreadgroups(threadGroupsPerGrid, threadsPerThreadgroup: threadsPerThreadGroup)
            }
            
    //
    //        //end command encoding
            autoreleasepool {
                computeCommandEncoder.endEncoding()
            }
            
    //
    //        //commit command buffer
            
            autoreleasepool {
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
            }
            
            
    //
    //        //get result
            
            let resultData = autoreleasepool {
                Data(bytesNoCopy: outputBuffer!.contents(), count: MemoryLayout<Float>.size * outputData.count, deallocator: .none)
            }
            let newOutputData = autoreleasepool {
                resultData.withUnsafeBytes {
                    Array(
                        UnsafeBufferPointer(
                            start: $0.baseAddress!.assumingMemoryBound(to: Float.self),
                            count: $0.count / MemoryLayout<Float>.size
                        )
                    )
                }
            }
            
            return Matrix(newOutputData, row, col)
        }
    }

    public static func Run_ExpOfMatrix(_ matrix: Matrix) -> Matrix {
        let temp = autoreleasepool{
            YuruGPUCore.Run_Default(functionName: "exp_of_matrix", inputDatas: [matrix.array], row: matrix.row, col: matrix.col)
        }
        return temp
    }

    public static func Run_ScalarDividedByMatrix(_ matrix: Matrix, scalar: Float) -> Matrix {
        let temp = autoreleasepool{
            YuruGPUCore.Run_Default(functionName: "scalar_divided_by_matrix", inputDatas: [matrix.array, [scalar]], row: matrix.row, col: matrix.col)
        }
        return temp
    }
    
//    private static func Run_Default(functionName: String, inputDatas: Matrix) -> Matrix {
//
//        //commandqueue and mtlLibrary here
//
//        //setup data
//        let outputData: [Float] = [Float](repeating: 0, count: inputDatas.row * inputDatas.col)
//
//        //metal setup
//        let commandBuffer: MTLCommandBuffer! = commandQueue.makeCommandBuffer()
//        let computeCommandEncoder: MTLComputeCommandEncoder! = commandBuffer.makeComputeCommandEncoder()
//        let function: MTLFunction! = shaderFunctions[functionName]
//        let computePipelineState: MTLComputePipelineState! = try! YuruGPUCore.device.makeComputePipelineState(function: function)
//        computeCommandEncoder.setComputePipelineState(computePipelineState)
////
////        //create buffer
//
//        let inputBuffer = YuruGPUCore.device.makeBuffer(bytes: inputDatas.array, length: MemoryLayout<Float>.size * inputDatas.array.count, options: [])
//
//        let outputBuffer = YuruGPUCore.device.makeBuffer(bytes: outputData, length: MemoryLayout<Float>.size * outputData.count, options: [])
////
////        //set buffer
//        computeCommandEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
//        computeCommandEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
////
////        //calculate threads
//        let width = computePipelineState.threadExecutionWidth
//        let threadGroupsPerGrid = MTLSize(width: (outputData.count + width - 1) / width, height: 1, depth: 1)
//        let threadsPerThreadGroup = MTLSize(width: width, height: 1, depth: 1)
//        computeCommandEncoder.dispatchThreadgroups(threadGroupsPerGrid, threadsPerThreadgroup: threadsPerThreadGroup)
////
////        //end command encoding
//        computeCommandEncoder.endEncoding()
////
////        //commit command buffer
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
////
////        //get result
//        let resultData = Data(bytesNoCopy: outputBuffer!.contents(), count: MemoryLayout<Float>.size * outputData.count, deallocator: .none)
//        let newOutputData = resultData.withUnsafeBytes {
//            Array(
//                UnsafeBufferPointer(
//                    start: $0.baseAddress!.assumingMemoryBound(to: Float.self),
//                    count: $0.count / MemoryLayout<Float>.size
//                )
//            )
//        }
//        return Matrix(newOutputData, inputDatas.row, inputDatas.col)
//
////        return Matrix([1], 1, 1)
//    }
//
//
//    public static func Run_ExpOfMatrix(_ matrix: Matrix) -> Matrix {
//        return YuruGPUCore.Run_Default(functionName: "exp_of_matrix", inputDatas: matrix)
//    }
//
//    public static func Run_ScalarDividedByMatrix(_ matrix: Matrix, scalar: Float) -> Matrix {
//        //commandqueue and mtlLibrary here
//
//        //setup data
//        let outputData: [Float] = [Float](repeating: 0, count: matrix.row * matrix.col)
//
//        //metal setup
//        let commandBuffer: MTLCommandBuffer! = commandQueue.makeCommandBuffer()
//        let computeCommandEncoder: MTLComputeCommandEncoder! = commandBuffer.makeComputeCommandEncoder()
//        let function: MTLFunction! = shaderFunctions["scalar_divided_by_matrix"]
//        let computePipelineState: MTLComputePipelineState! = try! YuruGPUCore.device.makeComputePipelineState(function: function)
//        computeCommandEncoder.setComputePipelineState(computePipelineState)
////
////        //create buffer
//
//        let inputBuffer = YuruGPUCore.device.makeBuffer(bytes: matrix.array, length: MemoryLayout<Float>.size * matrix.array.count, options: [])
//        let scalarBuffer = YuruGPUCore.device.makeBuffer(bytes: [scalar], length: MemoryLayout<Float>.size * 1, options: [])
//
//        let outputBuffer = YuruGPUCore.device.makeBuffer(bytes: outputData, length: MemoryLayout<Float>.size * outputData.count, options: [])
////
////        //set buffer
//        computeCommandEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
//        computeCommandEncoder.setBuffer(scalarBuffer, offset: 0, index: 1)
//        computeCommandEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
////
////        //calculate threads
//        let width = computePipelineState.threadExecutionWidth
//        let threadGroupsPerGrid = MTLSize(width: (outputData.count + width - 1) / width, height: 1, depth: 1)
//        let threadsPerThreadGroup = MTLSize(width: width, height: 1, depth: 1)
//        computeCommandEncoder.dispatchThreadgroups(threadGroupsPerGrid, threadsPerThreadgroup: threadsPerThreadGroup)
////
////        //end command encoding
//        computeCommandEncoder.endEncoding()
////
////        //commit command buffer
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
////
////        //get result
//        let resultData = Data(bytesNoCopy: outputBuffer!.contents(), count: MemoryLayout<Float>.size * outputData.count, deallocator: .none)
//        let newOutputData = resultData.withUnsafeBytes {
//            Array(
//                UnsafeBufferPointer(
//                    start: $0.baseAddress!.assumingMemoryBound(to: Float.self),
//                    count: $0.count / MemoryLayout<Float>.size
//                )
//            )
//        }
//        return Matrix(newOutputData, matrix.row, matrix.col)
//    }
}


