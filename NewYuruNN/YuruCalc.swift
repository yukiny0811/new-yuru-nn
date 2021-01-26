//
//  YuruCalc.swift
//  NewYuruNN
//
//  Created by クワシマ・ユウキ on 2021/01/25.
//

import Metal
import Foundation

class YuruCalc {
    public static func createFloatData(_ startFloat: Float, _ endFloat: Float, _ interval: Float) -> [Float] {
        var resultData: [Float] = []
        for i in stride(from: startFloat, to: endFloat, by: interval){
            resultData.append(i)
        }
        return resultData
    }
    
    public static func createRandomData(_ from: Float, _ to: Float, _ count: Int) -> [Float] {
        var resultData: [Float] = []
        for _ in 0..<count{
            resultData.append(Float.random(in: from...to))
        }
        return resultData
    }
    
    public static func sigmoid(x: Matrix) -> Matrix{
        
        let expFinishedMatrix = autoreleasepool {
            YuruGPUCore.Run_ExpOfMatrix(x.multiplyScalar(-1))
        }
        let ones = autoreleasepool {
            Matrix.createOnes(expFinishedMatrix.row, expFinishedMatrix.col)
        }
        let bunbo = autoreleasepool {
            expFinishedMatrix.sum(ones)
        }
        let result = autoreleasepool {
            YuruGPUCore.Run_ScalarDividedByMatrix(bunbo, scalar: 1)
        }
        return result
    }
    
}
