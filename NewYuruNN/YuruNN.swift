//
//  YuruNN.swift
//  NewYuruNN
//
//  Created by クワシマ・ユウキ on 2021/01/25.
//

import Foundation


class NN_Backprop_Kaiki {
    
    let saveData = UserDefaults.standard
    
    public init() {
        let inputData = YuruCalc.createFloatData(0, Float.pi * 2, 0.1)
        let correctData = inputData.map {
            sin($0)
        }
        let normalizedInputData = inputData.map {
            ($0 - Float.pi) / Float.pi
        }
        
        let lr: Float = 0.1
        let epoch = 2000
        
        let midLayer = NN_Backprop_Kaiki_MiddleLayer(upperLayerCount: 1, thisLayerCount: 3)
        let outLayer = NN_Backprop_Kaiki_OutputLayer(upperLayerCount: 3, thisLayerCount: 1)
        
        for i in 0..<correctData.count{
            print("raw:\(inputData[i]),input:\(normalizedInputData[i]),output:\(correctData[i])")
        }
        
        for k in 0..<epoch {
            autoreleasepool {
                var totalError: Float = 0
                var plotX: [Float] = []
                var plotY: [Float] = []
                
                for i in 0..<correctData.count{
                    
                    
                    
                    let x = normalizedInputData[i]
                    let c = correctData[i]
//
                    let tempInput = Matrix([x], 1, 1)
                    let tempCorrect = Matrix([c], 1, 1)
//
                    midLayer.forward(input: tempInput)
                    outLayer.forward(input: midLayer.output)
////
                    outLayer.back(correct: tempCorrect)
                    midLayer.back(gradOutput: outLayer.gradInput)
//
                    autoreleasepool {
                        midLayer.update(lr: lr)
                    }
                    
                    outLayer.update(lr: lr)
                    
////                    let res = outLayer.output.getFloatArray()[0]
////                    totalError += 1.0/2.0 * sqrt(res - correctData[i])
//
//                    if k % 200 == 0{
//                        let res = outLayer.output.array[0]
//                        totalError += 1.0/2.0 * sqrt(res - correctData[i])
//                        plotX.append(x)
//                        plotY.append(res)
//                        print(plotX)
//                        print(plotY)
//                        print(totalError)
//                        print(k)
//                    }
//
//
                }
            
                print(k)
            }
            
            
        }
        
//            saveData.set([(midLayer.weight.array, midLayer.weight.row, midLayer.weight.col),
//                          (midLayer.bias.array, midLayer.bias.row, midLayer.bias.col),
//                          (outLayer.weight.array, outLayer.weight.row, outLayer.weight.col),
//                          (outLayer.bias.array, outLayer.bias.row, outLayer.bias.col)], forKey: "model")
        
        let arrayDataRaw: [[Float]] = [midLayer.weight.array, midLayer.bias.array, outLayer.weight.array, outLayer.bias.array]
        let rowColDataRaw: [[Float]] = [
            [Float(midLayer.weight.row), Float(midLayer.weight.col)],
            [Float(midLayer.bias.row), Float(midLayer.bias.col)],
            [Float(outLayer.weight.row), Float(outLayer.weight.col)],
            [Float(outLayer.bias.row), Float(outLayer.bias.col)]
        ]
        let data = try! JSONEncoder().encode([arrayDataRaw, rowColDataRaw])
        saveData.set(data, forKey: "model")
        print("finished")
        
        
    }
    
    
}

class NN_Backprop_Kaiki_MiddleLayer {
    var weight: Matrix!
    var bias: Matrix!
    var input: Matrix!
    
    var gradW: Matrix!
    var gradB: Matrix!
    var gradInput: Matrix!
    
    var output: Matrix!
    
    public init(upperLayerCount: Int, thisLayerCount: Int){
        autoreleasepool{
            weight = autoreleasepool {
                Matrix(YuruCalc.createRandomData(-0.5, 0.5, upperLayerCount * thisLayerCount), upperLayerCount, thisLayerCount)
            }
            bias = autoreleasepool {
                Matrix(YuruCalc.createRandomData(-0.5, 0.5, thisLayerCount), 1, thisLayerCount)
            }
        }
        
    }
    
    public func forward(input: Matrix){
        self.input = input.copy()
        let u = input.product(weight).sum(bias)
        self.output = YuruCalc.sigmoid(x: u)
    }
    
    public func back(gradOutput: Matrix){
        let ones = Matrix.createOnes(self.output.row, self.output.col)
        let temp = ones.subtract(self.output)
        let delta = gradOutput.elementwiseProduct(temp).elementwiseProduct(self.output)
        
        self.gradW = input.transpose().product(delta)
        self.gradB = delta.copy()
        self.gradInput = delta.product(self.weight.transpose())
    }
    
    public func update(lr: Float) {
        autoreleasepool {
//            let gradWc = autoreleasepool {
//                self.gradW.copy()
//            }
//            let gradBc = autoreleasepool {
//                self.gradB.copy()
//            }
//
//            let weightC = autoreleasepool {
//                self.weight.copy()
//            }
//            let biasC = autoreleasepool {
//                self.bias.copy()
//            }
            
            let mulW = autoreleasepool{
                self.gradW.multiplyScalar(lr)
            }
            let mulB = autoreleasepool {
                self.gradB.multiplyScalar(lr)
            }
//
            self.weight = autoreleasepool {
                self.weight.subtract(mulW).deepcopy()
            }
//
            self.bias = autoreleasepool {
                self.bias.subtract(mulB).deepcopy()
            }
        }
        
    }
}

class NN_Backprop_Kaiki_OutputLayer {
    var weight: Matrix
    var bias: Matrix
    var input: Matrix!
    
    var gradW: Matrix!
    var gradB: Matrix!
    var gradInput: Matrix!
    
    var output: Matrix!
    
    public init(upperLayerCount: Int, thisLayerCount: Int){
        
        weight = autoreleasepool {
            Matrix(YuruCalc.createRandomData(-0.5, 0.5, upperLayerCount * thisLayerCount), upperLayerCount, thisLayerCount)
        }
        bias = autoreleasepool {
            Matrix(YuruCalc.createRandomData(-0.5, 0.5, thisLayerCount), 1, thisLayerCount)
        }
    }
    
    public func forward(input: Matrix){
        self.input = input.copy()
        self.output = input.product(weight).sum(bias)
        
    }
    
    public func back(correct: Matrix){
        let delta = self.output.subtract(correct)
        self.gradW = input.transpose().product(delta)
        self.gradB = delta.copy()
        self.gradInput = delta.product(self.weight.transpose())
        
//        print(gradW.array)
    }
    
    public func update(lr: Float) {
        self.weight = autoreleasepool {
            self.weight.subtract(self.gradW.multiplyScalar(lr)).deepcopy()
        }
        self.bias = autoreleasepool {
            self.bias.subtract(self.gradB.multiplyScalar(lr)).deepcopy()
        }
    }
}
