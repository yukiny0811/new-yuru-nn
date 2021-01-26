//
//  Matrix.swift
//  NewYuruNN
//
//  Created by クワシマ・ユウキ on 2021/01/25.
//

import Metal
import Accelerate

struct Matrix : CustomStringConvertible{
    
    let laObject: la_object_t
    
    var description: String {
        return self.get2DArray().map {
            $0.description
        }.joined(separator: "\n")
    }
    
    public init(_ array: [Float], _ tate: Int, _ yoko: Int) {
        self.laObject = la_matrix_from_float_buffer(array, la_count_t(tate), la_count_t(yoko), la_count_t(yoko), la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES))
    }
    
    public init(_ laObject: la_object_t) {
        self.laObject = laObject
    }
    
    public var row: Int {
        get {
            return Int(la_matrix_rows(self.laObject))
        }
    }
    
    public var col: Int {
        get {
            return Int(la_matrix_cols(self.laObject))
        }
    }
    
    public var array: [Float] {
        get {
            var array: [Float] = [Float](repeating: 0, count: self.row * self.col)
            la_matrix_to_float_buffer(&array, la_count_t(self.col), self.laObject)
            return array
        }
    }
    
    subscript(row: Int, col: Int) -> Float {
        return self.array[row * self.col + col]
    }
    
    public func get2DArray() -> [[Float]] {
        var result: [[Float]] = []
        for i in 0..<self.row {
            var temp: [Float] = []
            for j in 0..<self.col {
                temp.append(self[i, j])
            }
            result.append(temp)
        }
        return result
    }
    
    public func sum(_ other: Matrix) -> Matrix {
        let obj = la_sum(self.laObject, other.laObject)
        return Matrix(obj)
    }
    
    public func subtract(_ other: Matrix) -> Matrix {
        let obj = la_difference(self.laObject, other.laObject)
        return Matrix(obj)
    }
    
    public func multiplyScalar(_ scalar: Float) -> Matrix {
        let obj = la_scale_with_float(self.laObject, scalar)
        return Matrix(obj)
    }
    
    public func elementwiseProduct(_ other: Matrix) -> Matrix {
        let obj = la_elementwise_product(self.laObject, other.laObject)
        return Matrix(obj)
    }
    
    public func product(_ other: Matrix) -> Matrix {
        let obj = la_matrix_product(self.laObject, other.laObject)
        return Matrix(obj)
    }
    
    public func transpose() -> Matrix {
        let obj = la_transpose(self.laObject)
        return Matrix(obj)
    }
    
    public func sumOfAllElementsInRow() -> Matrix {
        let array = self.get2DArray()
        var result: [Float] = []
        for a in 0..<array.count{
            result.append(array[a].reduce(0, +))
        }
        return Matrix(result, result.count, 1)
    }
    
    
    
    
    public static func sum(_ m1: Matrix, _ m2: Matrix) -> Matrix {
        let obj = la_sum(m1.laObject, m2.laObject)
        return Matrix(obj)
    }
    
    public static func subtract(_ m1: Matrix, _ m2: Matrix) -> Matrix {
        let obj = la_difference(m1.laObject, m2.laObject)
        return Matrix(obj)
    }
    
    public static func multiplyScalar(_ m1: Matrix, _ s: Float) -> Matrix {
        let obj = la_scale_with_float(m1.laObject, s)
        return Matrix(obj)
    }
    
    public static func elementwiseProduct(_ m1: Matrix, _ m2: Matrix) -> Matrix {
        let obj = la_elementwise_product(m1.laObject, m2.laObject)
        return Matrix(obj)
    }
    
    public static func product(_ m1: Matrix, _ m2: Matrix) -> Matrix {
        let obj = la_matrix_product(m1.laObject, m2.laObject)
        return Matrix(obj)
    }
    
    public static func transpose(_ m: Matrix) -> Matrix {
        let obj = la_transpose(m.laObject)
        return Matrix(obj)
    }
    
    
    
    public func sumOfAllElements() -> Float {
        return self.array.reduce(0, +)
    }
    
    public func expAllElements() -> Matrix {
        return YuruGPUCore.Run_ExpOfMatrix(self)
    }
    
    public func copy() -> Matrix {
        let mat = self
        return mat
    }
    
    public func deepcopy() -> Matrix {
        return Matrix(self.array, self.row, self.col)
    }
    
    public static func createZeros(_ tate: Int, _ yoko: Int) -> Matrix {
        return Matrix([Float](repeating: 0, count: tate * yoko), tate, yoko)
    }
    
    public static func createOnes(_ tate: Int, _ yoko: Int) -> Matrix {
        return Matrix([Float](repeating: 1, count: tate * yoko), tate, yoko)
    }
    
}

//class Matrix : CustomStringConvertible{
//
//    let laObject: la_object_t
//
//    var description: String {
//        return self.get2DArray().map {
//            $0.description
//        }.joined(separator: "\n")
//    }
//
//    public init(_ array: [Float], _ tate: Int, _ yoko: Int) {
//        self.laObject = la_matrix_from_float_buffer(array, la_count_t(tate), la_count_t(yoko), la_count_t(yoko), la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES))
//    }
//
//    public init(_ laObject: la_object_t) {
//        self.laObject = laObject
//    }
//
//    private(set) public lazy var row: Int = { [unowned self] in
//        return Int(la_matrix_rows(self.laObject))
//    }()
//
//    private(set) public lazy var col: Int = { [unowned self] in
//        return Int(la_matrix_cols(self.laObject))
//    }()
//
//    private(set) public lazy var array: [Float] = { [unowned self] in
//        var array: [Float] = [Float](repeating: 0, count: self.row * self.col)
//        la_matrix_to_float_buffer(&array, la_count_t(self.col), self.laObject)
//        return array
//    }()
//
//    subscript(row: Int, col: Int) -> Float {
//        return self.array[row * self.col + col]
//    }
//
//    public func get2DArray() -> [[Float]] {
//        var result: [[Float]] = []
//        for i in 0..<self.row {
//            var temp: [Float] = []
//            for j in 0..<self.col {
//                temp.append(self[i, j])
//            }
//            result.append(temp)
//        }
//        return result
//    }
//
//    public func sum(_ other: Matrix) -> Matrix {
//        let obj = la_sum(self.laObject, other.laObject)
//        return Matrix(obj)
//    }
//
//    public func subtract(_ other: Matrix) -> Matrix {
//        let obj = la_difference(self.laObject, other.laObject)
//        return Matrix(obj)
//    }
//
//    public func multiplyScalar(_ scalar: Float) -> Matrix {
//        let obj = la_scale_with_float(self.laObject, scalar)
//        return Matrix(obj)
//    }
//
//    public func elementwiseProduct(_ other: Matrix) -> Matrix {
//        let obj = la_elementwise_product(self.laObject, other.laObject)
//        return Matrix(obj)
//    }
//
//    public func product(_ other: Matrix) -> Matrix {
//        let obj = la_matrix_product(self.laObject, other.laObject)
//        return Matrix(obj)
//    }
//
//    public func transpose() -> Matrix {
//        let obj = la_transpose(self.laObject)
//        return Matrix(obj)
//    }
//
//
//
//
//    public static func sum(_ m1: Matrix, _ m2: Matrix) -> Matrix {
//        let obj = la_sum(m1.laObject, m2.laObject)
//        return Matrix(obj)
//    }
//
//    public static func subtract(_ m1: Matrix, _ m2: Matrix) -> Matrix {
//        let obj = la_difference(m1.laObject, m2.laObject)
//        return Matrix(obj)
//    }
//
//    public static func multiplyScalar(_ m1: Matrix, _ s: Float) -> Matrix {
//        let obj = la_scale_with_float(m1.laObject, s)
//        return Matrix(obj)
//    }
//
//    public static func elementwiseProduct(_ m1: Matrix, _ m2: Matrix) -> Matrix {
//        let obj = la_elementwise_product(m1.laObject, m2.laObject)
//        return Matrix(obj)
//    }
//
//    public static func product(_ m1: Matrix, _ m2: Matrix) -> Matrix {
//        let obj = la_matrix_product(m1.laObject, m2.laObject)
//        return Matrix(obj)
//    }
//
//    public static func transpose(_ m: Matrix) -> Matrix {
//        let obj = la_transpose(m.laObject)
//        return Matrix(obj)
//    }
//
//
//
//    public func sumOfAllElements() -> Float {
//        return self.array.reduce(0, +)
//    }
//
//    public func expAllElements() -> Matrix {
//        return YuruGPUCore.Run_ExpOfMatrix(self)
//    }
//
//    public func copy() -> Matrix {
//        let obj = autoreleasepool {
//            self.laObject
//        }
//        return Matrix(obj)
//    }
//
//    public static func createZeros(_ tate: Int, _ yoko: Int) -> Matrix {
//        return Matrix([Float](repeating: 0, count: tate * yoko), tate, yoko)
//    }
//
//    public static func createOnes(_ tate: Int, _ yoko: Int) -> Matrix {
//        return Matrix([Float](repeating: 1, count: tate * yoko), tate, yoko)
//    }
//
//}
