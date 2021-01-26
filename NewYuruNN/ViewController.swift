//
//  ViewController.swift
//  NewYuruNN
//
//  Created by クワシマ・ユウキ on 2021/01/25.
//

import Cocoa

class ViewController: NSViewController {
    
    let saveData = UserDefaults.standard
    let fileManager = FileManager.default

    override func viewDidLoad() {
        super.viewDidLoad()
        
        YuruGPUCore.start()
        
//        NN_Backprop_Kaiki()
        
        let data = saveData.object(forKey: "model") as! Data
        let stored = try! JSONDecoder().decode([[[Float]]].self, from: data)
        print(stored)

        let mid = NN_Backprop_Kaiki_MiddleLayer(upperLayerCount: 1, thisLayerCount: 3)
        let out = NN_Backprop_Kaiki_OutputLayer(upperLayerCount: 3, thisLayerCount: 1)

        mid.weight = Matrix(stored[0][0], Int(stored[1][0][0]), Int(stored[1][0][1]))
        mid.bias = Matrix(stored[0][1], Int(stored[1][1][0]), Int(stored[1][1][1]))
        out.weight = Matrix(stored[0][2], Int(stored[1][2][0]), Int(stored[1][2][1]))
        out.bias = Matrix(stored[0][3], Int(stored[1][3][0]), Int(stored[1][3][1]))

        let input = Matrix([-0.63], 1, 1)

        mid.forward(input: input)
        out.forward(input: mid.output)
        print(out.output.array)
        
        let path = fileManager.urls(for: .documentDirectory, in: .userDomainMask)
        let url = path[0].appendingPathComponent("jsonFiles")
        try! fileManager.createDirectory(at: url, withIntermediateDirectories: true, attributes: nil)
        let jsonUrl = url.appendingPathComponent("model1.json")
        do {
            try data.write(to: jsonUrl)
        } catch {
            print("error")
        }

        print(jsonUrl)
        
        
        
        
        
        
        
//        print(stored)
//
//        print(data[0], data[1], data[2], data[3])
        

        // Do any additional setup after loading the view.
    }

    override var representedObject: Any? {
        didSet {
        // Update the view, if already loaded.
        }
    }


}

