//
//  CGImage.swift
//  
//
//  Created by Guillermo Cique Fernández on 26/2/23.
//

import CoreML
import Accelerate

extension CGImage {
    typealias PixelBufferPFx1 = vImage.PixelBuffer<vImage.PlanarF>
    typealias PixelBufferP8x3 = vImage.PixelBuffer<vImage.Planar8x3>
    typealias PixelBufferIFx3 = vImage.PixelBuffer<vImage.InterleavedFx3>
    typealias PixelBufferP8x1 = vImage.PixelBuffer<vImage.Planar8>
    typealias PixelBufferI8x2 = vImage.PixelBuffer<vImage.Interleaved8x2>
    typealias PixelBufferI8x3 = vImage.PixelBuffer<vImage.Interleaved8x3>
    typealias PixelBufferI8x4 = vImage.PixelBuffer<vImage.Interleaved8x4>
    
    public func toMLShapeArray(min: Float = -1.0, max: Float = 1.0) -> MLShapedArray<Float32> {
        var sourceFormat = vImage_CGImageFormat(cgImage: self)!
        var mediumFormat = vImage_CGImageFormat(
            bitsPerComponent: 8 * MemoryLayout<UInt8>.size,
            bitsPerPixel: 8 * MemoryLayout<UInt8>.size * 4,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.first.rawValue)
        )!
        let width = vImagePixelCount(exactly: width)!
        let height = vImagePixelCount(exactly: height)!
        
        var sourceImageBuffer = try! vImage_Buffer(cgImage: self)
        
        var mediumDesination = try! vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: mediumFormat.bitsPerPixel)
        
        let converter = vImageConverter_CreateWithCGImageFormat(
            &sourceFormat,
            &mediumFormat,
            nil,
            vImage_Flags(kvImagePrintDiagnosticsToConsole),
            nil
        )!.takeRetainedValue()
        
        vImageConvert_AnyToAny(converter, &sourceImageBuffer, &mediumDesination, nil, vImage_Flags(kvImagePrintDiagnosticsToConsole))
        
        var destinationA = try! vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
        var destinationR = try! vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
        var destinationG = try! vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
        var destinationB = try! vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
        
        var minFloat: [Float] = [min, min, min, min]
        var maxFloat: [Float] = [max, max, max, max]
        
        vImageConvert_ARGB8888toPlanarF(&mediumDesination, &destinationA, &destinationR, &destinationG, &destinationB, &maxFloat, &minFloat, .zero)
        
        let redData = Data(bytes: destinationR.data, count: Int(width) * Int(height) * MemoryLayout<Float>.size)
        let greenData = Data(bytes: destinationG.data, count: Int(width) * Int(height) * MemoryLayout<Float>.size)
        let blueData = Data(bytes: destinationB.data, count: Int(width) * Int(height) * MemoryLayout<Float>.size)
        
        let imageData = redData + greenData + blueData
        
        let shapedArray = MLShapedArray<Float32>(data: imageData, shape: [1, 3, Int(height), Int(width)])
        
        return shapedArray
    }
    
    func toAlphaMLShapeArray() -> MLShapedArray<Float32> {
        var sourceFormat = vImage_CGImageFormat(cgImage: self)!
        var mediumFormat = vImage_CGImageFormat(
            bitsPerComponent: 8 * MemoryLayout<UInt8>.size,
            bitsPerPixel: 8 * MemoryLayout<UInt8>.size * 4,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.first.rawValue)
        )!
        let width = vImagePixelCount(exactly: width)!
        let height = vImagePixelCount(exactly: height)!
        
        var sourceImageBuffer = try! vImage_Buffer(cgImage: self)
        
        var mediumDesination = try! vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: mediumFormat.bitsPerPixel)
        
        let converter = vImageConverter_CreateWithCGImageFormat(
            &sourceFormat,
            &mediumFormat,
            nil,
            vImage_Flags(kvImagePrintDiagnosticsToConsole),
            nil
        )!.takeRetainedValue()
        
        vImageConvert_AnyToAny(converter, &sourceImageBuffer, &mediumDesination, nil, vImage_Flags(kvImagePrintDiagnosticsToConsole))
        
        var destinationA = try! vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
        var destinationR = try! vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
        var destinationG = try! vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
        var destinationB = try! vImage_Buffer(width: Int(width), height: Int(height), bitsPerPixel: 8 * UInt32(MemoryLayout<Float>.size))
        
        var minFloat: [Float] = [0.0, 0.0, 0.0, 0.0]
        var maxFloat: [Float] = [1.0, 1.0, 1.0, 1.0]
        
        vImageConvert_ARGB8888toPlanarF(&mediumDesination, &destinationA, &destinationR, &destinationG, &destinationB, &maxFloat, &minFloat, .zero)
        
        let alphaData = Data(bytes: destinationA.data, count: Int(width) * Int(height) * MemoryLayout<Float>.size)
        let shapedArray = MLShapedArray<Float32>(data: alphaData, shape: [1, 1, Int(height), Int(width)])
        return shapedArray
    }
    
    func resize(size: CGSize) -> CGImage {
        let width: Int = Int(size.width)
        let height: Int = Int(size.height)
        guard self.width != width || self.height != height else { return self }

        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        context.interpolationQuality = .high
        context.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))

        return context.makeImage()!
    }
}
