//
//  Encoder.swift
//  
//
//  Created by Guillermo Cique Fernández on 2/12/22.
//

import Foundation
import CoreML
import Accelerate

/// A decoder model which produces RGB images from latent samples
public struct Encoder {

    /// VAE encoder model
    var model: MLModel

    /// Create encoder from Core ML model
    ///
    /// - Parameters
    ///     - model: Core ML model for VAE encoder
    public init(model: MLModel) {
        self.model = model
    }

    /// Prediction queue
    let queue = DispatchQueue(label: "encoder.predict")
    
    var inputImageDescription: MLFeatureDescription {
        model.modelDescription.inputDescriptionsByName["z"]!
    }

    /// The expected shape of the models latent sample input
    public var inputImageShape: [Int] {
        inputImageDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

    /// Encode image into latent samples
    ///
    ///  - Parameters:
    ///    - image: Input image
    ///  - Returns: Latent samples to decode
    public func encode(_ image: CGImage, random: ((Float32, Float32) -> Float32)) throws -> MLShapedArray<Float32> {
        let width: Int = inputImageShape[3]
        let height: Int = inputImageShape[2]
        let resizedImage = resizeImage(image, size: CGSize(width: width, height: height))
        let imageData = fromRGBCGImage(resizedImage)
        return try encode(imageData, random: random)
    }
    
    public func encode(_ imageData: MLShapedArray<Float32>, random: ((Float32, Float32) -> Float32)) throws -> MLShapedArray<Float32> {
        let dict = [inputName: MLMultiArray(imageData)]
        let input = try MLDictionaryFeatureProvider(dictionary: dict)

        let result = try queue.sync { try model.prediction(from: input) }
        let outputName = result.featureNames.first!
        let outputValue = result.featureValue(for: outputName)!.multiArrayValue!
        let output = MLShapedArray<Float32>(outputValue)
        
        // DiagonalGaussianDistribution
        let mean = output[0][0..<4]
        let logvar = MLShapedArray<Float32>(
            scalars: output[0][4..<8].scalars.map { min(max($0, -30), 20) },
            shape: mean.shape
        )
        let std = MLShapedArray<Float32>(
            scalars: logvar.scalars.map { exp(0.5 * $0) },
            shape: logvar.shape
        )
        let latent = MLShapedArray<Float32>(
            scalars: zip(mean.scalars, std.scalars).map { random($0, $1) },
            shape: logvar.shape
        )
        
        // Reference pipeline scales the latent after encoding
        let latentScaled = MLShapedArray<Float32>(
            scalars: latent.scalars.map { $0 * 0.18215 },
            shape: [1] + latent.shape
        )

        return latentScaled
    }

    var inputName: String {
        model.modelDescription.inputDescriptionsByName.first!.key
    }

    typealias PixelBufferPFx1 = vImage.PixelBuffer<vImage.PlanarF>
    typealias PixelBufferP8x3 = vImage.PixelBuffer<vImage.Planar8x3>
    typealias PixelBufferIFx3 = vImage.PixelBuffer<vImage.InterleavedFx3>
    typealias PixelBufferP8x1 = vImage.PixelBuffer<vImage.Planar8>
    typealias PixelBufferI8x2 = vImage.PixelBuffer<vImage.Interleaved8x2>
    typealias PixelBufferI8x3 = vImage.PixelBuffer<vImage.Interleaved8x3>
    typealias PixelBufferI8x4 = vImage.PixelBuffer<vImage.Interleaved8x4>
    
    func fromRGBCGImage(_ image: CGImage) -> MLShapedArray<Float32> {
        var sourceFormat = vImage_CGImageFormat(cgImage: image)!
        var mediumFormat = vImage_CGImageFormat(
            bitsPerComponent: 8 * MemoryLayout<UInt8>.size,
            bitsPerPixel: 8 * MemoryLayout<UInt8>.size * 4,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.first.rawValue)
        )!
        let width = vImagePixelCount(exactly: image.width)!
        let height = vImagePixelCount(exactly: image.height)!
        
        var sourceImageBuffer = try! vImage_Buffer(cgImage: image)
        
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
        
        var minFloat: [Float] = [-1.0, -1.0, -1.0, -1.0]
        var maxFloat: [Float] = [1.0, 1.0, 1.0, 1.0]
        
        vImageConvert_ARGB8888toPlanarF(&mediumDesination, &destinationA, &destinationR, &destinationG, &destinationB, &maxFloat, &minFloat, .zero)
        
        let redData = Data(bytes: destinationR.data, count: Int(width) * Int(height) * MemoryLayout<Float>.size)
        let greenData = Data(bytes: destinationG.data, count: Int(width) * Int(height) * MemoryLayout<Float>.size)
        let blueData = Data(bytes: destinationB.data, count: Int(width) * Int(height) * MemoryLayout<Float>.size)
        
        let imageData = redData + greenData + blueData
        
        let shapedArray = MLShapedArray<Float32>(data: imageData, shape: [1, 3, Int(height), Int(width)])
        
        return shapedArray
    }
    
    func alphaFromRGBCGImage(_ image: CGImage) -> MLShapedArray<Float32> {
        var sourceFormat = vImage_CGImageFormat(cgImage: image)!
        var mediumFormat = vImage_CGImageFormat(
            bitsPerComponent: 8 * MemoryLayout<UInt8>.size,
            bitsPerPixel: 8 * MemoryLayout<UInt8>.size * 4,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.first.rawValue)
        )!
        let width = vImagePixelCount(exactly: image.width)!
        let height = vImagePixelCount(exactly: image.height)!
        
        var sourceImageBuffer = try! vImage_Buffer(cgImage: image)
        
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
    
    func resizeImage(_ image: CGImage, size: CGSize) -> CGImage {
        let width: Int = Int(size.width)
        let height: Int = Int(size.height)
        guard image.width != width || image.height != height else { return image }

        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: image.bitsPerComponent,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        return context.makeImage()!
    }
}
