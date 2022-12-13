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
        let width: Int = image.width
        let height: Int = image.height
        let channelShape = [1, image.width, image.height]
        var format: vImage_CGImageFormat = .init(cgImage: image)!
        if format.bitsPerPixel == 32 {
            let uint8ImageAlpha = try! PixelBufferI8x4(cgImage: image, cgImageFormat: &format)
            let uint8Image: PixelBufferI8x3 = PixelBufferI8x3(width: width, height: height)
            
            // Drop alpha channel
            uint8ImageAlpha.convert(to: uint8Image, channelOrdering: vImage.ChannelOrdering.RGBA)
            
            let floatImage = PixelBufferIFx3(width: width, height: height)
            uint8Image.convert(to: floatImage) // maps [0 255] -> [0.0 1.0] and clips
            
            let floatChannels = floatImage.planarBuffers()
            let arrayChannels = floatChannels.map { channel in
                let cOut = PixelBufferPFx1(width: width, height: height)
                channel.multiply(by: 2.0, preBias: 0.0, postBias: -1.0, destination: cOut)
            
                return MLShapedArray(scalars: cOut.array, shape: channelShape)
            }
            
            var array = MLShapedArray<Float32>(concatenating: arrayChannels, alongAxis: 0)
            array = MLShapedArray<Float32>(scalars: array.scalars, shape: [1, 3, height, width])
            return array
        } else {
            let singleChannelUint8Image = try! PixelBufferP8x1(cgImage: image, cgImageFormat: &format)
            let channel = PixelBufferPFx1(width: width, height: height)
            singleChannelUint8Image.convert(to: channel)
            return MLShapedArray<Float32>(scalars: channel.array, shape: [1, 1, height, width])
        }
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
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        return context.makeImage()!
    }
}
