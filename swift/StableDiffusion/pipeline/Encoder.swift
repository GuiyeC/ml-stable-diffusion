//
//  Encoder.swift
//  
//
//  Created by Guillermo Cique Fernández on 2/12/22.
//

import Foundation
import CoreML
import Accelerate

/// A encoder model which produces RGB images from latent samples
@available(iOS 16.2, macOS 13.1, *)
public struct Encoder: ResourceManaging {

    /// VAE encoder model
    var model: ManagedMLModel

    /// Create encoder from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled VAE encoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: A encoder that will lazily load its required resources when needed or requested
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }

    /// Ensure the model has been loaded into memory
    public func loadResources() throws {
        try model.loadResources()
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
       model.unloadResources()
    }
    
    /// Request resources are pre-warmed by loading and unloading
    func prewarmResources() throws -> CGSize {
        try loadResources()
        let expectedInputSize = expectedInputSize
        unloadResources()
        return expectedInputSize
    }

    /// Prediction queue
    let queue = DispatchQueue(label: "encoder.predict")
    
    var inputImageDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName["z"]!
        }
    }

    /// The expected shape of the models latent sample input
    public var inputImageShape: [Int] {
        inputImageDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
    
    public var expectedInputSize: CGSize {
        let width: Int = inputImageShape[3]
        let height: Int = inputImageShape[2]
        return CGSize(width: width, height: height)
    }

    /// Encode image into latent samples
    ///
    ///  - Parameters:
    ///    - image: Input image
    ///  - Returns: Latent samples to decode
    public func encode(
        _ image: CGImage,
        scaleFactor: Float32 = 0.18215,
        random: ((Float32, Float32) -> Float32)
    ) throws -> MLShapedArray<Float32> {
        let resizedImage = image.resize(size: expectedInputSize)
        let imageData = resizedImage.toMLShapeArray()
        return try encode(imageData, scaleFactor: scaleFactor, random: random)
    }
    
    public func encode(
        _ imageData: MLShapedArray<Float32>,
        scaleFactor: Float32 = 0.18215,
        random: ((Float32, Float32) -> Float32)
    ) throws -> MLShapedArray<Float32> {
        let dict = [inputName: MLMultiArray(imageData)]
        let input = try MLDictionaryFeatureProvider(dictionary: dict)
        
        let result = try model.perform { model in
            try model.prediction(from: input)
        }
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
            scalars: latent.scalars.map { $0 * scaleFactor },
            shape: [1] + latent.shape
        )

        return latentScaled
    }

    var inputName: String {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.key
        }
    }
}
