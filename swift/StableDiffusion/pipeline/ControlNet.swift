// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// U-Net noise prediction model for stable diffusion
@available(iOS 16.2, macOS 13.1, *)
public struct ControlNet: ResourceManaging {
    public enum Method: String, Hashable, Decodable {
        case canny
        case depth
        case pose
        case mlsd
        case normal
        case scribble
        case hed
        case segmentation
        case unknown
        
        public init(from decoder: Swift.Decoder) throws {
            do {
                let container = try decoder.singleValueContainer()
                let rawValue = try container.decode(String.self)
                if let value = Self(rawValue: rawValue) {
                    self = value
                } else {
                    self = .unknown
                }
            } catch {
                self = .unknown
            }
        }
    }
    
    public struct Info {
        public let id: String
        public let converterVersion: String?
        public let attentionImplementation: AttentionImplementation?
        public let width: Int?
        public let height: Int?
        public let method: Method
        
        public var sampleSize: CGSize? {
            guard let width, let height else { return nil }
            return CGSize(width: width, height: height)
        }
    }

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    ///
    /// It can be in the form of a single model or multiple stages
    var model: ManagedMLModel
    public var info: Info

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - url: Location of single U-Net  compiled Core ML model
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public init(modelAt url: URL,
                configuration: MLModelConfiguration) {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
        self.info = ControlNet.info(for: url)
    }
    
    public var conditioningScale: Float = 1.0
    public var image: CGImage? {
        didSet {
            let resizedImage = image?.resize(size: sampleSize)
            if let data = resizedImage?.toMLShapeArray(min: 0.0) {
                imageData = MLShapedArray<Float32>(concatenating: [data, data], alongAxis: 0)
            } else {
                imageData = nil
            }
        }
    }
    public var imageData: MLShapedArray<Float32>?

    /// Ensure the model has been loaded into memory
    public func loadResources() throws {
        try model.loadResources()
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
       model.unloadResources()
    }
    
    var timestepDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName["timestep"]!
        }
    }
    
    /// The expected shape of the models timestemp input
    public var timestepShape: [Int] {
        timestepDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
    
    var inputImageDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName["controlnet_cond"]!
        }
    }

    /// The expected shape of the models latent sample input
    public var inputImageShape: [Int] {
        inputImageDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
    
    public var sampleSize: CGSize {
        let width: Int = inputImageShape[3]
        let height: Int = inputImageShape[2]
        return CGSize(width: width, height: height)
    }

    /// Batch prediction noise from latent samples
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    /// - Returns: Array of predicted noise residuals
    func predictResiduals(
        latents: [MLShapedArray<Float32>],
        timeStep: Int,
        hiddenStates: MLShapedArray<Float32>
    ) throws -> [([MLShapedArray<Float32>], MLShapedArray<Float32>)]? {
        guard let imageData else { return nil }
        
        // Match time step batch dimension to the model / latent samples
        let t = MLShapedArray<Float32>(repeating: Float32(timeStep), shape: timestepShape)

        // Form batch input to model
        let inputs = try latents.map {
            let dict: [String: Any] = [
                "sample" : MLMultiArray($0),
                "timestep" : MLMultiArray(t),
                "encoder_hidden_states": MLMultiArray(hiddenStates),
                "controlnet_cond": MLMultiArray(imageData)
            ]
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)

        // Batch predict with model
        let results = try model.perform { model in
            try model.predictions(fromBatch: batch)
        }

        // Pull out the results in Float32 format
        let noise = (0..<results.count).map { i in
            let result = results.features(at: i)
            
            // To conform to this func return type make sure we return float32
            // Use the fact that the concatenating constructor for MLMultiArray
            // can do type conversion:
            
            let downResNoise = [
                result.featureValue(for: "down_block_res_samples_00")!.multiArrayValue!,
                result.featureValue(for: "down_block_res_samples_01")!.multiArrayValue!,
                result.featureValue(for: "down_block_res_samples_02")!.multiArrayValue!,
                result.featureValue(for: "down_block_res_samples_03")!.multiArrayValue!,
                result.featureValue(for: "down_block_res_samples_04")!.multiArrayValue!,
                result.featureValue(for: "down_block_res_samples_05")!.multiArrayValue!,
                result.featureValue(for: "down_block_res_samples_06")!.multiArrayValue!,
                result.featureValue(for: "down_block_res_samples_07")!.multiArrayValue!,
                result.featureValue(for: "down_block_res_samples_08")!.multiArrayValue!,
                result.featureValue(for: "down_block_res_samples_09")!.multiArrayValue!,
                result.featureValue(for: "down_block_res_samples_10")!.multiArrayValue!,
                result.featureValue(for: "down_block_res_samples_11")!.multiArrayValue!,
            ]
            let downResFp32Noise = downResNoise.map {
                MLShapedArray<Float32>(MLMultiArray(
                    concatenating: [$0],
                    axis: 0,
                    dataType: .float32
                )).scaled(conditioningScale)
            }
            
            let midResNoise = result.featureValue(for: "mid_block_res_sample")!.multiArrayValue!
            let midResFp32Noise = MLMultiArray(
                concatenating: [midResNoise],
                axis: 0,
                dataType: .float32
            )
            return (downResFp32Noise, MLShapedArray<Float32>(midResFp32Noise).scaled(conditioningScale))
        }

        return noise
    }
    
    public static func info(for url: URL) -> Info {
        do {
            return try JSONDecoder().decode(Info.self, from: Data(contentsOf: url.appendingPathComponent("guernika.json")))
        } catch {
            return Info(
                id: url.lastPathComponent,
                converterVersion: nil,
                attentionImplementation: nil,
                width: nil,
                height: nil,
                method: .unknown
            )
        }
    }
}

extension MLShapedArray where ArrayLiteralElement == Float32 {
    func scaled(_ scaleFactor: Float32) -> MLShapedArray<Float32> {
        guard scaleFactor != 1 else { return self }
        return MLShapedArray<Float32>(
            scalars: scalars.map { $0 * scaleFactor },
            shape: shape
        )
    }
}

@available(macOS 13.1, *)
extension ControlNet.Info: Decodable {
    enum CodingKeys: String, CodingKey {
        case id = "identifier"
        case converterVersion = "converter_version"
        case attentionImplementation = "attention_implementation"
        case width
        case height
        case method
    }
}
