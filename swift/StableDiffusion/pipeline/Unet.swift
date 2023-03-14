// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// U-Net noise prediction model for stable diffusion
@available(iOS 16.2, macOS 13.1, *)
public struct Unet: ResourceManaging {
    public enum Function: String, Decodable {
        case standard
        case inpaint
        case instructions
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
        public let controlNetSupport: Bool?
        public let function: Function
        public let hiddenSize: Int?
        
        public var sampleSize: CGSize? {
            guard let width, let height else { return nil }
            return CGSize(width: width, height: height)
        }
    }

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    ///
    /// It can be in the form of a single model or multiple stages
    var models: [ManagedMLModel]
    public var info: Info

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - url: Location of single U-Net  compiled Core ML model
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public init(modelAt url: URL,
                configuration: MLModelConfiguration) {
        self.models = [ManagedMLModel(modelAt: url, configuration: configuration)]
        self.info = Unet.info(for: url.deletingLastPathComponent())
    }

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - urls: Location of chunked U-Net via urls to each compiled chunk
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public init(chunksAt urls: [URL],
                configuration: MLModelConfiguration) {
        self.models = urls.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
        self.info = Unet.info(for: urls.first!.deletingLastPathComponent())
    }

    /// Load resources.
    public func loadResources() throws {
        for model in models {
            try model.loadResources()
        }
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
        for model in models {
            model.unloadResources()
        }
    }

    /// Pre-warm resources
    public func prewarmResources() throws -> (Bool, Bool, Bool) {
        // Override default to pre-warm each model
        let model = models.first
        try model?.loadResources()
        let canInpaint = canInpaint
        let takesInstructions = takesInstructions
        let supportsControlNet = supportsControlNet
        model?.unloadResources()
        for model in models.dropFirst() {
            try model.loadResources()
            model.unloadResources()
        }
        return (canInpaint, takesInstructions, supportsControlNet)
    }

    var latentSampleDescription: MLFeatureDescription {
        try! models.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["sample"]!
        }
    }

    /// The expected shape of the models latent sample input
    public var latentSampleShape: [Int] {
        latentSampleDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
    
    var timestepDescription: MLFeatureDescription {
        try! models.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["timestep"]!
        }
    }
    
    /// The expected shape of the models timestemp input
    public var timestepShape: [Int] {
        timestepDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
    
    public var canInpaint: Bool {
        latentSampleShape[1] == 9
    }
    
    public var takesInstructions: Bool {
        timestepShape[0] == 3
    }
    
    public var supportsControlNet: Bool {
        try! models.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["mid_block_res_sample"] != nil
        }
    }

    /// Batch prediction noise from latent samples
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    /// - Returns: Array of predicted noise residuals
    func predictNoise(
        latents: [MLShapedArray<Float32>],
        additionalResiduals: [([MLShapedArray<Float32>], MLShapedArray<Float32>)]?,
        timeStep: Int,
        hiddenStates: MLShapedArray<Float32>
    ) throws -> [MLShapedArray<Float32>] {

        // Match time step batch dimension to the model / latent samples
        let t = MLShapedArray<Float32>(repeating: Float32(timeStep), shape: timestepShape)

        // Form batch input to model
        let inputs: [MLDictionaryFeatureProvider]
        if let additionalResiduals {
            inputs = try zip(latents, additionalResiduals).map { latent, residuals in
                let dict: [String: Any] = [
                    "sample" : MLMultiArray(latent),
                    "timestep" : MLMultiArray(t),
                    "encoder_hidden_states": MLMultiArray(hiddenStates),
                    "down_block_res_samples_00": MLMultiArray(residuals.0[0]),
                    "down_block_res_samples_01": MLMultiArray(residuals.0[1]),
                    "down_block_res_samples_02": MLMultiArray(residuals.0[2]),
                    "down_block_res_samples_03": MLMultiArray(residuals.0[3]),
                    "down_block_res_samples_04": MLMultiArray(residuals.0[4]),
                    "down_block_res_samples_05": MLMultiArray(residuals.0[5]),
                    "down_block_res_samples_06": MLMultiArray(residuals.0[6]),
                    "down_block_res_samples_07": MLMultiArray(residuals.0[7]),
                    "down_block_res_samples_08": MLMultiArray(residuals.0[8]),
                    "down_block_res_samples_09": MLMultiArray(residuals.0[9]),
                    "down_block_res_samples_10": MLMultiArray(residuals.0[10]),
                    "down_block_res_samples_11": MLMultiArray(residuals.0[11]),
                    "mid_block_res_sample": MLMultiArray(residuals.1)
                ]
                return try MLDictionaryFeatureProvider(dictionary: dict)
            }
        } else {
            inputs = try latents.map {
                let dict: [String: Any] = [
                    "sample" : MLMultiArray($0),
                    "timestep" : MLMultiArray(t),
                    "encoder_hidden_states": MLMultiArray(hiddenStates),
                    "down_block_res_samples_00": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 320, 64, 64])),
                    "down_block_res_samples_01": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 320, 64, 64])),
                    "down_block_res_samples_02": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 320, 64, 64])),
                    "down_block_res_samples_03": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 320, 32, 32])),
                    "down_block_res_samples_04": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 640, 32, 32])),
                    "down_block_res_samples_05": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 640, 32, 32])),
                    "down_block_res_samples_06": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 640, 16, 16])),
                    "down_block_res_samples_07": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 1280, 16, 16])),
                    "down_block_res_samples_08": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 1280, 16, 16])),
                    "down_block_res_samples_09": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 1280, 8, 8])),
                    "down_block_res_samples_10": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 1280, 8, 8])),
                    "down_block_res_samples_11": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 1280, 8, 8])),
                    "mid_block_res_sample": MLMultiArray(MLShapedArray<Float32>(repeating: 0, shape: [2, 1280, 8, 8])),
                ]
                return try MLDictionaryFeatureProvider(dictionary: dict)
            }
        }
        let batch = MLArrayBatchProvider(array: inputs)

        // Make predictions
        let results = try predictions(from: batch)

        // Pull out the results in Float32 format
        let noise = (0..<results.count).map { i in

            let result = results.features(at: i)
            let outputName = result.featureNames.first!

            let outputNoise = result.featureValue(for: outputName)!.multiArrayValue!

            // To conform to this func return type make sure we return float32
            // Use the fact that the concatenating constructor for MLMultiArray
            // can do type conversion:
            let fp32Noise = MLMultiArray(
                concatenating: [outputNoise],
                axis: 0,
                dataType: .float32
            )
            return MLShapedArray<Float32>(fp32Noise)
        }

        return noise
    }

    func predictions(from batch: MLBatchProvider) throws -> MLBatchProvider {

        var results = try models.first!.perform { model in
            try model.predictions(fromBatch: batch)
        }

        if models.count == 1 {
            return results
        }

        // Manual pipeline batch prediction
        let inputs = batch.arrayOfFeatureValueDictionaries
        for stage in models.dropFirst() {

            // Combine the original inputs with the outputs of the last stage
            let next = try results.arrayOfFeatureValueDictionaries
                .enumerated().map { (index, dict) in
                    let nextDict =  dict.merging(inputs[index]) { (out, _) in out }
                    return try MLDictionaryFeatureProvider(dictionary: nextDict)
            }
            let nextBatch = MLArrayBatchProvider(array: next)

            // Predict
            results = try stage.perform { model in
                try model.predictions(fromBatch: nextBatch)
            }
        }

        return results
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
                controlNetSupport: nil,
                function: .unknown,
                hiddenSize: nil
            )
        }
    }
}

extension MLFeatureProvider {
    var featureValueDictionary: [String : MLFeatureValue] {
        self.featureNames.reduce(into: [String : MLFeatureValue]()) { result, name in
            result[name] = self.featureValue(for: name)
        }
    }
}

extension MLBatchProvider {
    var arrayOfFeatureValueDictionaries: [[String : MLFeatureValue]] {
        (0..<self.count).map {
            self.features(at: $0).featureValueDictionary
        }
    }
}

@available(macOS 13.1, *)
extension Unet.Info: Decodable {
    enum CodingKeys: String, CodingKey {
        case id = "identifier"
        case converterVersion = "converter_version"
        case attentionImplementation = "attention_implementation"
        case width
        case height
        case controlNetSupport = "controlnet_support"
        case function
        case hiddenSize = "hidden_size"
    }
}
