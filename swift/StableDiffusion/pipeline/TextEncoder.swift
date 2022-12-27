// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

///  A model for encoding text
public class TextEncoder {

    /// Text tokenizer
    var tokenizer: BPETokenizer

    /// Embedding model
    private var url: URL
    private var config: MLModelConfiguration
    private var _model: MLModel?
    func model() throws -> MLModel {
        if let _model { return _model }
        let model = try MLModel(contentsOf: url, configuration: config)
        _model = model
        return model
    }
    
    /// Creates text encoder which embeds a tokenized string
    ///
    /// - Parameters:
    ///   - tokenizer: Tokenizer for input text
    ///   - url: URL for the model for encoding tokenized text
    ///   - config: Model loading configuration
    public init(tokenizer: BPETokenizer, url: URL, config: MLModelConfiguration) {
        self.tokenizer = tokenizer
        self.url = url
        self.config = config
    }

    /// Encode input text/string
    ///
    ///  - Parameters:
    ///     - text: Input text to be tokenized and then embedded
    ///  - Returns: Embedding representing the input text
    public func encode(_ text: String) throws -> MLShapedArray<Float32> {

        // Get models expected input length
        let inputLength = inputShape.last!

        // Tokenize, padding to the expected length
        var (tokens, ids) = tokenizer.tokenize(input: text, minCount: inputLength)

        // Truncate if necessary
        if ids.count > inputLength {
            tokens = tokens.dropLast(tokens.count - inputLength)
            ids = ids.dropLast(ids.count - inputLength)
            let truncated = tokenizer.decode(tokens: tokens)
            print("Needed to truncate input '\(text)' to '\(truncated)'")
        }

        // Use the model to generate the embedding
        return try encode(ids: ids)
    }

    /// Prediction queue
    let queue = DispatchQueue(label: "textencoder.predict")

    func encode(ids: [Int]) throws -> MLShapedArray<Float32> {
        let model = try model()
        let inputName = inputDescription.name
        let inputShape = inputShape

        let floatIds = ids.map { Float32($0) }
        let inputArray = MLShapedArray<Float32>(scalars: floatIds, shape: inputShape)
        let inputFeatures = try! MLDictionaryFeatureProvider(
            dictionary: [inputName: MLMultiArray(inputArray)])

        let result = try queue.sync { try model.prediction(from: inputFeatures) }
        let embeddingFeature = result.featureValue(for: "last_hidden_state")
        return MLShapedArray<Float32>(converting: embeddingFeature!.multiArrayValue!)
    }

    var inputDescription: MLFeatureDescription {
        _model!.modelDescription.inputDescriptionsByName.first!.value
    }

    var inputShape: [Int] {
        inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
}
