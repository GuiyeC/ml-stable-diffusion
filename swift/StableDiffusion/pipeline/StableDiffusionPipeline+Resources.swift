// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

public extension StableDiffusionPipeline {

    /// Create stable diffusion pipeline using model resources at a
    /// specified URL
    ///
    /// - Parameters:
    ///    - baseURL: URL pointing to directory holding all model
    ///               and tokenization resources
    ///   - configuration: The configuration to load model resources with
    ///   - disableSafety: Load time disable of safety to save memory
    /// - Returns:
    ///  Pipeline ready for image generation if all  necessary resources loaded
    convenience init(resourcesAt baseURL: URL,
         configuration config: MLModelConfiguration = .init(),
         disableSafety: Bool = false) throws {

        /// Expect URL of each resource
        let textEncoderURL = baseURL.appending(path: "TextEncoder.mlmodelc")
        let unetURL = baseURL.appending(path: "Unet.mlmodelc")
        let unetChunk1URL = baseURL.appending(path: "UnetChunk1.mlmodelc")
        let unetChunk2URL = baseURL.appending(path: "UnetChunk2.mlmodelc")
        let encoderURL = baseURL.appending(path: "VAEEncoder.mlmodelc")
        let decoderURL = baseURL.appending(path: "VAEDecoder.mlmodelc")
        let safetyCheckerURL = baseURL.appending(path: "SafetyChecker.mlmodelc")
        let vocabURL = baseURL.appending(path: "vocab.json")
        let mergesURL = baseURL.appending(path: "merges.txt")

        // Text tokenizer and encoder
        let tokenizer = try BPETokenizer(mergesAt: mergesURL, vocabularyAt: vocabURL)
        let textEncoder = TextEncoder(tokenizer: tokenizer, url: textEncoderURL, config: config)

        // Unet model
        let unet: Unet
        if FileManager.default.fileExists(atPath: unetChunk1URL.path) &&
            FileManager.default.fileExists(atPath: unetChunk2URL.path) {
            unet = Unet(chunkUrls: [unetChunk1URL, unetChunk2URL], config: config)
        } else {
            unet = Unet(url: unetURL, config: config)
        }
        
        // Optional image encoder
        var encoder: Encoder? = nil
        if FileManager.default.fileExists(atPath: encoderURL.path) {
            let encoderModel = try MLModel(contentsOf: encoderURL, configuration: config)
            encoder = Encoder(model: encoderModel)
        }
        
        // Image Decoder
        let decoderModel = try MLModel(contentsOf: decoderURL, configuration: config)
        let decoder = Decoder(model: decoderModel)

        // Optional safety checker
        var safetyChecker: SafetyChecker? = nil
        if !disableSafety &&
            FileManager.default.fileExists(atPath: safetyCheckerURL.path) {
            let checkerModel = try MLModel(contentsOf: safetyCheckerURL, configuration: config)
            safetyChecker = SafetyChecker(model: checkerModel)
        }

        // Construct pipeline
        self.init(textEncoder: textEncoder,
                  unet: unet,
                  encoder: encoder,
                  decoder: decoder,
                  safetyChecker: safetyChecker)
    }
}
