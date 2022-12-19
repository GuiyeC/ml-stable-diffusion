// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML
import Accelerate
import CoreGraphics

/// A pipeline used to generate image samples from text input using stable diffusion
///
/// This implementation matches:
/// [Hugging Face Diffusers Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)
public struct StableDiffusionPipeline {

    /// Model to generate embeddings for tokenized input text
    var textEncoder: TextEncoder

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    var unet: Unet
    
    /// Model used to generate initial image for latent diffusion process
    var encoder: Encoder? = nil
    
    /// Model used to generate final image from latent diffusion process
    var decoder: Decoder

    /// Optional model for checking safety of generated image
    var safetyChecker: SafetyChecker? = nil

    /// Reports whether this pipeline can perform safety checks
    public var canSafetyCheck: Bool {
        safetyChecker != nil
    }
    
    /// Reports whether this pipeline can perform image to image
    public var canGenerateVariations: Bool {
        encoder != nil
    }
    
    /// Reports whether this pipeline can perform image to image
    public var canInpaint: Bool {
        encoder != nil && unet.latentSampleShape[1] == 9
    }
    
    /// Expected encoder input size
    public var expectedInputSize: CGSize? {
        guard let encoder else { return nil }
        let width: Int = encoder.inputImageShape[3]
        let height: Int = encoder.inputImageShape[2]
        return CGSize(width: width, height: height)
    }

    /// Creates a pipeline using the specified models and tokenizer
    ///
    /// - Parameters:
    ///   - textEncoder: Model for encoding tokenized text
    ///   - unet: Model for noise prediction on latent samples
    ///   - decoder: Model for decoding latent sample to image
    ///   - safetyChecker: Optional model for checking safety of generated images
    ///   - guidanceScale: Influence of the text prompt on generation process
    /// - Returns: Pipeline ready for image generation
    public init(
        textEncoder: TextEncoder,
        unet: Unet,
        encoder: Encoder? = nil,
        decoder: Decoder,
        safetyChecker: SafetyChecker? = nil
    ) {
        self.textEncoder = textEncoder
        self.unet = unet
        self.encoder = encoder
        self.decoder = decoder
        self.safetyChecker = safetyChecker
    }
    
    /// Text to image generation using stable diffusion
    ///
    /// - Parameters:
    ///   - prompt: Text prompt to guide sampling
    ///   - stepCount: Number of inference steps to perform
    ///   - imageCount: Number of samples/images to generate for the input prompt
    ///   - seed: Random seed which
    ///   - disableSafety: Safety checks are only performed if `self.canSafetyCheck && !disableSafety`
    ///   - progressHandler: Callback to perform after each step, stops on receiving false response
    /// - Returns: An array of `imageCount` optional images.
    ///            The images will be nil if safety checks were performed and found the result to be un-safe
    public func generateImages(
        input: SampleInput,
        imageCount: Int = 1,
        disableSafety: Bool = false,
        progressHandler: (Progress) -> Bool = { _ in true }
    ) throws -> [CGImage?] {
        let mainTick = CFAbsoluteTimeGetCurrent()
        // Encode the input prompt as well as a blank unconditioned input
        let promptEmbedding = try textEncoder.encode(input.prompt)
        let blankEmbedding = try textEncoder.encode(input.negativePrompt)

        // Convert to Unet hidden state representation
        let concatEmbedding = MLShapedArray<Float32>(
            concatenating: [blankEmbedding, promptEmbedding],
            alongAxis: 0
        )

        let hiddenStates = toHiddenStates(concatEmbedding)

        /// Setup schedulers
        let scheduler = (0..<imageCount).map { _ in Scheduler(strength: input.strength, stepCount: input.stepCount) }

        // Generate random latent samples from specified seed
        var random = NumPyRandomSource(seed: UInt32(input.seed))
        var latents = try generateLatentSamples(imageCount, input: input, random: &random, scheduler: scheduler[0])
        // Prepare mask only for inpainting
        let inpaintingLatents = try prepareMaskLatents(input: input, random: &random)

        // De-noising loop
        for (step,t) in scheduler[0].timeSteps.enumerated() {
            // Expand the latents for classifier-free guidance
            // and input to the Unet noise prediction model
            var latentUnetInput = latents.map {
                MLShapedArray<Float32>(concatenating: [$0, $0], alongAxis: 0)
            }
            
            // Concat mask in case we are doing inpainting
            if let (mask, maskedImage) = inpaintingLatents {
                latentUnetInput = latentUnetInput.map {
                    MLShapedArray<Float32>(concatenating: [$0, mask, maskedImage], alongAxis: 1)
                }
            }

            // Predict noise residuals from latent samples
            // and current time step conditioned on hidden states
            var noise = try unet.predictNoise(
                latents: latentUnetInput,
                timeStep: t,
                hiddenStates: hiddenStates
            )

            noise = performGuidance(noise, guidanceScale: input.guidanceScale)

            // Have the scheduler compute the previous (t-1) latent
            // sample given the predicted noise and current sample
            for i in 0..<imageCount {
                latents[i] = scheduler[i].step(
                    output: noise[i],
                    timeStep: t,
                    sample: latents[i]
                )
            }

            // Report progress
            let progress = Progress(
                pipeline: self,
                prompt: input.prompt,
                step: step,
                stepCount: scheduler[0].timeSteps.count - 1,
                currentLatentSamples: latents,
                isSafetyEnabled: canSafetyCheck && !disableSafety
            )
            if !progressHandler(progress) {
                // Stop if requested by handler
                return []
            }
        }

        // Decode the latent samples to images
        let images = try decodeToImages(latents, disableSafety: disableSafety)
        
        let mainTock = CFAbsoluteTimeGetCurrent()
        let runtime = String(format:"%.2fs", mainTock - mainTick)
        print("Time", runtime)
        
        return images
    }
    
    func generateLatentSamples(_ count: Int, input: SampleInput, random: inout NumPyRandomSource, scheduler: Scheduler) throws -> [MLShapedArray<Float32>] {
        var sampleShape = unet.latentSampleShape
        sampleShape[0] = 1
        // Latent shape for inpainting
        if input.inpaintMask != nil {
            sampleShape[1] -= 5
        }
        
        let stdev = scheduler.initNoiseSigma
        let samples = (0..<count).map { _ in
            MLShapedArray<Float32>(
                converting: random.normalShapedArray(sampleShape, mean: 0.0, stdev: Double(stdev)))
        }
        if let image = input.initImage, input.strength != nil {
            guard let encoder = encoder else {
                fatalError("Tried to generate image variations without an Encoder")
            }
            let latent = try encoder.encode(image, random: { mean, std in
                Float32(random.nextNormal(mean: Double(mean), stdev: Double(std)))
            })
            return scheduler.addNoise(originalSample: latent, noise: samples)
        }
        
        return samples
    }
    
    func prepareMaskLatents(input: SampleInput, random: inout NumPyRandomSource) throws -> (MLShapedArray<Float32>, MLShapedArray<Float32>)? {
        guard let image = input.initImage, let mask = input.inpaintMask else { return nil }
        guard let encoder = encoder else {
            fatalError("Tried to inpaint without an Encoder")
        }
        var imageData = encoder.fromRGBCGImage(image)
        var maskData = MLShapedArray<Float32>(converting: encoder.fromRGBCGImage(mask))
        if maskData.shape[1] > 1 {
            maskData = MLShapedArray<Float32>(
                scalars: maskData[0][0].scalars,
                shape: [1, 1, maskData.shape[2], maskData.shape[3]]
            )
        }
        // This is reversed because: image * (mask < 0.5)
        maskData = MLShapedArray<Float32>(
            scalars: maskData.scalars.map { $0 < 0 ? 0 : 1 },
            shape: maskData.shape
        )
        imageData = MLShapedArray<Float32>(
            scalars: zip(imageData.scalars, maskData.scalars + maskData.scalars + maskData.scalars).map { $0 * $1 },
            shape: imageData.shape
        )
        
        // Encode the mask image into latents space so we can concatenate it to the latents
        var maskedImageLatent = try encoder.encode(imageData, random: { mean, std in
            Float32(random.nextNormal(mean: Double(mean), stdev: Double(std)))
        })
        
        let resizedMask = encoder.resizeImage(mask, size: CGSize(width: maskedImageLatent.shape[3], height: maskedImageLatent.shape[2]))
        maskData = encoder.fromRGBCGImage(resizedMask)
        if maskData.shape[1] > 1 {
            maskData = MLShapedArray<Float32>(
                scalars: maskData[0][0].scalars,
                shape: [1, 1, maskData.shape[2], maskData.shape[3]]
            )
        }
        maskData = MLShapedArray<Float32>(
            scalars: maskData.scalars.map { $0 < 0 ? 1 : 0 },
            shape: maskData.shape
        )
//        maskData = MLShapedArray(scalars: maskData.scalars, shape: [1] + maskData.shape)
        
        // Expand the latents for classifier-free guidance
        // and input to the Unet noise prediction model
        maskData = MLShapedArray<Float32>(concatenating: [maskData, maskData], alongAxis: 0)
        maskedImageLatent = MLShapedArray<Float32>(concatenating: [maskedImageLatent, maskedImageLatent], alongAxis: 0)
        
        return (maskData, maskedImageLatent)
    }

    func toHiddenStates(_ embedding: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        // Unoptimized manual transpose [0, 2, None, 1]
        // e.g. From [2, 77, 768] to [2, 768, 1, 77]
        let fromShape = embedding.shape
        let stateShape = [fromShape[0],fromShape[2], 1, fromShape[1]]
        var states = MLShapedArray<Float32>(repeating: 0.0, shape: stateShape)
        for i0 in 0..<fromShape[0] {
            for i1 in 0..<fromShape[1] {
                for i2 in 0..<fromShape[2] {
                    states[scalarAt:i0,i2,0,i1] = embedding[scalarAt:i0, i1, i2]
                }
            }
        }
        return states
    }

    func performGuidance(_ noise: [MLShapedArray<Float32>], guidanceScale: Float) -> [MLShapedArray<Float32>] {
        noise.map { performGuidance($0, guidanceScale: guidanceScale) }
    }

    func performGuidance(_ noise: MLShapedArray<Float32>, guidanceScale: Float) -> MLShapedArray<Float32> {

        let blankNoiseScalars = noise[0].scalars
        let textNoiseScalars = noise[1].scalars

        var resultScalars = blankNoiseScalars

        for i in 0..<resultScalars.count {
            // unconditioned + guidance*(text - unconditioned)
            resultScalars[i] += guidanceScale*(textNoiseScalars[i]-blankNoiseScalars[i])
        }

        var shape = noise.shape
        shape[0] = 1
        return MLShapedArray<Float32>(scalars: resultScalars, shape: shape)
    }

    func decodeToImages(_ latents: [MLShapedArray<Float32>],
                        disableSafety: Bool) throws -> [CGImage?] {


        let images = try decoder.decode(latents)

        // If safety is disabled return what was decoded
        if disableSafety {
            return images
        }

        // If there is no safety checker return what was decoded
        guard let safetyChecker = safetyChecker else {
            return images
        }

        // Otherwise change images which are not safe to nil
        let safeImages = try images.map { image in
            try safetyChecker.isSafe(image) ? image : nil
        }

        return safeImages
    }

}

extension StableDiffusionPipeline {
    /// Sampling progress details
    public struct Progress {
        public let pipeline: StableDiffusionPipeline
        public let prompt: String
        public let step: Int
        public let stepCount: Int
        public let currentLatentSamples: [MLShapedArray<Float32>]
        public let isSafetyEnabled: Bool
        public var currentImages: [CGImage?] {
            try! pipeline.decodeToImages(
                currentLatentSamples,
                disableSafety: !isSafetyEnabled)
        }
    }
}
