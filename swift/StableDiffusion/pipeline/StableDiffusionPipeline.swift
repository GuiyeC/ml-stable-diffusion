// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML
import Accelerate
import CoreGraphics

/// Schedulers compatible with StableDiffusionPipeline
public enum StableDiffusionScheduler {
    /// Scheduler that uses a pseudo-linear multi-step (PLMS) method
    case pndmScheduler
    /// Scheduler that uses a second order DPM-Solver++ algorithm
    case dpmSolverMultistepScheduler
}

/// A pipeline used to generate image samples from text input using stable diffusion
///
/// This implementation matches:
/// [Hugging Face Diffusers Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)
@available(iOS 16.2, macOS 13.1, *)
public class StableDiffusionPipeline: ResourceManaging {

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

    /// Option to reduce memory during image generation
    ///
    /// If true, the pipeline will lazily load TextEncoder, Unet, Decoder, and SafetyChecker
    /// when needed and aggressively unload their resources after
    ///
    /// This will increase latency in favor of reducing memory
    var reduceMemory: Bool = false

    /// Reports whether this pipeline can perform safety checks
    public var canSafetyCheck: Bool {
        safetyChecker != nil
    }
    
    /// Reports whether this pipeline can perform image to image
    public var canGenerateVariations: Bool {
        encoder != nil
    }
    
    /// Reports whether this pipeline can perform image to image
    public var canInpaint: Bool = false
    
    public var takesInstructions: Bool = false
    
    /// Expected encoder input size
    public var expectedInputSize: CGSize?

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
        safetyChecker: SafetyChecker? = nil,
        reduceMemory: Bool = false
    ) {
        self.textEncoder = textEncoder
        self.unet = unet
        self.encoder = encoder
        self.decoder = decoder
        self.safetyChecker = safetyChecker
        self.reduceMemory = reduceMemory
    }

    /// Load required resources for this pipeline
    ///
    /// If reducedMemory is true this will instead call prewarmResources instead
    /// and let the pipeline lazily load resources as needed
    public func loadResources() throws {
        if reduceMemory {
            try prewarmResources()
        } else {
            try textEncoder.loadResources()
            try encoder?.loadResources()
            expectedInputSize = encoder?.expectedInputSize
            try unet.loadResources()
            canInpaint = unet.canInpaint && encoder != nil
            takesInstructions = unet.takesInstructions && encoder != nil
            try decoder.loadResources()
            try safetyChecker?.loadResources()
        }
    }

    /// Unload the underlying resources to free up memory
    public func unloadResources() {
        textEncoder.unloadResources()
        encoder?.unloadResources()
        unet.unloadResources()
        decoder.unloadResources()
        safetyChecker?.unloadResources()
    }

    // Prewarm resources one at a time
    public func prewarmResources() throws {
        try textEncoder.prewarmResources()
        expectedInputSize = try encoder?.prewarmResources()
        let (canInpaint, takesInstructions) = try unet.prewarmResources()
        self.canInpaint = canInpaint && encoder != nil
        self.takesInstructions = takesInstructions && encoder != nil
        try decoder.prewarmResources()
        try safetyChecker?.prewarmResources()
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
        let hiddenStates = try hiddenStates(from: input)

        /// Setup schedulers
        let scheduler: [Scheduler] = (0..<imageCount).map { _ in
            switch input.scheduler {
            case .pndmScheduler: return PNDMScheduler(strength: input.strength, stepCount: input.stepCount)
            case .dpmSolverMultistepScheduler: return DPMSolverMultistepScheduler(strength: input.strength, stepCount: input.stepCount)
            }
        }

        // Generate random latent samples from specified seed
        var random = NumPyRandomSource(seed: input.seed)
        var latents = try generateLatentSamples(imageCount, input: input, random: &random, scheduler: scheduler[0])
        let imageLatent = try generateImageLatent(imageCount, input: input, random: &random, scheduler: scheduler[0])
        // Prepare mask only for inpainting
        let inpaintingLatents = try prepareMaskLatents(input: input, random: &random)

        // De-noising loop
        for (step,t) in scheduler[0].timeSteps.enumerated() {
            // Expand the latents for classifier-free guidance
            // and input to the Unet noise prediction model
            var latentUnetInput: [MLShapedArray<Float32>]
            if let imageLatent {
                latentUnetInput = latents.map {
                    MLShapedArray<Float32>(concatenating: [$0, $0, $0], alongAxis: 0)
                }
                latentUnetInput = latentUnetInput.map {
                    MLShapedArray<Float32>(concatenating: [$0, imageLatent], alongAxis: 1)
                }
            } else {
                latentUnetInput = latents.map {
                    MLShapedArray<Float32>(concatenating: [$0, $0], alongAxis: 0)
                }
                // Concat mask in case we are doing inpainting
                if let (mask, maskedImage) = inpaintingLatents {
                    latentUnetInput = latentUnetInput.map {
                        MLShapedArray<Float32>(concatenating: [$0, mask, maskedImage], alongAxis: 1)
                    }
                }
            }

            // Predict noise residuals from latent samples
            // and current time step conditioned on hidden states
            var noise = try unet.predictNoise(
                latents: latentUnetInput,
                timeStep: t,
                hiddenStates: hiddenStates
            )

            noise = performGuidance(noise, guidanceScale: input.guidanceScale, imageGuidanceScale: input.imageGuidanceScale)

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
        // Latent shape for inpainting or pix2pix
        if input.inpaintMask != nil || input.imageGuidanceScale != nil {
            sampleShape[1] = 4
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
    
    func generateImageLatent(_ count: Int, input: SampleInput, random: inout NumPyRandomSource, scheduler: Scheduler) throws -> MLShapedArray<Float32>? {
        guard let image = input.initImage, input.imageGuidanceScale != nil else { return nil }
        guard let encoder = encoder else {
            fatalError("Tried to generate image variations without an Encoder")
        }
        let latent = try encoder.encode(image, scaleFactor: 1, random: { mean, std in
            Float32(random.nextNormal(mean: Double(mean), stdev: Double(std)))
        })
        let zeroLatent = MLShapedArray<Float32>(repeating: 0, shape: latent.shape)
        return MLShapedArray<Float32>(concatenating: [latent, latent, zeroLatent], alongAxis: 0)
    }
    
    func prepareMaskLatents(input: SampleInput, random: inout NumPyRandomSource) throws -> (MLShapedArray<Float32>, MLShapedArray<Float32>)? {
        guard let image = input.initImage, let mask = input.inpaintMask else { return nil }
        guard let encoder = encoder else {
            fatalError("Tried to inpaint without an Encoder")
        }
        var imageData = encoder.fromRGBCGImage(image)
        // This is reversed because: image * (mask < 0.5)
        let maskDataScalars = encoder.alphaFromRGBCGImage(mask).scalars.map { 1 - $0 }
        let maskedImageScalars = imageData.scalars.enumerated().map { index, value in
            value * maskDataScalars[index % maskDataScalars.count]
        }
        imageData = MLShapedArray<Float32>(scalars: maskedImageScalars, shape: imageData.shape)
        
        // Encode the mask image into latents space so we can concatenate it to the latents
        var maskedImageLatent = try encoder.encode(imageData, random: { mean, std in
            Float32(random.nextNormal(mean: Double(mean), stdev: Double(std)))
        })
        
        let resizedMask = encoder.resizeImage(mask, size: CGSize(width: maskedImageLatent.shape[3], height: maskedImageLatent.shape[2]))
        var maskData = encoder.alphaFromRGBCGImage(resizedMask)
        
        // Expand the latents for classifier-free guidance
        // and input to the Unet noise prediction model
        maskData = MLShapedArray<Float32>(concatenating: [maskData, maskData], alongAxis: 0)
        maskedImageLatent = MLShapedArray<Float32>(concatenating: [maskedImageLatent, maskedImageLatent], alongAxis: 0)
        
        return (maskData, maskedImageLatent)
    }

    private var lastInput: SampleInput?
    private var lastHiddenStates: MLShapedArray<Float32>?
    func hiddenStates(from input: SampleInput) throws -> MLShapedArray<Float32> {
        if lastInput?.prompt == input.prompt, lastInput?.negativePrompt == input.negativePrompt, let lastHiddenStates {
            return lastHiddenStates
        }
        // Encode the input prompt as well as a blank unconditioned input
        let promptEmbedding = try textEncoder.encode(input.prompt)
        let blankEmbedding = try textEncoder.encode(input.negativePrompt)
        
        if reduceMemory {
            textEncoder.unloadResources()
        }

        // Convert to Unet hidden state representation
        let concatEmbedding: MLShapedArray<Float32>
        if input.imageGuidanceScale != nil {
            // pix2pix has two negative embeddings, and unlike in other pipelines latents are ordered [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
            concatEmbedding = MLShapedArray<Float32>(
                concatenating: [promptEmbedding, blankEmbedding, blankEmbedding],
                alongAxis: 0
            )
        } else {
            concatEmbedding = MLShapedArray<Float32>(
                concatenating: [blankEmbedding, promptEmbedding],
                alongAxis: 0
            )
        }
        let hiddenStates = toHiddenStates(concatEmbedding)
        
        lastInput = input
        lastHiddenStates = hiddenStates
        
        return hiddenStates
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

    func performGuidance(_ noise: [MLShapedArray<Float32>], guidanceScale: Float, imageGuidanceScale: Float?) -> [MLShapedArray<Float32>] {
        noise.map { performGuidance($0, guidanceScale: guidanceScale, imageGuidanceScale: imageGuidanceScale) }
    }

    func performGuidance(_ noise: MLShapedArray<Float32>, guidanceScale: Float, imageGuidanceScale: Float?) -> MLShapedArray<Float32> {
        var resultScalars: [Float]
        if let imageGuidanceScale {
            let textNoiseScalars = noise[0].scalars
            let imageNoiseScalars = noise[1].scalars
            let blankNoiseScalars = noise[2].scalars

            resultScalars = blankNoiseScalars

            for i in 0..<resultScalars.count {
                // unconditioned + guidance*(text - image) + imageGuidance*(image - unconditioned)
                resultScalars[i] += (
                    guidanceScale*(textNoiseScalars[i]-imageNoiseScalars[i]) +
                    imageGuidanceScale*(imageNoiseScalars[i]-blankNoiseScalars[i])
                )
            }
        } else {
            let blankNoiseScalars = noise[0].scalars
            let textNoiseScalars = noise[1].scalars

            resultScalars = blankNoiseScalars

            for i in 0..<resultScalars.count {
                // unconditioned + guidance*(text - unconditioned)
                resultScalars[i] += guidanceScale*(textNoiseScalars[i]-blankNoiseScalars[i])
            }
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

@available(iOS 16.2, macOS 13.1, *)
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
            do {
                return try pipeline.decodeToImages(currentLatentSamples, disableSafety: !isSafetyEnabled)
            } catch {
                print("Error decoding progress images", error.localizedDescription)
                return []
            }
        }
    }
}
