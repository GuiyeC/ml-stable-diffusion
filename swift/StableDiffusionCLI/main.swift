// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import ArgumentParser
import CoreGraphics
import CoreML
import Foundation
import StableDiffusion
import UniformTypeIdentifiers

@available(iOS 16.2, macOS 13.1, *)
struct StableDiffusionSample: ParsableCommand {

    static let configuration = CommandConfiguration(
        abstract: "Run stable diffusion to generate images guided by a text prompt",
        version: "0.1"
    )

    @Argument(help: "Input string prompt")
    var prompt: String = ""
    
    @Option(
        help: ArgumentHelp(
            "Path to stable diffusion resources.",
            discussion: "The resource directory should contain\n" +
                " - *compiled* models: {TextEncoder,Unet,VAEDecoder}.mlmodelc\n" +
                " - tokenizer info: vocab.json, merges.txt",
            valueName: "directory-path"
        )
    )
    var resourcePath: String = "./model/"

    @Option(help: "Number of images to sample / generate")
    var imageCount: Int = 1
    
    @Option(help: "Number of diffusion steps to perform")
    var stepCount: Int = 70
    
    @Option(help: "Indicates how much to transform the reference `image`. Must be between 0 and 1.")
    var strength: Float = 0.75

    @Option(
        help: ArgumentHelp(
            "How often to save samples at intermediate steps",
            discussion: "Set to 0 to only save the final sample"
        )
    )
    var saveEvery: Int = 0

    @Option(help: "Output path")
    var outputPath: String = "./"

    @Option(help: "Random seed")
    var seed: Int = Int.random(in: (0..<Int(UInt32.max)))

    @Option(help: "Compute units to load model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine}")
    var computeUnits: ComputeUnits = .all

    @Flag(help: "Disable safety checking")
    var disableSafety: Bool = false

    mutating func run() throws {
        guard FileManager.default.fileExists(atPath: resourcePath) else {
            throw RunError.resources("Resource path does not exist \(resourcePath)")
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits.asMLComputeUnits
        let resourceURL = URL(filePath: resourcePath)

        log("Loading resources and creating pipeline\n")
        log("(Note: This can take a while the first time using these resources)\n")
        let pipeline = try StableDiffusionPipeline(resourcesAt: resourceURL,
                                                   configuration: config,
                                                   disableSafety: disableSafety)

        log("Sampling ...\n")
        let sampleTimer = SampleTimer()
        sampleTimer.start()
        
        let input = SampleInput(
            prompt: prompt,
            seed: seed,
            stepCount: stepCount
        )
        let images = try pipeline.generateImages(
            input: input,
            imageCount: imageCount
        ) { progress in
            sampleTimer.stop()
            handleProgress(progress, sampleTimer, input: input)
            if progress.stepCount != progress.step {
                sampleTimer.start()
            }
            return true
        }
        
        _ = try saveImages(images, input: input, logNames: true)
    }

    func handleProgress(
        _ progress: StableDiffusionPipeline.Progress,
        _ sampleTimer: SampleTimer,
        input: SampleInput
    ) {
        log("Step \(progress.step) of \(progress.stepCount) ")
        log(" [")
        log(String(format: "mean: %.2f, ", 1.0/sampleTimer.mean))
        log(String(format: "median: %.2f, ", 1.0/sampleTimer.median))
        log(String(format: "last %.2f", 1.0/sampleTimer.allSamples.last!))
        log("] step/sec")

        if saveEvery > 0, progress.step % saveEvery == 0 {
            let saveCount = (try? saveImages(progress.currentImages, input: input, step: progress.step)) ?? 0
            log(" saved \(saveCount) image\(saveCount != 1 ? "s" : "")")
        }
        log("\n")
    }

    func saveImages(
        _ images: [CGImage?],
        input: SampleInput,
        step: Int? = nil,
        logNames: Bool = false
    ) throws -> Int {
        let url = URL(filePath: outputPath)
        let lastFile = try FileManager.default.contentsOfDirectory(atPath: outputPath)
            .sorted()
            .last?
            .prefix(while: { $0 != "." }) ?? "-1"
        let newFile = Int(lastFile).map { $0 + 1 } ?? 0
        var saved = 0
        for i in 0 ..< images.count {

            guard let image = images[i] else {
                if logNames {
                    log("Image \(i) failed safety check and was not saved")
                }
                continue
            }
            
            let name = imageName(i, step: step)
            var fileURL = url.appending(path: String(format: "%05d", newFile + saved))
            if let step = step {
                fileURL.append(path: ".\(step)")
            }
            fileURL.appendPathExtension("png")

            guard let dest = CGImageDestinationCreateWithURL(fileURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
                throw RunError.saving("Failed to create destination for \(fileURL)")
            }
            let metadata = [
                kCGImagePropertyPNGDictionary: [
                    kCGImagePropertyPNGTitle: NSString(string: input.prompt),
                    kCGImagePropertyPNGDescription:
                        NSString(string: """
                            Seed: \(input.seed)
                            Steps: \(input.stepCount)
                            Strength: \(input.strength?.description ?? "-")
                            Guidance scale: \(input.guidanceScale)
                            """),
                    kCGImagePropertyPNGSoftware: NSString(string: "\(input.seed)\n\n\(input.prompt)"),
                ] as CFDictionary
            ] as CFDictionary
            CGImageDestinationAddImage(dest, image, metadata)
            if !CGImageDestinationFinalize(dest) {
                throw RunError.saving("Failed to save \(fileURL)")
            }
            if logNames {
                log("Saved \(name)\n")
            }
            saved += 1
        }
        return saved
    }

    func imageName(_ sample: Int, step: Int? = nil) -> String {
        let fileCharLimit = 75
        var name = prompt.prefix(fileCharLimit).replacingOccurrences(of: " ", with: "_")
        if imageCount != 1 {
            name += ".\(sample)"
        }

        name += ".\(seed)"

        if let step = step {
            name += ".\(step)"
        } else {
            name += ".final"
        }
        name += ".png"
        return name
    }

    func log(_ str: String, term: String = "") {
        print(str, terminator: term)
    }
}

enum RunError: Error {
    case resources(String)
    case saving(String)
}

@available(iOS 16.2, macOS 13.1, *)
enum ComputeUnits: String, ExpressibleByArgument, CaseIterable {
    case all, cpuAndGPU, cpuOnly, cpuAndNeuralEngine
    var asMLComputeUnits: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuOnly: return .cpuOnly
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        }
    }
}

@available(iOS 16.2, macOS 13.1, *)
enum SchedulerOption: String, ExpressibleByArgument {
    case pndm, dpmpp
    var stableDiffusionScheduler: StableDiffusionScheduler {
        switch self {
        case .pndm: return .pndmScheduler
        case .dpmpp: return .dpmSolverMultistepScheduler
        }
    }
}

if #available(iOS 16.2, macOS 13.1, *) {
    StableDiffusionSample.main()
} else {
    print("Unsupported OS")
}
