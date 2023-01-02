//
//  SampleInput.swift
//
//
//  Created by Guillermo Cique Fernández on 14/11/22.
//

import Foundation
import CoreGraphics

public struct SampleInput: Hashable {
    public var prompt: String
    public var negativePrompt: String
    public var initImage: CGImage?
    public var strength: Float?
    public var inpaintMask: CGImage?
    public var seed: UInt32
    public var stepCount: Int
    /// Controls the influence of the text prompt on sampling process (0=random images)
    public var guidanceScale: Float
    
    public init(
        prompt: String,
        negativePrompt: String = "",
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        guidanceScale: Float = 7.5
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = nil
        self.strength = nil
        self.inpaintMask = nil
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
    }
    
    public init(
        prompt: String,
        negativePrompt: String = "",
        initImage: CGImage?,
        strength: Float = 0.75,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        guidanceScale: Float = 7.5
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage
        self.strength = strength
        self.inpaintMask = nil
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
    }
    
    public init(
        prompt: String,
        negativePrompt: String = "",
        initImage: CGImage?,
        inpaintMask: CGImage,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 20,
        guidanceScale: Float = 7.5
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage
        self.strength = nil
        self.inpaintMask = inpaintMask
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
    }
}

#if os(iOS)
import UIKit

public extension SampleInput {
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: UIImage,
        strength: Float = 0.75,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 50,
        guidanceScale: Float = 7.5
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage.cgImage!
        self.strength = strength
        self.inpaintMask = nil
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
    }
    
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: UIImage,
        inpaintMask: CGImage,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 50,
        guidanceScale: Float = 7.5
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage.cgImage!
        self.strength = nil
        self.inpaintMask = inpaintMask
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
    }
    
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: UIImage,
        inpaintMask: UIImage,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 50,
        guidanceScale: Float = 7.5
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage.cgImage!
        self.strength = nil
        self.inpaintMask = inpaintMask.cgImage!
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
    }
}
#endif

#if os(macOS)
import AppKit

public extension SampleInput {
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: NSImage,
        strength: Float = 0.75,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 50,
        guidanceScale: Float = 7.5
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        var imageRect = CGRect(x: 0, y: 0, width: initImage.size.width, height: initImage.size.height)
        self.initImage = initImage.cgImage(forProposedRect: &imageRect, context: nil, hints: nil)!
        self.strength = strength
        self.inpaintMask = nil
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
    }
    
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: NSImage,
        inpaintMask: CGImage,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 50,
        guidanceScale: Float = 7.5
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        var imageRect = CGRect(x: 0, y: 0, width: initImage.size.width, height: initImage.size.height)
        self.initImage = initImage.cgImage(forProposedRect: &imageRect, context: nil, hints: nil)!
        self.strength = nil
        self.inpaintMask = inpaintMask
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
    }
    
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: NSImage,
        inpaintMask: NSImage,
        seed: UInt32 = UInt32.random(in: 0...UInt32.max),
        stepCount: Int = 50,
        guidanceScale: Float = 7.5
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        var imageRect = CGRect(x: 0, y: 0, width: initImage.size.width, height: initImage.size.height)
        self.initImage = initImage.cgImage(forProposedRect: &imageRect, context: nil, hints: nil)!
        self.strength = nil
        var inpaintMaskRect = CGRect(x: 0, y: 0, width: inpaintMask.size.width, height: inpaintMask.size.height)
        self.inpaintMask = inpaintMask.cgImage(forProposedRect: &inpaintMaskRect, context: nil, hints: nil)!
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
    }
}
#endif
