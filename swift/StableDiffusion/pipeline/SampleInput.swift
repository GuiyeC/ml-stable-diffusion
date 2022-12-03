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
    public var initImage: CGImage?
    public var strength: Float?
    public var seed: Int
    public var stepCount: Int
    /// Controls the influence of the text prompt on sampling process (0=random images)
    public var guidanceScale: Float
    
    public init(
        prompt: String,
        seed: Int = Int.random(in: 0...Int(UInt32.max)),
        stepCount: Int = 20,
        guidanceScale: Float = 7.5
    ) {
        self.prompt = prompt
        self.initImage = nil
        self.strength = nil
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
    }
    
    public init(
        prompt: String,
        initImage: CGImage?,
        strength: Float = 0.75,
        seed: Int = Int.random(in: 0...Int(UInt32.max)),
        stepCount: Int = 20,
        guidanceScale: Float = 5.0
    ) {
        self.prompt = prompt
        self.initImage = initImage
        self.strength = strength
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
        initImage: UIImage,
        strength: Float = 0.75,
        seed: Int = Int.random(in: 0...Int(UInt32.max)),
        stepCount: Int = 50,
        guidanceScale: Float = 5.0
    ) {
        self.prompt = prompt
        self.initImage = initImage.cgImage!
        self.strength = strength
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
        initImage: NSImage,
        strength: Float = 0.75,
        seed: Int = Int.random(in: 0...Int(UInt32.max)),
        stepCount: Int = 50,
        guidanceScale: Float = 5.0
    ) {
        self.prompt = prompt
        var imageRect = CGRect(x: 0, y: 0, width: initImage.size.width, height: initImage.size.height)
        self.initImage = initImage.cgImage(forProposedRect: &imageRect, context: nil, hints: nil)!
        self.strength = strength
        self.seed = seed
        self.stepCount = stepCount
        self.guidanceScale = guidanceScale
    }
}
#endif
