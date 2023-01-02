// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import CoreML

@available(iOS 16.2, macOS 13.1, *)
public protocol Scheduler {
    /// Number of diffusion steps performed during training
    var trainStepCount: Int { get }

    /// Number of inference steps to be performed
    var inferenceStepCount: Int { get }

    /// Training diffusion time steps index by inference time step
    var timeSteps: [Int] { get }

    /// Schedule of betas which controls the amount of noise added at each timestep
    var betas: [Float] { get }

    /// 1 - betas
    var alphas: [Float] { get }

    /// Cached cumulative product of alphas
    var alphasCumProd: [Float] { get }

    /// Standard deviation of the initial noise distribution
    var initNoiseSigma: Float { get }

    /// Compute a de-noised image sample and step scheduler state
    ///
    /// - Parameters:
    ///   - output: The predicted residual noise output of learned diffusion model
    ///   - timeStep: The current time step in the diffusion chain
    ///   - sample: The current input sample to the diffusion model
    /// - Returns: Predicted de-noised sample at the previous time step
    /// - Postcondition: The scheduler state is updated.
    ///   The state holds the current sample and history of model output noise residuals
    func step(
        output: MLShapedArray<Float32>,
        timeStep t: Int,
        sample s: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32>
}

@available(iOS 16.2, macOS 13.1, *)
public extension Scheduler {
    var initNoiseSigma: Float { 1 }
}

@available(iOS 16.2, macOS 13.1, *)
public extension Scheduler {
    /// Compute weighted sum of shaped arrays of equal shapes
    ///
    /// - Parameters:
    ///   - weights: The weights each array is multiplied by
    ///   - values: The arrays to be weighted and summed
    /// - Returns: sum_i weights[i]*values[i]
    func weightedSum(_ weights: [Double], _ values: [MLShapedArray<Float32>]) -> MLShapedArray<Float32> {
        assert(weights.count > 1 && values.count == weights.count)
        assert(values.allSatisfy({ $0.scalarCount == values.first!.scalarCount }))
        var w = Float(weights.first!)
        var scalars = values.first!.scalars.map({ $0 * w })
        for next in 1 ..< values.count {
            w = Float(weights[next])
            let nextScalars = values[next].scalars
            for i in 0 ..< scalars.count {
                scalars[i] += w * nextScalars[i]
            }
        }
        return MLShapedArray(scalars: scalars, shape: values.first!.shape)
    }
    
    func addNoise(
        originalSample: MLShapedArray<Float32>,
        noise: [MLShapedArray<Float32>]
    ) -> [MLShapedArray<Float32>] {
        let alphaProdt = alphasCumProd[timeSteps[0]]
        let betaProdt = 1 - alphaProdt
        let sqrtAlphaProdt = sqrt(alphaProdt)
        let sqrtBetaProdt = sqrt(betaProdt)
        
        let noisySamples = noise.map {
            weightedSum(
                [Double(sqrtAlphaProdt), Double(sqrtBetaProdt)],
                [originalSample, $0]
            )
        }

        return noisySamples
    }
}

/// How to map a beta range to a sequence of betas to step over
@available(iOS 16.2, macOS 13.1, *)
public enum BetaSchedule {
    /// Linear stepping between start and end
    case linear
    /// Steps using linspace(sqrt(start),sqrt(end))^2
    case scaledLinear
}

/// Evenly spaced floats between specified interval
///
/// - Parameters:
///   - start: Start of the interval
///   - end: End of the interval
///   - count: The number of floats to return between [*start*, *end*]
/// - Returns: Float array with *count* elements evenly spaced between at *start* and *end*
func linspace(_ start: Float, _ end: Float, _ count: Int) -> [Float] {
    let scale = (end - start) / Float(count - 1)
    return (0..<count).map { Float($0)*scale + start }
}

extension Collection {
    /// Collection element index from the back. *self[back: 1]* yields the last element
    public subscript(back i: Int) -> Element {
        return self[index(endIndex, offsetBy: -i)]
    }
}
