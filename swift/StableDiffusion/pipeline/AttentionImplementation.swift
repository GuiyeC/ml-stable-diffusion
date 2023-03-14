//
//  AttentionImplementation.swift
//  
//
//  Created by Guillermo Cique Fernández on 14/3/23.
//

import Foundation

public enum AttentionImplementation: String, Decodable, CustomStringConvertible {
    case original = "ORIGINAL"
    case splitEinsum = "SPLIT_EINSUM"
    
    public var description: String {
        switch self {
        case .original: return "Original"
        case .splitEinsum: return "Split einsum"
        }
    }
}
