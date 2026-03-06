// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - ToolCallParser Protocol

/// Protocol for parsing tool call content from model output.
///
/// Different models use different formats for tool calls. This protocol provides
/// a common interface for parsing tool calls from model output text.
///
/// Reference: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tool_parsers
public protocol ToolCallParser: Sendable {
    /// The start tag that indicates a tool call is beginning.
    /// Returns `nil` for inline formats that don't use wrapper tags.
    var startTag: String? { get }

    /// The end tag that indicates a tool call has ended.
    /// Returns `nil` for inline formats that don't use wrapper tags.
    var endTag: String? { get }

    /// Parse the content into a `ToolCall`.
    /// - Parameters:
    ///   - content: The text content to parse (may include tags)
    ///   - tools: Optional tool schemas for type-aware parsing
    /// - Returns: A `ToolCall` if parsing succeeds, `nil` otherwise
    func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall?
}

// MARK: - ToolCallFormatProvider Protocol

/// Provider that describes how a model family emits tool calls.
///
/// External packages can register custom providers instead of modifying MLXLMCommon.
public protocol ToolCallFormatProvider: Sendable {
    /// Stable serialization identifier for this format.
    var format: ToolCallFormat { get }

    /// Create a parser for the tool call format.
    func createParser() -> any ToolCallParser

    /// Returns true when this provider should be used for the given model type.
    func matches(modelType: String) -> Bool
}

public extension ToolCallFormatProvider {
    func matches(modelType _: String) -> Bool {
        false
    }
}

// MARK: - ToolCallFormat

/// Stable serialization identifier for a tool call format.
///
/// Built-in formats are exposed as static values, but callers can define and
/// register their own formats via ``ToolCallFormatProvider``.
public struct ToolCallFormat: RawRepresentable, Hashable, Sendable, Codable, CaseIterable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// Default JSON format used by Llama, Qwen, and most models.
    /// Example: `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
    public static let json = Self(rawValue: "json")

    /// LFM2/LFM2.5 Pythonic format with model-specific tags.
    /// Example: `<|tool_call_start|>[func(arg='value')]<|tool_call_end|>`
    public static let lfm2 = Self(rawValue: "lfm2")

    /// XML function format used by Qwen3.x families.
    /// Example: `<function=name><parameter=key>value</parameter></function>`
    public static let xmlFunction = Self(rawValue: "xml_function")

    /// GLM4 format with arg_key/arg_value tags.
    /// Example: `func<arg_key>k</arg_key><arg_value>v</arg_value>`
    public static let glm4 = Self(rawValue: "glm4")

    /// Gemma function call format.
    /// Example: `call:name{key:value,k:<escape>str<escape>}`
    public static let gemma = Self(rawValue: "gemma")

    /// Kimi K2 format with functions prefix.
    /// Example: `functions.name:0<|tool_call_argument_begin|>{"key": "value"}`
    public static let kimiK2 = Self(rawValue: "kimi_k2")

    /// MiniMax M2 format with invoke/parameter tags.
    /// Example: `<invoke name="f"><parameter name="k">v</parameter></invoke>`
    public static let minimaxM2 = Self(rawValue: "minimax_m2")

    public static var allCases: [ToolCallFormat] {
        ToolCallFormatRegistry.shared.registeredFormats
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.init(rawValue: try container.decode(String.self))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    /// Create the registered parser for this format.
    public func createParser() -> any ToolCallParser {
        ToolCallFormatRegistry.shared.createParser(for: self)
    }

    /// Infer the tool call format from a model type.
    public static func infer(from modelType: String) -> ToolCallFormat? {
        ToolCallFormatRegistry.shared.infer(from: modelType)
    }

    /// Register a custom tool call provider.
    ///
    /// If another provider already exists for the same `format`, it will be replaced.
    public static func register(_ provider: some ToolCallFormatProvider) {
        ToolCallFormatRegistry.shared.register(provider)
    }
}

// MARK: - Registry

public final class ToolCallFormatRegistry: @unchecked Sendable {
    public static let shared = ToolCallFormatRegistry()

    private let lock = NSLock()
    private var providersByFormat: [ToolCallFormat: any ToolCallFormatProvider] = [:]
    private var formatOrder: [ToolCallFormat] = []
    private var inferenceProviders: [any ToolCallFormatProvider] = []

    public var registeredFormats: [ToolCallFormat] {
        lock.withLock { formatOrder }
    }

    public func register(_ provider: some ToolCallFormatProvider) {
        let erased: any ToolCallFormatProvider = provider
        lock.withLock {
            if providersByFormat[provider.format] == nil {
                formatOrder.append(provider.format)
            }
            providersByFormat[provider.format] = erased
            inferenceProviders.removeAll { $0.format == provider.format }
            inferenceProviders.append(erased)
        }
    }

    public func createParser(for format: ToolCallFormat) -> any ToolCallParser {
        let provider = lock.withLock { providersByFormat[format] }
        if let provider {
            return provider.createParser()
        }

        #if DEBUG
        print("[MLXLMCommon] Unknown toolCallFormat=\(format.rawValue), falling back to json")
        #endif
        return JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
    }

    public func infer(from modelType: String) -> ToolCallFormat? {
        let normalizedType = modelType.lowercased()
        return lock.withLock {
            inferenceProviders.first { $0.matches(modelType: normalizedType) }?.format
        }
    }

    private init() {
        register(BuiltinJSONToolCallFormatProvider())
        register(BuiltinLFM2ToolCallFormatProvider())
        register(BuiltinXMLFunctionToolCallFormatProvider())
        register(BuiltinGLM4ToolCallFormatProvider())
        register(BuiltinGemmaToolCallFormatProvider())
        register(BuiltinKimiK2ToolCallFormatProvider())
        register(BuiltinMiniMaxM2ToolCallFormatProvider())
    }
}

// MARK: - Built-in Providers

private struct BuiltinJSONToolCallFormatProvider: ToolCallFormatProvider {
    let format: ToolCallFormat = .json

    func createParser() -> any ToolCallParser {
        JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
    }
}

private struct BuiltinLFM2ToolCallFormatProvider: ToolCallFormatProvider {
    let format: ToolCallFormat = .lfm2

    func createParser() -> any ToolCallParser {
        PythonicToolCallParser(startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
    }

    func matches(modelType: String) -> Bool {
        modelType.hasPrefix("lfm2")
    }
}

private struct BuiltinXMLFunctionToolCallFormatProvider: ToolCallFormatProvider {
    let format: ToolCallFormat = .xmlFunction

    func createParser() -> any ToolCallParser {
        XMLFunctionParser()
    }

    func matches(modelType: String) -> Bool {
        modelType.hasPrefix("qwen3_5")
    }
}

private struct BuiltinGLM4ToolCallFormatProvider: ToolCallFormatProvider {
    let format: ToolCallFormat = .glm4

    func createParser() -> any ToolCallParser {
        GLM4ToolCallParser()
    }

    func matches(modelType: String) -> Bool {
        modelType.hasPrefix("glm4")
    }
}

private struct BuiltinGemmaToolCallFormatProvider: ToolCallFormatProvider {
    let format: ToolCallFormat = .gemma

    func createParser() -> any ToolCallParser {
        GemmaFunctionParser()
    }

    func matches(modelType: String) -> Bool {
        modelType == "gemma"
    }
}

private struct BuiltinKimiK2ToolCallFormatProvider: ToolCallFormatProvider {
    let format: ToolCallFormat = .kimiK2

    func createParser() -> any ToolCallParser {
        KimiK2ToolCallParser()
    }
}

private struct BuiltinMiniMaxM2ToolCallFormatProvider: ToolCallFormatProvider {
    let format: ToolCallFormat = .minimaxM2

    func createParser() -> any ToolCallParser {
        MiniMaxM2ToolCallParser()
    }
}
