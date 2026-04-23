import Foundation
import MLX
import Testing

@testable import MLXEmbedders
@testable import MLXLMCommon

@Suite("EmbeddingGemma MLX Benchmark", .serialized)
struct EmbeddingGemmaBenchmarkTests {

    private static let q4ModelDirectory = URL(fileURLWithPath: NSString(
        string: "~/.cache/huggingface/hub/models--mlx-community--embeddinggemma-300m-4bit/snapshots/5d9ef074df3957afc5c77127f208fddbc3c54187"
    ).expandingTildeInPath)

    private static let bf16ModelDirectory = URL(fileURLWithPath: NSString(
        string: "~/.cache/huggingface/hub/models--mlx-community--embeddinggemma-300m-bf16/snapshots/manual"
    ).expandingTildeInPath)

    // Pre-computed token IDs for benchmark texts (same as swift-lm benchmark).
    // These were produced by the EmbeddingGemma tokenizer (Gemma 3 vocab).
    // Using fixed tokens removes tokenizer from the measurement.
    private static let benchmarkTokenIDs: [[Int32]] = [
        // "SwiftLM performs text embeddings on Apple Silicon with Metal."
        [2, 14888, 5364, 24818, 1872, 55640, 611, 12186, 30805, 675, 20805, 235265],
        // "quantized embedding throughput on macOS"
        [2, 79723, 39830, 79695, 611, 47700],
        // "Sentence-transformer style retrieval depends on stable normalized vectors."
        [2, 79185, 235290, 88406, 3411, 55857, 12097, 611, 13440, 43067, 37498, 235265],
    ]

    private func benchmarkIterations() -> Int {
        let rawValue = ProcessInfo.processInfo.environment["MLX_EMBEDDING_BENCH_ITERATIONS"] ?? "2"
        guard let iterations = Int(rawValue), iterations > 0 else { return 2 }
        return iterations
    }

    @Test("Q4 EmbeddingGemma MLX throughput", .timeLimit(.minutes(10)))
    func q4Throughput() async throws {
        try measureThroughput(
            modelDirectory: Self.q4ModelDirectory,
            variant: "q4"
        )
    }

    @Test("BF16 EmbeddingGemma MLX throughput", .timeLimit(.minutes(10)))
    func bf16Throughput() async throws {
        try measureThroughput(
            modelDirectory: Self.bf16ModelDirectory,
            variant: "bf16"
        )
    }

    private func measureThroughput(modelDirectory: URL, variant: String) throws {
        guard FileManager.default.fileExists(
            atPath: modelDirectory.appendingPathComponent("config.json").path
        ) else {
            print("[MLX.Skip] No \(variant) model found at \(modelDirectory.path)")
            return
        }

        let model = try loadSynchronous(
            modelDirectory: modelDirectory,
            modelName: "embeddinggemma-300m-\(variant)"
        )
        let pooler = loadPooling(modelDirectory: modelDirectory, model: model)

        let tokenArrays = Self.benchmarkTokenIDs.map { ids in
            MLXArray(ids).reshaped(1, -1)
        }
        let iterations = benchmarkIterations()

        print("[MLX.EmbeddingGemma.Config] variant=\(variant) iterations=\(iterations) inputs=\(tokenArrays.count)")

        // Warmup
        for tokens in tokenArrays {
            autoreleasepool {
                let output = model(tokens, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
                let pooled = pooler(output)
                pooled.eval()
            }
        }

        let clock = ContinuousClock()
        var checksum: Float = 0
        let start = clock.now
        for _ in 0..<iterations {
            for tokens in tokenArrays {
                autoreleasepool {
                    let output = model(tokens, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
                    let pooled = pooler(output)
                    pooled.eval()
                    checksum += pooled[0, 0].item(Float.self)
                }
            }
        }
        let duration = clock.now - start

        let embeddingCount = iterations * tokenArrays.count
        let seconds = Double(duration.components.seconds)
            + (Double(duration.components.attoseconds) / 1_000_000_000_000_000_000)
        let embeddingsPerSecond = Double(embeddingCount) / seconds
        let secondsString = String(format: "%.3f", seconds)
        let throughputString = String(format: "%.3f", embeddingsPerSecond)
        let checksumString = String(format: "%.6f", checksum)
        print(
            "[MLX.EmbeddingGemma.Perf] variant=\(variant) "
                + "embeddings=\(embeddingCount) "
                + "seconds=\(secondsString) "
                + "embeddingsPerSecond=\(throughputString) "
                + "checksum=\(checksumString)"
        )

        #expect(checksum.isFinite)
        #expect(embeddingsPerSecond > 0)
    }
}
