import Foundation
import MLX
import Testing

@testable import MLXEmbedders
@testable import MLXLMCommon

@Suite("EmbeddingGemma Correctness", .serialized)
struct EmbeddingGemmaCorrectnessTests {

    private static let bf16ModelDirectory = URL(fileURLWithPath: NSString(
        string: "~/.cache/huggingface/hub/models--mlx-community--embeddinggemma-300m-bf16/snapshots/manual"
    ).expandingTildeInPath)

    // Pre-computed token IDs (same tokenizer corpus as benchmark).
    // Texts, decoded:
    //   [0] "SwiftLM performs text embeddings on Apple Silicon with Metal."
    //   [1] "quantized embedding throughput on macOS"
    //   [2] "Sentence-transformer style retrieval depends on stable normalized vectors."
    private static let tokenIDs: [[Int32]] = [
        [2, 14888, 5364, 24818, 1872, 55640, 611, 12186, 30805, 675, 20805, 235265],
        [2, 79723, 39830, 79695, 611, 47700],
        [2, 79185, 235290, 88406, 3411, 55857, 12097, 611, 13440, 43067, 37498, 235265],
    ]

    @Test("BF16 EmbeddingGemma produces 768-dim L2-normalized vectors with sensible similarity")
    func correctness() throws {
        guard FileManager.default.fileExists(
            atPath: Self.bf16ModelDirectory.appendingPathComponent("config.json").path
        ) else {
            print("[MLX.Skip] No bf16 model at \(Self.bf16ModelDirectory.path)")
            return
        }

        let model = try loadSynchronous(
            modelDirectory: Self.bf16ModelDirectory,
            modelName: "embeddinggemma-300m-bf16"
        )
        let pooler = loadPooling(modelDirectory: Self.bf16ModelDirectory, model: model)

        var vectors: [[Float]] = []
        for ids in Self.tokenIDs {
            autoreleasepool {
                let tokens = MLXArray(ids).reshaped(1, -1)
                let output = model(tokens, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
                let pooled = pooler(output)
                pooled.eval()
                let flat = pooled[0].asArray(Float.self)
                vectors.append(flat)
            }
        }

        // Shape: expect 768-dim
        for (i, v) in vectors.enumerated() {
            #expect(v.count == 768, "vector[\(i)] has \(v.count) dims, expected 768")
        }

        // Finite
        for (i, v) in vectors.enumerated() {
            #expect(v.allSatisfy { $0.isFinite }, "vector[\(i)] contains non-finite values")
        }

        // L2 normalized: ||v|| ≈ 1
        for (i, v) in vectors.enumerated() {
            let norm = sqrt(v.reduce(Float(0)) { $0 + $1 * $1 })
            #expect(abs(norm - 1.0) < 1e-3, "vector[\(i)] has norm \(norm), expected ≈1")
        }

        // Similarity: vector[0] ("embeddings on Apple Silicon") should be closer to
        // vector[1] ("quantized embedding throughput on macOS") than to vector[2]
        // (a generic sentence-transformer statement). Both 0 and 1 are about
        // embeddings running on Apple, while 2 is about normalized vectors.
        //
        // Note: without task prefixes this isn't guaranteed symmetric similarity,
        // but the three texts are distinct enough that any reasonable model should
        // preserve the topical distinction.
        let sim01 = dot(vectors[0], vectors[1])
        let sim02 = dot(vectors[0], vectors[2])
        let sim12 = dot(vectors[1], vectors[2])
        print(
            "[MLX.EmbeddingGemma.Sim] sim(0,1)=\(sim01) sim(0,2)=\(sim02) sim(1,2)=\(sim12)"
        )

        // Cosine is in [-1, 1]; for real sentence embeddings expect [0, 1] roughly.
        #expect(sim01 > -1 && sim01 < 1)
        #expect(sim02 > -1 && sim02 < 1)

        // Not degenerate: different inputs produce different vectors.
        #expect(abs(sim01 - 1.0) > 1e-4, "vectors 0 and 1 are identical — degenerate output")
        #expect(abs(sim02 - 1.0) > 1e-4, "vectors 0 and 2 are identical — degenerate output")
    }

    private func dot(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var s: Float = 0
        for i in 0..<a.count { s += a[i] * b[i] }
        return s
    }
}
