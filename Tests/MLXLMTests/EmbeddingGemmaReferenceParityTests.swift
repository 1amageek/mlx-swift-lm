import Foundation
import MLX
import Testing

@testable import MLXEmbedders
@testable import MLXLMCommon

/// Parity check against sentence-transformers reference vectors.
///
/// Loads a reference snapshot with 768-dim embeddings produced by
/// `SentenceTransformer("google/embeddinggemma-300m")` in Python, plus the
/// pre-computed token IDs for each input (tokenized with the Gemma tokenizer
/// shipped in `mlx-community/embeddinggemma-300m-bf16`, prefixed with the
/// query/document prompts from `config_sentence_transformers.json`).
///
/// Using pre-computed token IDs isolates the model under test: any cosine
/// miss is attributable to the MLX forward pass (pooling, dense projection,
/// normalization), not tokenizer divergence. This matches the isolation
/// strategy already used in `EmbeddingGemmaCorrectnessTests` and keeps the
/// parity test free of swift-transformers dependencies.
///
/// The threshold 0.99 is what swift-lm uses for the same EmbeddingGemma
/// parity test; it absorbs bf16 vs fp32 drift without masking a real model
/// regression (e.g. incorrect pooling axis, missing normalization, dense
/// head misapplied).
@Suite("EmbeddingGemma Reference Parity", .serialized)
struct EmbeddingGemmaReferenceParityTests {

    private static let bf16ModelDirectory = URL(fileURLWithPath: NSString(
        string: "~/.cache/huggingface/hub/models--mlx-community--embeddinggemma-300m-bf16/snapshots/manual"
    ).expandingTildeInPath)

    /// Minimum acceptable mean cosine between MLX and reference vectors.
    private static let minimumMeanCosine: Float = 0.99

    /// Minimum acceptable per-vector cosine. Looser than the mean to absorb
    /// one outlier from quantization on very short inputs without letting a
    /// systemic divergence slip through.
    private static let minimumPerVectorCosine: Float = 0.985

    @Test("MLX EmbeddingGemma matches sentence-transformers reference", .timeLimit(.minutes(10)))
    func referenceParity() throws {
        guard FileManager.default.fileExists(
            atPath: Self.bf16ModelDirectory.appendingPathComponent("config.json").path
        ) else {
            print("[MLX.Skip] No bf16 model at \(Self.bf16ModelDirectory.path)")
            return
        }

        let snapshot = try loadReferenceSnapshot()
        let model = try loadSynchronous(
            modelDirectory: Self.bf16ModelDirectory,
            modelName: "embeddinggemma-300m-bf16"
        )
        let pooler = loadPooling(modelDirectory: Self.bf16ModelDirectory, model: model)

        var documentCosines: [String: Float] = [:]
        documentCosines.reserveCapacity(snapshot.dataset.documents.count)
        for document in snapshot.dataset.documents {
            guard let reference = snapshot.documentEmbeddings[document.id] else {
                Issue.record("Missing reference embedding for document \(document.id)")
                continue
            }
            guard let tokenIDs = snapshot.documentTokens[document.id] else {
                Issue.record("Missing pre-computed tokens for document \(document.id)")
                continue
            }
            let embedding = embed(tokenIDs: tokenIDs, model: model, pooler: pooler)
            validateVector(embedding, id: "doc:\(document.id)", expectedDim: snapshot.embeddingDimension)
            documentCosines[document.id] = cosine(embedding, reference)
        }

        var queryCosines: [String: Float] = [:]
        queryCosines.reserveCapacity(snapshot.dataset.queries.count)
        for query in snapshot.dataset.queries {
            guard let reference = snapshot.queryEmbeddings[query.id] else {
                Issue.record("Missing reference embedding for query \(query.id)")
                continue
            }
            guard let tokenIDs = snapshot.queryTokens[query.id] else {
                Issue.record("Missing pre-computed tokens for query \(query.id)")
                continue
            }
            let embedding = embed(tokenIDs: tokenIDs, model: model, pooler: pooler)
            validateVector(embedding, id: "query:\(query.id)", expectedDim: snapshot.embeddingDimension)
            queryCosines[query.id] = cosine(embedding, reference)
        }

        let meanDocCos = mean(documentCosines.values)
        let meanQueryCos = mean(queryCosines.values)
        let minDocCos = documentCosines.values.min() ?? 0
        let minQueryCos = queryCosines.values.min() ?? 0

        print(
            "[MLX.EmbeddingGemma.Parity] dataset=\(snapshot.dataset.name) "
                + "docs=\(snapshot.dataset.documents.count) queries=\(snapshot.dataset.queries.count) "
                + "meanDocCos=\(String(format: "%.4f", meanDocCos)) "
                + "meanQueryCos=\(String(format: "%.4f", meanQueryCos)) "
                + "minDocCos=\(String(format: "%.4f", minDocCos)) "
                + "minQueryCos=\(String(format: "%.4f", minQueryCos))"
        )
        for (id, cos) in documentCosines.sorted(by: { $0.value < $1.value }).prefix(3) {
            print("[MLX.EmbeddingGemma.Parity.DocLow] id=\(id) cos=\(String(format: "%.4f", cos))")
        }
        for (id, cos) in queryCosines.sorted(by: { $0.value < $1.value }).prefix(3) {
            print("[MLX.EmbeddingGemma.Parity.QueryLow] id=\(id) cos=\(String(format: "%.4f", cos))")
        }

        #expect(meanDocCos >= Self.minimumMeanCosine, "mean document cosine \(meanDocCos) < \(Self.minimumMeanCosine)")
        #expect(meanQueryCos >= Self.minimumMeanCosine, "mean query cosine \(meanQueryCos) < \(Self.minimumMeanCosine)")
        #expect(minDocCos >= Self.minimumPerVectorCosine, "minimum document cosine \(minDocCos) < \(Self.minimumPerVectorCosine)")
        #expect(minQueryCos >= Self.minimumPerVectorCosine, "minimum query cosine \(minQueryCos) < \(Self.minimumPerVectorCosine)")
    }

    // MARK: - Helpers

    private func embed(tokenIDs: [Int32], model: EmbeddingModel, pooler: Pooling) -> [Float] {
        autoreleasepool {
            let tokens = MLXArray(tokenIDs).reshaped(1, -1)
            let output = model(tokens, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
            let pooled = pooler(output)
            pooled.eval()
            return pooled[0].asArray(Float.self)
        }
    }

    private func validateVector(_ vector: [Float], id: String, expectedDim: Int) {
        #expect(vector.count == expectedDim, "\(id) has \(vector.count) dims, expected \(expectedDim)")
        #expect(vector.allSatisfy { $0.isFinite }, "\(id) contains non-finite values")
        let norm = sqrt(vector.reduce(Float(0)) { $0 + $1 * $1 })
        #expect(abs(norm - 1.0) < 1e-2, "\(id) norm \(norm) not L2-normalized")
    }

    private func loadReferenceSnapshot() throws -> ReferenceSnapshot {
        guard let url = Bundle.module.url(forResource: "embeddinggemma_reference", withExtension: "json") else {
            throw TestError.resourceNotFound("embeddinggemma_reference.json")
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ReferenceSnapshot.self, from: data)
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dot / denom : 0
    }

    private func mean(_ values: Dictionary<String, Float>.Values) -> Float {
        guard values.isEmpty == false else { return 0 }
        let total = values.reduce(Float(0), +)
        return total / Float(values.count)
    }
}

// MARK: - Schema

private struct ReferenceSnapshot: Decodable {
    let dataset: Dataset
    let embeddingDimension: Int
    let documentEmbeddings: [String: [Float]]
    let queryEmbeddings: [String: [Float]]
    let documentTokens: [String: [Int32]]
    let queryTokens: [String: [Int32]]
}

private struct Dataset: Decodable {
    let name: String
    let queryPromptName: String?
    let documentPromptName: String?
    let documents: [Document]
    let queries: [Query]
}

private struct Document: Decodable {
    let id: String
    let text: String
}

private struct Query: Decodable {
    let id: String
    let text: String
    let relevantDocumentIDs: [String]
}

private enum TestError: Error, CustomStringConvertible {
    case resourceNotFound(String)
    var description: String {
        switch self {
        case .resourceNotFound(let name): return "Test resource not found: \(name)"
        }
    }
}
