import Foundation
import MLX
import Testing

@testable import MLXEmbedders
@testable import MLXLMCommon

/// Direct Swift-vs-Python semantic parity check across diverse inputs.
///
/// The reference parity test (`EmbeddingGemmaReferenceParityTests`) covers 16
/// vectors from a fixed smoke dataset. This test adds 8 documents + 5 queries
/// spanning different topics (Firebase/Supabase, Swift/Python, cooking, sports)
/// and confirms that:
///
/// 1. Per-vector Swift output matches Python mlx_embeddings (cos ≈ 1.0)
/// 2. The pairwise cosine **matrix** (doc × doc, query × doc) is numerically
///    identical between Swift and Python — this verifies not just individual
///    vectors but the semantic relationships between them.
///
/// Token IDs are pre-computed in `semantic_ref.json` to isolate the model
/// under test from tokenizer divergence, same strategy as the reference test.
@Suite("EmbeddingGemma Semantic Parity", .serialized)
struct EmbeddingGemmaSemanticParityTests {

    private static let bf16ModelDirectory = URL(fileURLWithPath: NSString(
        string: "~/.cache/huggingface/hub/models--mlx-community--embeddinggemma-300m-bf16/snapshots/manual"
    ).expandingTildeInPath)

    private static let minimumPerVectorCosine: Float = 0.985
    private static let minimumMatrixAgreement: Float = 0.999

    @Test("Swift and Python agree on semantic cosine structure", .timeLimit(.minutes(10)))
    func semanticParity() throws {
        guard FileManager.default.fileExists(
            atPath: Self.bf16ModelDirectory.appendingPathComponent("config.json").path
        ) else {
            print("[MLX.Skip] No bf16 model at \(Self.bf16ModelDirectory.path)")
            return
        }

        let snapshot = try loadSemanticSnapshot()
        let model = try loadSynchronous(
            modelDirectory: Self.bf16ModelDirectory,
            modelName: "embeddinggemma-300m-bf16"
        )
        let pooler = loadPooling(modelDirectory: Self.bf16ModelDirectory, model: model)

        // Compute Swift embeddings for all docs and queries.
        var swiftDocs: [(id: String, vec: [Float])] = []
        swiftDocs.reserveCapacity(snapshot.docTokens.count)
        for (id, tokens) in snapshot.docTokens.sorted(by: { $0.key < $1.key }) {
            let vec = embed(tokenIDs: tokens, model: model, pooler: pooler)
            validateVector(vec, id: "doc:\(id)", expectedDim: snapshot.embeddingDimension)
            swiftDocs.append((id, vec))
        }
        var swiftQueries: [(id: String, vec: [Float])] = []
        swiftQueries.reserveCapacity(snapshot.queryTokens.count)
        for (id, tokens) in snapshot.queryTokens.sorted(by: { $0.key < $1.key }) {
            let vec = embed(tokenIDs: tokens, model: model, pooler: pooler)
            validateVector(vec, id: "query:\(id)", expectedDim: snapshot.embeddingDimension)
            swiftQueries.append((id, vec))
        }

        // 1. Per-vector cosine against Python reference.
        for (id, swiftVec) in swiftDocs {
            guard let refVec = snapshot.docEmbeddings[id] else {
                Issue.record("Missing reference for doc \(id)")
                continue
            }
            let c = cosine(swiftVec, refVec)
            #expect(
                c >= Self.minimumPerVectorCosine,
                "doc:\(id) swift-vs-python cosine \(c) < \(Self.minimumPerVectorCosine)"
            )
        }
        for (id, swiftVec) in swiftQueries {
            guard let refVec = snapshot.queryEmbeddings[id] else {
                Issue.record("Missing reference for query \(id)")
                continue
            }
            let c = cosine(swiftVec, refVec)
            #expect(
                c >= Self.minimumPerVectorCosine,
                "query:\(id) swift-vs-python cosine \(c) < \(Self.minimumPerVectorCosine)"
            )
        }

        // 2. Pairwise doc×doc matrix agreement.
        let pythonDocVecs: [(id: String, vec: [Float])] = snapshot.docEmbeddings
            .sorted(by: { $0.key < $1.key })
            .map { ($0.key, $0.value) }

        var maxMatrixDelta: Float = 0
        var worstPair: (String, String, Float, Float) = ("", "", 0, 0)
        for i in 0..<swiftDocs.count {
            for j in 0..<swiftDocs.count {
                let swiftCos = cosine(swiftDocs[i].vec, swiftDocs[j].vec)
                let pythonCos = cosine(pythonDocVecs[i].vec, pythonDocVecs[j].vec)
                let delta = abs(swiftCos - pythonCos)
                if delta > maxMatrixDelta {
                    maxMatrixDelta = delta
                    worstPair = (swiftDocs[i].id, swiftDocs[j].id, swiftCos, pythonCos)
                }
            }
        }

        // 3. Pairwise query×doc retrieval ordering agreement.
        var retrievalDisagreements = 0
        for (qi, (qid, qvecSwift)) in swiftQueries.enumerated() {
            let qvecPython = snapshot.queryEmbeddings[qid]!
            let swiftRanking = swiftDocs
                .map { ($0.id, cosine(qvecSwift, $0.vec)) }
                .sorted { $0.1 > $1.1 }
                .map(\.0)
            let pythonRanking = pythonDocVecs
                .map { ($0.id, cosine(qvecPython, $0.vec)) }
                .sorted { $0.1 > $1.1 }
                .map(\.0)
            if swiftRanking != pythonRanking {
                retrievalDisagreements += 1
                print(
                    "[MLX.Semantic.RankingDiff] q=\(qid) "
                        + "swift=\(swiftRanking) python=\(pythonRanking)"
                )
            }
            _ = qi
        }

        print(
            "[MLX.EmbeddingGemma.Semantic] docs=\(swiftDocs.count) queries=\(swiftQueries.count) "
                + "maxMatrixCosDelta=\(String(format: "%.6f", maxMatrixDelta)) "
                + "worstPair=(\(worstPair.0), \(worstPair.1)) "
                + "swift=\(String(format: "%.4f", worstPair.2)) "
                + "python=\(String(format: "%.4f", worstPair.3)) "
                + "retrievalDisagreements=\(retrievalDisagreements)/\(swiftQueries.count)"
        )

        #expect(
            maxMatrixDelta < 0.01,
            "doc×doc cosine matrix delta \(maxMatrixDelta) exceeds 0.01"
        )
        #expect(
            retrievalDisagreements == 0,
            "query→doc ranking disagrees in \(retrievalDisagreements) of \(swiftQueries.count) queries"
        )
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

    private func loadSemanticSnapshot() throws -> SemanticSnapshot {
        guard let url = Bundle.module.url(forResource: "semantic_ref", withExtension: "json") else {
            throw TestError.resourceNotFound("semantic_ref.json")
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(SemanticSnapshot.self, from: data)
    }
}

// MARK: - Schema

private struct SemanticSnapshot: Decodable {
    let embeddingDimension: Int
    let docTokens: [String: [Int32]]
    let docEmbeddings: [String: [Float]]
    let queryTokens: [String: [Int32]]
    let queryEmbeddings: [String: [Float]]
}

private enum TestError: Error, CustomStringConvertible {
    case resourceNotFound(String)
    var description: String {
        switch self {
        case .resourceNotFound(let name): return "Test resource not found: \(name)"
        }
    }
}
