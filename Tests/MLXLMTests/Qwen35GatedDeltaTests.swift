// Copyright © 2026 Apple Inc.

import MLX
@testable import MLXLMCommon
import XCTest

final class Qwen35GatedDeltaTests: XCTestCase {

    func testSharedGatedDeltaUpdateProducesExpectedShapes() throws {
        try requireRuntimeOptIn()

        let q = MLXArray.ones([1, 2, 1, 32])
        let k = MLXArray.ones([1, 2, 1, 32])
        let v = MLXArray.ones([1, 2, 1, 8])
        let a = MLXArray.zeros([1, 2, 1])
        let b = MLXArray.zeros([1, 2, 1])
        let aLog = MLXArray.zeros([1])
        let dtBias = MLXArray.zeros([1])

        let (output, state) = gatedDeltaUpdate(
            q: q,
            k: k,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias
        )

        XCTAssertEqual(output.shape, [1, 2, 1, 8])
        XCTAssertEqual(state.shape, [1, 1, 8, 32])
    }

    func testSharedGatedDeltaUpdateAcceptsMask() throws {
        try requireRuntimeOptIn()

        let q = MLXArray.ones([1, 2, 1, 32])
        let k = MLXArray.ones([1, 2, 1, 32])
        let v = MLXArray.ones([1, 2, 1, 8])
        let a = MLXArray.zeros([1, 2, 1])
        let b = MLXArray.zeros([1, 2, 1])
        let aLog = MLXArray.zeros([1])
        let dtBias = MLXArray.zeros([1])
        let mask = MLXArray([true, false], [1, 2])

        let (output, state) = gatedDeltaUpdate(
            q: q,
            k: k,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            mask: mask
        )

        XCTAssertEqual(output.shape, [1, 2, 1, 8])
        XCTAssertEqual(state.shape, [1, 1, 8, 32])
    }

    private func requireRuntimeOptIn() throws {
        guard ProcessInfo.processInfo.environment["QWEN35_RUNTIME_TESTS"] == "1" else {
            throw XCTSkip(
                "Requires MLX runtime execution. Set QWEN35_RUNTIME_TESTS=1 when running under a Metal-capable test environment."
            )
        }
    }
}
