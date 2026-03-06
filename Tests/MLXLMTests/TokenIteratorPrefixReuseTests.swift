@testable import MLXLMCommon
import XCTest

final class TokenIteratorPrefixReuseTests: XCTestCase {
    func testSlicePlanUsesTrailingAxisForBatchedTokens() {
        let plan = ReusedPrefixSlicePlan.make(
            tokenShape: [1, 5],
            maskShape: [1, 5],
            cachedTokenCount: 3
        )

        XCTAssertEqual(
            plan,
            ReusedPrefixSlicePlan(
                tokenSlice: .twoDimensionalTrailingAxis(start: 3),
                maskSlice: .twoDimensionalTrailingAxis(start: 3)
            )
        )
    }

    func testSlicePlanUsesOneDimensionalRangeForFlatTokens() {
        let plan = ReusedPrefixSlicePlan.make(
            tokenShape: [5],
            maskShape: [5],
            cachedTokenCount: 3
        )

        XCTAssertEqual(
            plan,
            ReusedPrefixSlicePlan(
                tokenSlice: .oneDimensional(start: 3),
                maskSlice: .oneDimensional(start: 3)
            )
        )
    }

    func testSlicePlanRejectsCachedPrefixThatConsumesEntireSequence() {
        XCTAssertNil(
            ReusedPrefixSlicePlan.make(
                tokenShape: [1, 5],
                maskShape: [1, 5],
                cachedTokenCount: 5
            )
        )
    }

    func testSlicePlanRejectsUnsupportedMaskRank() {
        XCTAssertNil(
            ReusedPrefixSlicePlan.make(
                tokenShape: [1, 5],
                maskShape: [1, 1, 5],
                cachedTokenCount: 3
            )
        )
    }
}
