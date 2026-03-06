import MLXLMCommon
import XCTest

final class PromptCacheSnapshotTests: XCTestCase {
    func testSnapshotRoundTripPreservesCacheTypesAndMetadata() throws {
        let arraysCache = ArraysCache(size: 0)

        let snapshot = try capturePromptCache(
            cache: [arraysCache],
            prefixTokenCount: 42,
            metadata: ["scope": "unit"]
        )

        XCTAssertEqual(snapshot.prefixTokenCount, 42)
        XCTAssertEqual(snapshot.cacheClasses, ["ArraysCache"])
        XCTAssertEqual(snapshot.metadata["scope"], "unit")
        XCTAssertEqual(snapshot.cacheMetaState, [[""]])
        XCTAssertEqual(snapshot.cacheState.count, 1)
        XCTAssertTrue(snapshot.cacheState[0].isEmpty)

        let restoredCaches = try materializePromptCache(from: snapshot)
        let restoredArraysCache = try XCTUnwrap(restoredCaches[0] as? ArraysCache)
        XCTAssertEqual(restoredArraysCache.state.count, 0)
        XCTAssertEqual(restoredArraysCache.metaState, [""])
    }

    func testSnapshotCaptureAndMaterializationDoNotShareLiveCaches() throws {
        let cache = ArraysCache(size: 0)

        let snapshot = try capturePromptCache(cache: [cache], prefixTokenCount: 2)

        let firstMaterialization = try materializePromptCache(from: snapshot)
        let firstCache = try XCTUnwrap(firstMaterialization[0] as? ArraysCache)

        let secondMaterialization = try materializePromptCache(from: snapshot)
        let secondCache = try XCTUnwrap(secondMaterialization[0] as? ArraysCache)
        XCTAssertNotEqual(
            ObjectIdentifier(firstCache),
            ObjectIdentifier(secondCache)
        )
    }
}
