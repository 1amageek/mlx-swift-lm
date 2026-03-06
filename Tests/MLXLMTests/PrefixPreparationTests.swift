import MLXLMCommon
@testable import MLXLLM
import MLXVLM
import XCTest

final class PrefixPreparationTests: XCTestCase {
    func testLLMPrepareAddsGenerationPrompt() throws {
        let recorder = ChatTemplateCallRecorder()
        let tokenizer = TestTokenizer(recorder: recorder)
        let processor = LLMUserInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator()
        )

        _ = try processor.promptTokens(for: UserInput(prompt: "hello"), mode: .generation)

        XCTAssertEqual(recorder.addGenerationPromptValues, [true])
    }

    func testLLMPreparePrefixOmitsGenerationPrompt() throws {
        let recorder = ChatTemplateCallRecorder()
        let tokenizer = TestTokenizer(recorder: recorder)
        let processor = LLMUserInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator()
        )

        _ = try processor.promptTokens(for: UserInput(prompt: "hello"), mode: .prefixSnapshot)

        XCTAssertEqual(recorder.addGenerationPromptValues, [false])
    }

    func testQwen3VLPrepareAddsGenerationPromptForFullGeneration() async throws {
        let recorder = ChatTemplateCallRecorder()
        let tokenizer = TestTokenizer(recorder: recorder)
        let processor = Qwen3VLProcessor(try makeQwen3VLProcessorConfiguration(), tokenizer: tokenizer)

        _ = try processor.promptTokens(for: UserInput(prompt: "hello"), mode: .generation)

        XCTAssertEqual(recorder.addGenerationPromptValues, [true])
    }

    func testQwen3VLPreparePrefixOmitsGenerationPromptForTextOnlyInput() async throws {
        let recorder = ChatTemplateCallRecorder()
        let tokenizer = TestTokenizer(recorder: recorder)
        let processor = Qwen3VLProcessor(try makeQwen3VLProcessorConfiguration(), tokenizer: tokenizer)

        _ = try processor.promptTokens(for: UserInput(prompt: "hello"), mode: .prefixSnapshot)

        XCTAssertEqual(recorder.addGenerationPromptValues, [false])
    }

    private func makeQwen3VLProcessorConfiguration() throws -> Qwen3VLProcessorConfiguration {
        let json = """
        {
          "image_mean": [0.5, 0.5, 0.5],
          "image_std": [0.5, 0.5, 0.5],
          "min_pixels": 3136,
          "max_pixels": 12845056,
          "merge_size": 2,
          "patch_size": 14,
          "temporal_patch_size": 2,
          "image_processor_type": "Qwen3VLImageProcessor"
        }
        """
        return try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self,
            from: Data(json.utf8)
        )
    }
}
