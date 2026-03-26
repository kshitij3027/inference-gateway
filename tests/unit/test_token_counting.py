from gateway.token_counting import count_tokens


class TestCountTokens:
    def test_tiktoken_openai_model(self):
        """tiktoken resolves gpt-4o to an encoding."""
        result = count_tokens("hello world", "gpt-4o")
        assert isinstance(result, int)
        assert result > 0

    def test_tiktoken_gpt4o_mini(self):
        result = count_tokens("hello world", "gpt-4o-mini")
        assert isinstance(result, int)
        assert result > 0

    def test_unknown_model_falls_back_to_chars(self):
        """Unknown models use chars/4 fallback."""
        text = "a" * 100
        result = count_tokens(text, "some-custom-model")
        assert result == 25  # 100 / 4

    def test_empty_string(self):
        result = count_tokens("", "gpt-4o")
        assert result == 0

    def test_fallback_calculation(self):
        """Verify chars/4 math for unknown model."""
        result = count_tokens("hello world!!", "tinyllama")
        assert result == len("hello world!!") // 4
