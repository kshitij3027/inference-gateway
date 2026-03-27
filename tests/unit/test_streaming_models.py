from gateway.models import ChatCompletionChunk, ChunkChoice, ChunkDelta


class TestChunkDelta:
    def test_role_only(self):
        delta = ChunkDelta(role="assistant")
        assert delta.role == "assistant"
        assert delta.content is None

    def test_content_only(self):
        delta = ChunkDelta(content="Hello")
        assert delta.role is None
        assert delta.content == "Hello"

    def test_empty_delta(self):
        delta = ChunkDelta()
        assert delta.role is None
        assert delta.content is None

    def test_both_fields(self):
        delta = ChunkDelta(role="assistant", content="Hi")
        assert delta.role == "assistant"
        assert delta.content == "Hi"


class TestChunkChoice:
    def test_with_finish_reason(self):
        choice = ChunkChoice(
            delta=ChunkDelta(),
            finish_reason="stop",
        )
        assert choice.index == 0
        assert choice.finish_reason == "stop"

    def test_without_finish_reason(self):
        choice = ChunkChoice(delta=ChunkDelta(content="Hi"))
        assert choice.finish_reason is None


class TestChatCompletionChunk:
    def test_round_trip(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-test123",
            created=1700000000,
            model="tinyllama",
            choices=[
                ChunkChoice(delta=ChunkDelta(content="Hello"))
            ],
        )
        data = chunk.model_dump()
        assert data["id"] == "chatcmpl-test123"
        assert data["object"] == "chat.completion.chunk"
        assert data["created"] == 1700000000
        assert data["model"] == "tinyllama"
        assert data["choices"][0]["delta"]["content"] == "Hello"
        assert data["choices"][0]["finish_reason"] is None

    def test_json_serialization(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-abc",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                ChunkChoice(delta=ChunkDelta(role="assistant"))
            ],
        )
        json_str = chunk.model_dump_json()
        assert '"chat.completion.chunk"' in json_str
        assert '"assistant"' in json_str
