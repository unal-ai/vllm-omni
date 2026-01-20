try:
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionStreamResponse,
    )
except ImportError:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionStreamResponse,
    )


class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    modality: str | None = "text"
