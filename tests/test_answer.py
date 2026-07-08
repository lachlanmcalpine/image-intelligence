import numpy as np
import pytest

from imgint.answer import AnswerRefused, ClaudeAnswerer


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_FakeTextBlock(text)]
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def test_ask_returns_text_from_response():
    fake_client = _FakeClient(_FakeResponse("Yes, I can see a red mug on the desk."))
    answerer = ClaudeAnswerer(client=fake_client)

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    answer = answerer.ask(image, "Is there a mug on the desk?")

    assert answer == "Yes, I can see a red mug on the desk."


def test_ask_sends_image_and_question_in_correct_shape():
    fake_client = _FakeClient(_FakeResponse("ok"))
    answerer = ClaudeAnswerer(client=fake_client)

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    answerer.ask(image, "What's here?")

    sent = fake_client.messages.last_kwargs
    assert sent["model"] == "claude-sonnet-5"

    content = sent["messages"][0]["content"]
    image_block = next(b for b in content if b["type"] == "image")
    text_block = next(b for b in content if b["type"] == "text")

    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/png"
    assert text_block["text"] == "What's here?"


def test_ask_raises_on_refusal():
    fake_client = _FakeClient(_FakeResponse("", stop_reason="refusal"))
    answerer = ClaudeAnswerer(client=fake_client)

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(AnswerRefused):
        answerer.ask(image, "anything")
