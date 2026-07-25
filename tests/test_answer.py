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


def test_ask_many_sends_all_images_with_labels_in_one_call():
    fake_client = _FakeClient(_FakeResponse("You worked at your desk."))
    answerer = ClaudeAnswerer(client=fake_client)

    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    labels = ["Image 1, captured 2026-07-09 10:00:00", "Image 2, captured 2026-07-09 11:00:00", "Image 3, captured 2026-07-09 12:00:00"]
    answer = answerer.ask_many(images, "what did I do this morning?", labels=labels)

    assert answer == "You worked at your desk."

    content = fake_client.messages.last_kwargs["messages"][0]["content"]
    image_blocks = [b for b in content if b["type"] == "image"]
    text_blocks = [b for b in content if b["type"] == "text"]

    assert len(image_blocks) == 3
    # each label precedes its image, and the question rides in the final block
    assert text_blocks[0]["text"].startswith("Image 1, captured")
    assert "what did I do this morning?" in text_blocks[-1]["text"]
    # exactly one API call for all three images
    assert all(b["source"]["media_type"] == "image/png" for b in image_blocks)


def test_ask_many_default_labels_when_none_given():
    fake_client = _FakeClient(_FakeResponse("ok"))
    answerer = ClaudeAnswerer(client=fake_client)

    answerer.ask_many([np.zeros((8, 8, 3), dtype=np.uint8)] * 2, "anything?")

    content = fake_client.messages.last_kwargs["messages"][0]["content"]
    text_blocks = [b for b in content if b["type"] == "text"]
    assert text_blocks[0]["text"] == "Image 1:"
    assert text_blocks[1]["text"] == "Image 2:"


def test_ask_many_raises_on_refusal():
    fake_client = _FakeClient(_FakeResponse("", stop_reason="refusal"))
    answerer = ClaudeAnswerer(client=fake_client)

    with pytest.raises(AnswerRefused):
        answerer.ask_many([np.zeros((8, 8, 3), dtype=np.uint8)], "anything?")


def test_claude_ocr_reader_returns_transcription_and_sends_jpeg():
    from imgint.ocr import ClaudeOcrReader

    fake_client = _FakeClient(_FakeResponse("baby job\nDescription: a whiteboard note"))
    reader = ClaudeOcrReader(client=fake_client)

    text = reader.read_text(np.full((40, 60, 3), 200, dtype=np.uint8))
    assert text == "baby job\nDescription: a whiteboard note"

    # image is sent downscaled as JPEG (never PNG -- a photo as PNG exceeds 10MB)
    block = next(b for b in fake_client.messages.last_kwargs["messages"][0]["content"] if b["type"] == "image")
    assert block["source"]["media_type"] == "image/jpeg"


def test_claude_ocr_reader_empty_on_refusal():
    from imgint.ocr import ClaudeOcrReader

    fake_client = _FakeClient(_FakeResponse("", stop_reason="refusal"))
    reader = ClaudeOcrReader(client=fake_client)
    # a refusal must not crash ingest -- just no text for that frame
    assert reader.read_text(np.zeros((8, 8, 3), dtype=np.uint8)) == ""
