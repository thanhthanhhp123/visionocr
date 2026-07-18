import json
from pathlib import Path

from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info


PROMPT_TEMPLATE = (
    "Trích xuất thông tin hóa đơn Việt Nam từ ảnh.\n"
    "{ocr_hint}"
    "Trả về JSON với format:\n"
    '{{"store_name":"","date":"YYYY-MM-DD","total":0,"discount":0,'
    '"items":[{{"name":"","unit_price":0,"quantity":0,"total_price":0}}]}}'
)


class InvoiceDataset(Dataset):
    def __init__(self, jsonl_path: str, processor):
        self.jsonl_path = Path(jsonl_path)
        self.image_dir  = self.jsonl_path.parent / "images"
        with open(jsonl_path, encoding="utf-8") as data_file:
            self.data = [json.loads(line) for line in data_file]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self._messages_from_sample(self.data[idx])

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        images, _ = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )

        input_ids = inputs["input_ids"][0]
        labels    = input_ids.clone()

        # Mask prompt tokens — only compute loss on assistant response
        assistant_tokens = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        for i in range(len(input_ids) - len(assistant_tokens)):
            if input_ids[i : i + len(assistant_tokens)].tolist() == assistant_tokens:
                labels[: i + len(assistant_tokens)] = -100
                break

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values":   inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),  # fix: (1,3) → (3,)
            "labels":         labels,
        }

    def _messages_from_sample(self, sample):
        if "messages" in sample:
            return sample["messages"]

        if "image" not in sample or "label" not in sample:
            raise KeyError("Expected sample to contain either 'messages' or both 'image' and 'label'")

        ocr_hint = ""
        if sample.get("ocr_text"):
            ocr_hint = f"OCR text tham khảo:\n{sample['ocr_text']}\n\n"

        prompt = PROMPT_TEMPLATE.format(ocr_hint=ocr_hint)
        label = json.dumps(sample["label"], ensure_ascii=False)

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._resolve_image_path(sample["image"])},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]

    def _resolve_image_path(self, image_path):
        image_path = Path(image_path)
        if image_path.exists():
            return str(image_path)

        local_image_path = self.image_dir / image_path.name
        if local_image_path.exists():
            return str(local_image_path)

        return str(image_path)
