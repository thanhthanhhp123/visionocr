import json
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
        self.data      = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]

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