# ./utils/text_cond.py
import torch
from transformers import CLIPTokenizer, CLIPTextModel

class TextCondEncoder(torch.nn.Module):
    def __init__(self, clip_name="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_name)
        self.device = device
        self.to(device)
        self.text_encoder.eval()

    def forward(self, text_list):
        """
        text_list: List[str]，比如 [prompt0, prompt1, ...]
        返回: (B, D) 的向量（用 CLS token）
        """
        if len(text_list) == 0:
            raise ValueError("empty text_list")

        inputs = self.tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.text_encoder(**inputs)
            # 取 [CLS]，简单粗暴
            emb = out.last_hidden_state[:, 0, :]  # (B, D)

        return emb
