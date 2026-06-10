from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class OpenCLIPBackbone:
    model_name: str
    pretrained: str
    device: str = "cuda"
    precision: str = "fp32"

    def __post_init__(self) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("Install open_clip_torch to use CLIP-family backbones.") from exc

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.model, self.train_preprocess, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            precision=self.precision,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def id(self) -> str:
        return f"{self.model_name}__{self.pretrained}".replace("/", "-")

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.model.encode_image(images.to(self.device))
        features = features.float()
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[start : start + batch_size]).to(self.device)
            outputs.append(F.normalize(self.model.encode_text(tokens).float(), dim=-1).cpu())
        return torch.cat(outputs, dim=0)
