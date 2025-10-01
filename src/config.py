"""Configuration module for YOLO detection system."""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch


@dataclass
class YOLOConfig:
    """YOLO 모델 설정 클래스."""

    # 모델 설정
    model_name: str = "yolov8n"
    weights: Optional[str] = None
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # 추론 파라미터
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: int = 640
    max_det: int = 100

    # 클래스 필터 (None이면 모든 클래스)
    classes: Optional[list] = None

    # 성능 설정
    half: bool = False  # FP16 추론
    augment: bool = False  # 테스트 시간 증강

    # 출력 설정
    save_txt: bool = False  # 텍스트 결과 저장
    save_conf: bool = True  # 신뢰도 점수 저장
    save_crop: bool = False  # 탐지된 객체 크롭 저장

    # 시각화 설정
    line_thickness: int = 2
    hide_labels: bool = False
    hide_conf: bool = False

    def __post_init__(self):
        """설정 유효성 검사."""
        if self.conf_threshold < 0 or self.conf_threshold > 1:
            raise ValueError(f"conf_threshold는 0-1 사이여야 합니다: {self.conf_threshold}")

        if self.iou_threshold < 0 or self.iou_threshold > 1:
            raise ValueError(f"iou_threshold는 0-1 사이여야 합니다: {self.iou_threshold}")

        if self.img_size <= 0:
            raise ValueError(f"img_size는 양수여야 합니다: {self.img_size}")

        if self.weights is None:
            self.weights = f"{self.model_name}.pt"

    def to_dict(self) -> dict:
        """설정을 딕셔너리로 변환."""
        return {
            "model_name": self.model_name,
            "weights": self.weights,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "img_size": self.img_size,
            "max_det": self.max_det,
            "classes": self.classes,
            "half": self.half,
            "augment": self.augment,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "YOLOConfig":
        """딕셔너리로부터 설정 생성."""
        return cls(**config_dict)

    def __str__(self) -> str:
        """설정 정보 출력."""
        return (
            f"YOLOConfig(\n"
            f"  model={self.model_name}\n"
            f"  device={self.device}\n"
            f"  conf={self.conf_threshold}\n"
            f"  iou={self.iou_threshold}\n"
            f"  img_size={self.img_size}\n"
            f")"
        )


# 사전 정의된 설정
PRESET_CONFIGS = {
    "fast": YOLOConfig(
        model_name="yolov8n",
        conf_threshold=0.3,
        img_size=640,
        half=False  # FP16 비활성화 (타입 불일치 방지)
    ),
    "balanced": YOLOConfig(
        model_name="yolov8s",
        conf_threshold=0.25,
        img_size=640,
    ),
    "accurate": YOLOConfig(
        model_name="yolov8m",
        conf_threshold=0.2,
        img_size=640,
    ),
}
