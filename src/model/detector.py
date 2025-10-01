"""YOLO detection module."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

from ..config import YOLOConfig


@dataclass
class Detection:
    """단일 객체 탐지 결과."""

    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self) -> dict:
        """딕셔너리로 변환."""
        return {
            "bbox": self.bbox,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Detection":
        """딕셔너리로부터 생성."""
        return cls(
            bbox=tuple(data["bbox"]),
            class_id=data["class_id"],
            class_name=data["class_name"],
            confidence=data["confidence"],
        )

    def __repr__(self) -> str:
        """문자열 표현."""
        x1, y1, x2, y2 = self.bbox
        return (
            f"Detection(class={self.class_name}, "
            f"conf={self.confidence:.2f}, "
            f"bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}])"
        )


class YOLODetector:
    """YOLO 객체 탐지 클래스."""

    def __init__(self, config: YOLOConfig):
        """
        Args:
            config: YOLO 설정
        """
        self.config = config
        self.model = None
        self.device = config.device

        # 모델 로딩
        self._load_model()

    def _load_model(self):
        """YOLO 모델 로딩."""
        try:
            print(f"YOLO 모델 로딩 중: {self.config.model_name}")
            print(f"가중치: {self.config.weights}")
            print(f"디바이스: {self.device}")

            # Ultralytics YOLO 모델 로딩
            self.model = YOLO(self.config.weights)

            # 디바이스 설정
            self.model.to(self.device)

            # FP16 설정
            if self.config.half and self.device != "cpu":
                self.model.half()

            print(f"모델 로딩 완료")
            print(f"클래스 수: {len(self.model.names)}")

        except Exception as e:
            raise RuntimeError(f"모델 로딩 실패: {str(e)}")

    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        verbose: bool = False,
    ) -> List[Detection]:
        """
        단일 이미지 객체 탐지.

        Args:
            image: 이미지 배열 또는 파일 경로
            verbose: 상세 로그 출력 여부

        Returns:
            탐지 결과 리스트
        """
        if self.model is None:
            raise RuntimeError("모델이 로딩되지 않았습니다")

        # YOLO 추론
        results = self.model.predict(
            source=image,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            max_det=self.config.max_det,
            classes=self.config.classes,
            augment=self.config.augment,
            verbose=verbose,
            device=self.device,
        )

        # 결과 파싱
        detections = self._parse_results(results[0])

        return detections

    def predict_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        verbose: bool = False,
    ) -> List[List[Detection]]:
        """
        배치 이미지 객체 탐지.

        Args:
            images: 이미지 리스트
            verbose: 상세 로그 출력 여부

        Returns:
            각 이미지별 탐지 결과 리스트
        """
        if self.model is None:
            raise RuntimeError("모델이 로딩되지 않았습니다")

        # 배치 추론
        results = self.model.predict(
            source=images,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            max_det=self.config.max_det,
            classes=self.config.classes,
            augment=self.config.augment,
            verbose=verbose,
            device=self.device,
        )

        # 각 이미지별 결과 파싱
        batch_detections = []
        for result in results:
            detections = self._parse_results(result)
            batch_detections.append(detections)

        return batch_detections

    def _parse_results(self, result) -> List[Detection]:
        """
        YOLO 결과를 Detection 객체로 변환.

        Args:
            result: ultralytics Results 객체

        Returns:
            Detection 객체 리스트
        """
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # 각 탐지 객체 파싱
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
            detection = Detection(
                bbox=tuple(bbox.tolist()),
                class_id=int(cls_id),
                class_name=self.model.names[cls_id],
                confidence=float(conf),
            )
            detections.append(detection)

        return detections

    def get_model_info(self) -> dict:
        """모델 정보 반환."""
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "num_classes": len(self.model.names),
            "class_names": self.model.names,
            "img_size": self.config.img_size,
            "conf_threshold": self.config.conf_threshold,
            "iou_threshold": self.config.iou_threshold,
        }

    def __repr__(self) -> str:
        """문자열 표현."""
        return (
            f"YOLODetector(\n"
            f"  model={self.config.model_name}\n"
            f"  device={self.device}\n"
            f"  classes={len(self.model.names) if self.model else 'N/A'}\n"
            f")"
        )
