"""Visualization module for detection results."""

from typing import List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..model.detector import Detection


class Visualizer:
    """탐지 결과 시각화 클래스."""

    # COCO 클래스 색상 (80개)
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
    ]

    def __init__(
        self,
        line_thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        hide_labels: bool = False,
        hide_conf: bool = False,
    ):
        """
        Args:
            line_thickness: 바운딩 박스 선 두께
            font_scale: 폰트 크기
            font_thickness: 폰트 두께
            hide_labels: 레이블 숨김 여부
            hide_conf: 신뢰도 점수 숨김 여부
        """
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        copy: bool = True,
    ) -> np.ndarray:
        """
        이미지에 탐지 결과 그리기.

        Args:
            image: 원본 이미지
            detections: 탐지 결과 리스트
            copy: 이미지 복사 여부

        Returns:
            탐지 결과가 그려진 이미지
        """
        if copy:
            image = image.copy()

        for detection in detections:
            self._draw_single_detection(image, detection)

        return image

    def _draw_single_detection(self, image: np.ndarray, detection: Detection):
        """단일 탐지 결과 그리기."""
        x1, y1, x2, y2 = map(int, detection.bbox)

        # 색상 선택
        color = self._get_color(detection.class_id)

        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_thickness)

        # 레이블 그리기
        if not self.hide_labels:
            label = self._create_label(detection)
            self._draw_label(image, label, (x1, y1), color)

    def _create_label(self, detection: Detection) -> str:
        """레이블 텍스트 생성."""
        if self.hide_conf:
            return detection.class_name
        else:
            return f"{detection.class_name} {detection.confidence:.2f}"

    def _draw_label(
        self,
        image: np.ndarray,
        label: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ):
        """레이블 텍스트와 배경 그리기."""
        x, y = position

        # 텍스트 크기 계산
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            self.font_thickness,
        )

        # 배경 박스 그리기
        cv2.rectangle(
            image,
            (x, y - label_height - baseline - 5),
            (x + label_width, y),
            color,
            -1,
        )

        # 텍스트 그리기
        cv2.putText(
            image,
            label,
            (x, y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness,
        )

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """클래스 ID에 따른 색상 반환."""
        return self.COLORS[class_id % len(self.COLORS)]

    def show(
        self,
        image: np.ndarray,
        detections: List[Detection],
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None,
    ):
        """
        matplotlib으로 이미지 표시.

        Args:
            image: 이미지
            detections: 탐지 결과
            figsize: Figure 크기
            title: 제목
        """
        # 탐지 결과 그리기
        annotated_image = self.draw_detections(image, detections)

        # RGB → BGR 변환 (matplotlib는 RGB)
        if annotated_image.shape[2] == 3:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # 플롯
        plt.figure(figsize=figsize)
        plt.imshow(annotated_image)
        plt.axis("off")

        if title:
            plt.title(title)
        else:
            plt.title(f"Detected Objects: {len(detections)}")

        plt.tight_layout()
        plt.show()

    def create_comparison(
        self,
        images: List[np.ndarray],
        detections_list: List[List[Detection]],
        titles: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> Figure:
        """
        여러 이미지 비교 시각화.

        Args:
            images: 이미지 리스트
            detections_list: 각 이미지별 탐지 결과
            titles: 각 이미지 제목
            figsize: Figure 크기

        Returns:
            matplotlib Figure
        """
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=figsize)

        if num_images == 1:
            axes = [axes]

        for idx, (image, detections) in enumerate(zip(images, detections_list)):
            # 탐지 결과 그리기
            annotated = self.draw_detections(image, detections)

            # RGB 변환
            if annotated.shape[2] == 3:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # 플롯
            axes[idx].imshow(annotated)
            axes[idx].axis("off")

            # 제목
            if titles and idx < len(titles):
                axes[idx].set_title(titles[idx])
            else:
                axes[idx].set_title(f"Objects: {len(detections)}")

        plt.tight_layout()
        return fig

    @staticmethod
    def create_detection_summary(detections: List[Detection]) -> dict:
        """
        탐지 결과 요약 통계.

        Args:
            detections: 탐지 결과 리스트

        Returns:
            요약 통계 딕셔너리
        """
        if not detections:
            return {
                "total": 0,
                "by_class": {},
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "min_confidence": 0.0,
            }

        # 클래스별 카운트
        class_counts = {}
        confidences = []

        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
            confidences.append(det.confidence)

        return {
            "total": len(detections),
            "by_class": class_counts,
            "avg_confidence": np.mean(confidences),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
        }

    @staticmethod
    def print_detection_summary(detections: List[Detection]):
        """탐지 결과 요약 출력."""
        summary = Visualizer.create_detection_summary(detections)

        print(f"\n{'='*50}")
        print(f"탐지 결과 요약")
        print(f"{'='*50}")
        print(f"총 객체 수: {summary['total']}")
        print(f"\n클래스별 분포:")
        for class_name, count in summary['by_class'].items():
            print(f"  - {class_name}: {count}")
        print(f"\n신뢰도 통계:")
        print(f"  - 평균: {summary['avg_confidence']:.3f}")
        print(f"  - 최대: {summary['max_confidence']:.3f}")
        print(f"  - 최소: {summary['min_confidence']:.3f}")
        print(f"{'='*50}\n")
