"""Result export module."""

import json
from pathlib import Path
from typing import List, Union, Optional
from datetime import datetime
import numpy as np
import cv2

from ..model.detector import Detection


class ResultExporter:
    """탐지 결과 저장 클래스."""

    def __init__(self, output_dir: Union[str, Path] = "data/results"):
        """
        Args:
            output_dir: 출력 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_image(
        self,
        image: np.ndarray,
        filename: Optional[str] = None,
        prefix: str = "result",
    ) -> Path:
        """
        이미지 저장.

        Args:
            image: 저장할 이미지
            filename: 파일명 (None이면 타임스탬프 사용)
            prefix: 파일명 접두사

        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.jpg"

        filepath = self.output_dir / filename

        # BGR 변환 (OpenCV는 BGR)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(filepath), image)
        print(f"이미지 저장: {filepath}")

        return filepath

    def save_detections_json(
        self,
        detections: List[Detection],
        filename: Optional[str] = None,
        prefix: str = "detections",
        metadata: Optional[dict] = None,
    ) -> Path:
        """
        탐지 결과를 JSON으로 저장.

        Args:
            detections: 탐지 결과 리스트
            filename: 파일명
            prefix: 파일명 접두사
            metadata: 추가 메타데이터

        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.json"

        filepath = self.output_dir / filename

        # JSON 데이터 구성
        data = {
            "timestamp": datetime.now().isoformat(),
            "num_detections": len(detections),
            "detections": [det.to_dict() for det in detections],
        }

        if metadata:
            data["metadata"] = metadata

        # 저장
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"탐지 결과 저장: {filepath}")

        return filepath

    def save_detections_txt(
        self,
        detections: List[Detection],
        image_shape: tuple,
        filename: Optional[str] = None,
        prefix: str = "detections",
        format: str = "yolo",
    ) -> Path:
        """
        탐지 결과를 텍스트로 저장 (YOLO 또는 COCO 형식).

        Args:
            detections: 탐지 결과 리스트
            image_shape: 이미지 크기 (H, W)
            filename: 파일명
            prefix: 파일명 접두사
            format: 저장 형식 ("yolo" 또는 "coco")

        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.txt"

        filepath = self.output_dir / filename

        h, w = image_shape[:2]

        with open(filepath, "w") as f:
            for det in detections:
                if format == "yolo":
                    # YOLO 형식: class_id x_center y_center width height
                    x1, y1, x2, y2 = det.bbox
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    line = f"{det.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                else:  # COCO 형식
                    # COCO 형식: class_id x1 y1 width height confidence
                    x1, y1, x2, y2 = det.bbox
                    width = x2 - x1
                    height = y2 - y1
                    line = f"{det.class_id} {x1:.2f} {y1:.2f} {width:.2f} {height:.2f} {det.confidence:.4f}\n"

                f.write(line)

        print(f"텍스트 결과 저장: {filepath}")

        return filepath

    def save_batch_results(
        self,
        images: List[np.ndarray],
        detections_list: List[List[Detection]],
        image_names: Optional[List[str]] = None,
        save_images: bool = True,
        save_json: bool = True,
    ) -> dict:
        """
        배치 결과 저장.

        Args:
            images: 이미지 리스트
            detections_list: 각 이미지별 탐지 결과
            image_names: 이미지 파일명 리스트
            save_images: 이미지 저장 여부
            save_json: JSON 저장 여부

        Returns:
            저장된 파일 경로 딕셔너리
        """
        results = {
            "images": [],
            "json": [],
            "summary": {
                "total_images": len(images),
                "total_detections": sum(len(dets) for dets in detections_list),
            }
        }

        for idx, (image, detections) in enumerate(zip(images, detections_list)):
            # 파일명 생성
            if image_names and idx < len(image_names):
                base_name = Path(image_names[idx]).stem
            else:
                base_name = f"batch_{idx:04d}"

            # 이미지 저장
            if save_images:
                img_path = self.save_image(
                    image,
                    filename=f"{base_name}.jpg",
                )
                results["images"].append(str(img_path))

            # JSON 저장
            if save_json:
                json_path = self.save_detections_json(
                    detections,
                    filename=f"{base_name}.json",
                    metadata={"image_index": idx, "image_name": base_name},
                )
                results["json"].append(str(json_path))

        return results

    def load_detections_json(self, filepath: Union[str, Path]) -> List[Detection]:
        """
        JSON 파일에서 탐지 결과 로딩.

        Args:
            filepath: JSON 파일 경로

        Returns:
            탐지 결과 리스트
        """
        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        detections = [Detection.from_dict(det) for det in data["detections"]]

        return detections

    def create_summary_report(
        self,
        detections_list: List[List[Detection]],
        filename: str = "summary_report.txt",
    ) -> Path:
        """
        전체 탐지 결과 요약 보고서 생성.

        Args:
            detections_list: 모든 이미지의 탐지 결과
            filename: 보고서 파일명

        Returns:
            저장된 파일 경로
        """
        filepath = self.output_dir / filename

        total_detections = sum(len(dets) for dets in detections_list)
        total_images = len(detections_list)

        # 클래스별 통계
        class_counts = {}
        all_confidences = []

        for detections in detections_list:
            for det in detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                all_confidences.append(det.confidence)

        # 보고서 작성
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("YOLO 객체 탐지 결과 요약 보고서\n")
            f.write("="*60 + "\n\n")

            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("전체 통계\n")
            f.write("-"*60 + "\n")
            f.write(f"처리 이미지 수: {total_images}\n")
            f.write(f"총 탐지 객체 수: {total_detections}\n")
            f.write(f"이미지당 평균 객체 수: {total_detections/total_images:.2f}\n\n")

            if all_confidences:
                f.write("신뢰도 통계\n")
                f.write("-"*60 + "\n")
                f.write(f"평균 신뢰도: {np.mean(all_confidences):.4f}\n")
                f.write(f"최대 신뢰도: {max(all_confidences):.4f}\n")
                f.write(f"최소 신뢰도: {min(all_confidences):.4f}\n\n")

            f.write("클래스별 탐지 통계\n")
            f.write("-"*60 + "\n")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100
                f.write(f"{class_name:20s}: {count:5d} ({percentage:5.2f}%)\n")

            f.write("\n" + "="*60 + "\n")

        print(f"요약 보고서 저장: {filepath}")

        return filepath
