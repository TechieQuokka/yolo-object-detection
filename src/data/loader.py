"""Image loading module."""

import os
from pathlib import Path
from typing import Union, List, Optional
import numpy as np
from PIL import Image
import cv2


class DataLoader:
    """이미지 로더 클래스."""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, use_cv2: bool = True):
        """
        Args:
            use_cv2: OpenCV 사용 여부 (True: cv2, False: PIL)
        """
        self.use_cv2 = use_cv2

    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        """
        단일 이미지 로딩.

        Args:
            path: 이미지 파일 경로

        Returns:
            RGB 형식의 numpy 배열 (H, W, C)

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            ValueError: 지원하지 않는 형식이거나 로딩 실패
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"지원하지 않는 이미지 형식입니다: {path.suffix}\n"
                f"지원 형식: {self.SUPPORTED_FORMATS}"
            )

        try:
            if self.use_cv2:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"이미지 로딩 실패: {path}")
                # BGR → RGB 변환
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = Image.open(path)
                image = np.array(image.convert("RGB"))

            return image

        except Exception as e:
            raise ValueError(f"이미지 로딩 중 오류 발생: {path}\n{str(e)}")

    def load_batch(self, paths: List[Union[str, Path]]) -> List[np.ndarray]:
        """
        여러 이미지 로딩.

        Args:
            paths: 이미지 파일 경로 리스트

        Returns:
            이미지 배열 리스트
        """
        images = []
        errors = []

        for path in paths:
            try:
                image = self.load_image(path)
                images.append(image)
            except Exception as e:
                errors.append((path, str(e)))

        if errors:
            error_msg = "\n".join([f"{path}: {error}" for path, error in errors])
            print(f"Warning: {len(errors)}개 이미지 로딩 실패:\n{error_msg}")

        return images

    def load_from_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
        max_images: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        디렉토리에서 이미지 로딩.

        Args:
            directory: 이미지 디렉토리 경로
            recursive: 하위 디렉토리 포함 여부
            max_images: 최대 로딩 이미지 수

        Returns:
            이미지 배열 리스트
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory}")

        if not directory.is_dir():
            raise ValueError(f"디렉토리가 아닙니다: {directory}")

        # 이미지 파일 찾기
        if recursive:
            image_files = [
                f
                for f in directory.rglob("*")
                if f.suffix.lower() in self.SUPPORTED_FORMATS
            ]
        else:
            image_files = [
                f
                for f in directory.iterdir()
                if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS
            ]

        # 정렬 및 제한
        image_files = sorted(image_files)
        if max_images:
            image_files = image_files[:max_images]

        print(f"디렉토리 '{directory}'에서 {len(image_files)}개 이미지 발견")

        return self.load_batch(image_files)

    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        이미지 유효성 검사.

        Args:
            image: numpy 배열 이미지

        Returns:
            유효 여부
        """
        if not isinstance(image, np.ndarray):
            return False

        if image.ndim not in [2, 3]:
            return False

        if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
            return False

        if image.size == 0:
            return False

        return True

    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """
        이미지 정보 추출.

        Args:
            image: numpy 배열 이미지

        Returns:
            이미지 메타데이터 딕셔너리
        """
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "height": image.shape[0],
            "width": image.shape[1],
            "channels": image.shape[2] if image.ndim == 3 else 1,
            "size": image.size,
            "min": image.min(),
            "max": image.max(),
        }
