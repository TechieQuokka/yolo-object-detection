"""Image preprocessing module."""

from typing import Tuple, Optional
import numpy as np
import cv2
import torch


class Preprocessor:
    """이미지 전처리 클래스."""

    def __init__(
        self,
        target_size: int = 640,
        normalize: bool = True,
        to_tensor: bool = True,
    ):
        """
        Args:
            target_size: 목표 이미지 크기 (정사각형)
            normalize: 정규화 여부 (0-255 → 0-1)
            to_tensor: PyTorch 텐서 변환 여부
        """
        self.target_size = target_size
        self.normalize = normalize
        self.to_tensor = to_tensor

    def resize(
        self,
        image: np.ndarray,
        target_size: Optional[int] = None,
        keep_ratio: bool = True,
    ) -> np.ndarray:
        """
        이미지 리사이즈.

        Args:
            image: 입력 이미지
            target_size: 목표 크기 (None이면 self.target_size 사용)
            keep_ratio: 종횡비 유지 여부

        Returns:
            리사이즈된 이미지
        """
        target_size = target_size or self.target_size
        h, w = image.shape[:2]

        if keep_ratio:
            # 종횡비 유지하면서 리사이즈
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # 패딩 추가
            pad_h = target_size - new_h
            pad_w = target_size - new_w
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            resized = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),  # 회색 패딩
            )
        else:
            # 강제 리사이즈
            resized = cv2.resize(
                image, (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )

        return resized

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 정규화 (0-255 → 0-1).

        Args:
            image: 입력 이미지

        Returns:
            정규화된 이미지
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image

    def to_torch_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        NumPy 배열을 PyTorch 텐서로 변환.

        Args:
            image: NumPy 이미지 (H, W, C)

        Returns:
            PyTorch 텐서 (C, H, W)
        """
        # HWC → CHW 변환
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))

        # NumPy → Tensor
        tensor = torch.from_numpy(image.copy())

        return tensor

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        전체 전처리 파이프라인.

        Args:
            image: 입력 이미지 (H, W, C)

        Returns:
            전처리된 텐서 (C, H, W) 또는 배열
        """
        # 1. 리사이즈
        processed = self.resize(image)

        # 2. 정규화
        if self.normalize:
            processed = self.normalize_image(processed)

        # 3. 텐서 변환
        if self.to_tensor:
            processed = self.to_torch_tensor(processed)

        return processed

    def preprocess_batch(self, images: list) -> torch.Tensor:
        """
        배치 이미지 전처리.

        Args:
            images: 이미지 리스트

        Returns:
            배치 텐서 (B, C, H, W)
        """
        processed_images = []

        for image in images:
            processed = self.preprocess(image)
            processed_images.append(processed)

        if self.to_tensor:
            # 배치 차원 추가
            batch_tensor = torch.stack(processed_images)
            return batch_tensor
        else:
            return np.array(processed_images)

    @staticmethod
    def denormalize(image: np.ndarray) -> np.ndarray:
        """
        정규화 해제 (0-1 → 0-255).

        Args:
            image: 정규화된 이미지

        Returns:
            원래 스케일 이미지
        """
        if image.dtype == np.float32 or image.dtype == np.float64:
            return (image * 255).astype(np.uint8)
        return image

    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        PyTorch 텐서를 NumPy 배열로 변환.

        Args:
            tensor: PyTorch 텐서 (C, H, W)

        Returns:
            NumPy 배열 (H, W, C)
        """
        # Tensor → NumPy
        array = tensor.cpu().numpy()

        # CHW → HWC 변환
        if array.ndim == 3:
            array = np.transpose(array, (1, 2, 0))

        return array

    def postprocess_for_display(self, tensor: torch.Tensor) -> np.ndarray:
        """
        디스플레이용 후처리 (텐서 → 이미지).

        Args:
            tensor: 전처리된 텐서

        Returns:
            디스플레이 가능한 이미지 (H, W, C), uint8
        """
        # 텐서 → NumPy
        image = self.tensor_to_numpy(tensor)

        # 정규화 해제
        image = self.denormalize(image)

        return image
