#!/usr/bin/env python3
"""YOLO 객체 탐지 실행 스크립트"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import PRESET_CONFIGS
from src.data.loader import DataLoader
from src.model.detector import YOLODetector
from src.output.visualizer import Visualizer
from src.output.exporter import ResultExporter


def main():
    """메인 실행 함수"""

    print("\n" + "="*70)
    print("YOLO 객체 탐지 시스템")
    print("="*70 + "\n")

    # 1. 설정 및 모델 초기화
    print("📦 [1/5] 모델 초기화...")
    config = PRESET_CONFIGS["fast"]
    detector = YOLODetector(config)
    print(f"   ✓ 모델: {config.model_name}")
    print(f"   ✓ 디바이스: {config.device}")
    print(f"   ✓ 신뢰도 임계값: {config.conf_threshold}")

    # 2. 데이터 로더 초기화
    print("\n📂 [2/5] 데이터 로더 초기화...")
    loader = DataLoader(use_cv2=True)
    print("   ✓ 데이터 로더 준비 완료")

    # 3. 이미지 로딩
    print("\n🖼️  [3/5] 이미지 로딩...")
    image_path = project_root / "data/samples/bus.jpg"

    if not image_path.exists():
        print(f"   ✗ 이미지를 찾을 수 없습니다: {image_path}")
        print("   먼저 샘플 이미지를 다운로드하세요:")
        print("   curl -o data/samples/bus.jpg https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg")
        return

    image = loader.load_image(image_path)
    image_info = loader.get_image_info(image)
    print(f"   ✓ 파일: {image_path.name}")
    print(f"   ✓ 크기: {image_info['width']}x{image_info['height']}")
    print(f"   ✓ 채널: {image_info['channels']}")

    # 4. 객체 탐지
    print("\n🔍 [4/5] 객체 탐지 실행...")
    detections = detector.predict(image, verbose=False)

    print(f"   ✓ 탐지 완료: {len(detections)}개 객체 발견\n")

    if detections:
        print("   탐지된 객체:")
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det.bbox
            print(f"     {i}. {det.class_name:15s} - 신뢰도: {det.confidence:.3f} - 위치: ({int(x1)},{int(y1)}) → ({int(x2)},{int(y2)})")
    else:
        print("   ⚠️  탐지된 객체가 없습니다")

    # 5. 결과 시각화 및 저장
    print("\n💾 [5/5] 결과 저장...")

    # 시각화
    visualizer = Visualizer(
        line_thickness=3,
        font_scale=0.7,
        hide_labels=False,
        hide_conf=False
    )
    annotated_image = visualizer.draw_detections(image, detections)

    # 결과 저장
    exporter = ResultExporter(output_dir=project_root / "data/results")

    img_path = exporter.save_image(
        annotated_image,
        filename=f"detected_{image_path.name}"
    )

    json_path = exporter.save_detections_json(
        detections,
        filename=f"result_{image_path.stem}.json",
        metadata={
            "source": str(image_path),
            "model": config.model_name,
            "image_size": f"{image_info['width']}x{image_info['height']}"
        }
    )

    print(f"   ✓ 이미지: {img_path.name}")
    print(f"   ✓ JSON: {json_path.name}")

    # 통계 요약
    print("\n" + "="*70)
    print("📊 탐지 결과 통계")
    print("="*70)

    summary = visualizer.create_detection_summary(detections)
    print(f"총 객체 수: {summary['total']}")

    if summary['by_class']:
        print("\n클래스별 분포:")
        for class_name, count in summary['by_class'].items():
            percentage = (count / summary['total']) * 100
            print(f"  • {class_name:15s}: {count:2d}개 ({percentage:5.1f}%)")

    if summary['total'] > 0:
        print(f"\n신뢰도:")
        print(f"  • 평균: {summary['avg_confidence']:.3f}")
        print(f"  • 최대: {summary['max_confidence']:.3f}")
        print(f"  • 최소: {summary['min_confidence']:.3f}")

    print("\n" + "="*70)
    print("✅ 완료!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
