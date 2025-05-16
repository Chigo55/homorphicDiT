import os
from pathlib import Path
from PIL import Image
import numpy as np


def contains_nan_in_image(image_path):
    try:
        img = Image.open(fp=image_path).convert("RGB")  # RGB로 변환
        img_array = np.array(object=img).astype(
            dtype=np.float32) / 255.0  # 0~1 범위로 정규화
        return np.isnan(img_array).any()
    except Exception as e:
        print(f"[에러] {image_path} 파일을 처리하는 중 오류 발생: {e}")
        return False


def check_images_for_nan(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    directory = Path(directory)
    image_paths = list(directory.rglob(pattern="*"))  # 모든 하위 디렉토리 포함 탐색

    nan_images = []

    for path in image_paths:
        if path.suffix.lower() in extensions:
            if contains_nan_in_image(image_path=path):
                print(f"[경고] NaN 값이 포함된 이미지 발견: {path}")
                nan_images.append(path)

    print(f"\n총 {len(nan_images)}개의 이미지에서 NaN이 발견되었습니다.")
    return nan_images


# 사용 예시
if __name__ == "__main__":
    for image_dir in Path("data/1_train").iterdir():  # 여기에 이미지 디렉토리 경로 지정
        print(image_dir)
        nan_files = check_images_for_nan(directory=image_dir)
