# 수식 기호 분할

수식 이미지를 기호별로 잘라 다음 CNN에 넘긴다. 수업에서 쓴 **고전 전처리**가 기본이며, **TensorFlow U-Net 이진화**도 선택할 수 있다. 두 경로 모두 같은 연결 성분 추출, 병합, crop 코드를 사용한다. 기호 분류와 LaTeX 생성은 별도 단계다.

입력에서 **분수는 `/`로 쓰고, 서로 다른 기호의 획은 닿거나 겹치지 않게 쓴다.** `=`의 두 줄이나 `i`의 점처럼 같은 기호의 여러 획은 떨어져 있어도 된다. 이 모듈은 기호 간 접촉을 자동으로 복구하지 않는다.

## 실행

Python 3.12와 uv가 필요하다. 샘플 수식과 가중치를 포함한다.

```powershell
uv sync --locked
uv run --locked ocr-segment segment examples/slash.png --profile rendered --method classical --output datasets/classical.npz --debug datasets/classical
uv run --locked ocr-segment segment examples/slash.png --profile rendered --method unet --output datasets/unet.npz --debug datasets/unet
```

두 명령에는 `python -m ocr_keras.segmentation.final`을 대신 쓸 수도 있다. 손글씨 입력에서는 `--profile rendered`를 빼면 손글씨용 병합 CNN을 사용한다. 고전 방식은 손글씨에 Otsu, 렌더링 이미지에는 밝기 180 미만 기준을 쓴다. `--otsu`를 주면 렌더링에서도 Otsu를 사용한다. U-Net은 검증에서 정한 0.65로 이진화한다. 크기 보정에는 두 방식에서 기존 Otsu를 사용한다.

출력 NPZ는 기호별 `images` (`float32 N×1×48×48`), `boxes` (전처리 이미지의 `x0, y0, x1, y1`), `geometry` (위치·크기 8개)를 담는다. 세 배열의 같은 행은 같은 기호다. `--debug`는 이진 마스크 `ink.png`와 기호 그룹 `segments.json`을 저장한다.

## 실험 재현

```powershell
$data = "datasets/rendered_slash_v1"
uv run --locked python -m ocr_keras.segmentation.rendered_dataset --output $data
uv run --locked python -m ocr_keras.segmentation.rendered_merge $data
uv run --locked ocr-segment evaluate --profile rendered --dataset $data --variants "$data/degraded" --split test --output "$data/compare.json"
```

병합기를 고정한 렌더링 시험 수식 180개의 기호 분할 F1:

| 방식 | 깨끗한 원본 | 흐림·그림자 합성본 |
|---|---:|---:|
| 고전: 고정 밝기 기준 | 95.73% | 19.54% |
| 고전: Otsu | 46.62% | 48.73% |
| TensorFlow U-Net | 88.07% | 40.18% |

[고정 밝기 비교](results/final_comparison.json)와 [Otsu 비교](results/final_comparison_otsu.json)를 포함한다. 실제 사진이나 최종 LaTeX 인식 정확도를 측정한 결과는 아니다.

U-Net 학습은 `python -m ocr_keras.segmentation.unet train`을 사용한다. 렌더링 데이터에서는 `--dataset $data --source-variants "$data/degraded" --mode gray --validate-both --output datasets/rendered_unet`로 별도 학습할 수 있다. 이후 새 가중치를 쓰려면 `--unet datasets/rendered_unet`과 검증에서 정한 `--foreground` 값을 지정한다. Linux에서 CUDA를 쓸 때는 `uv run --locked --extra gpu`에 `--device gpu`를 함께 사용한다.

포함된 U-Net은 CROHME InkML 획 마스크를 학습한 모델이다. 렌더링 데이터는 Matplotlib MathText 글꼴과 기호별 픽셀 소속으로 자체 생성한다. 원본 CROHME 자료와 생성 캐시는 포함하지 않았다. 구현과 평가 코드 작성에 생성형 AI를 보조로 활용했다.
