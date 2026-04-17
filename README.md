<div align="center">

# Embedding Gemma 300M — ONNX INT8 Static Quantized

팔만대장경 의미 검색을 위한 MRL 지원 경량화 임베딩 모델 (768/512/256/128차원)

![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow?style=for-the-badge)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-green?style=for-the-badge)
![Android](https://img.shields.io/badge/Android-On--Device-blue?style=for-the-badge)

![Version](https://img.shields.io/badge/version-1.0.0-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Quantization](https://img.shields.io/badge/quantization-INT8_Static-orange?style=flat-square)

**[HuggingFace 모델 보기 →](https://huggingface.co/jaehyun-kim/gemma-300m-onnx-quantized)**

</div>

---

## 목차

1. [소개](#소개)
2. [주요 기능](#주요-기능)
3. [기술 스택 / 최적화 내역](#기술-스택--최적화-내역)
4. [아키텍처 / 구현 원리](#아키텍처--구현-원리)
5. [데이터 흐름](#데이터-흐름)
6. [설치 및 사용](#설치-및-사용)
7. [Flutter 통합](#flutter-통합)
8. [Roadmap](#roadmap)
9. [라이선스](#라이선스)

---

## 소개

google/embedding-gemma-300m을 팔만대장경 의미 검색에 최적화한 ONNX Static INT8 양자화 모델입니다. 불경 텍스트 코퍼스로 수집한 칼리브레이션 데이터셋을 활용하여 활성화 범위를 사전 측정하고, 이를 바탕으로 Static INT8 양자화를 수행하여 도메인 특화 정확도를 최대한 유지합니다. MRL(Matryoshka Representation Learning)을 지원하여 768/512/256/128차원을 유연하게 선택할 수 있으며, 앱에서는 256차원을 기본으로 사용하여 속도와 메모리를 최적화합니다.

> 불경 텍스트로 칼리브레이션된 Static INT8 양자화는 일반 Dynamic 양자화 대비 도메인 내 검색 정확도가 더 높습니다.

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| Static INT8 양자화 | 불경 칼리브레이션 데이터 기반으로 활성화 범위 사전 측정 |
| MRL 지원 | 768/512/256/128차원 중 앱 요구사항에 맞게 선택 가능 |
| 도메인 특화 최적화 | 팔만대장경 텍스트로 칼리브레이션하여 불경 검색 성능 극대화 |
| 어휘 프루닝 | 한국어·영어·팔리어 토큰 보존으로 모델 경량화 |
| 온디바이스 추론 | onnxruntime_v2 Flutter 패키지로 Android 온디바이스 실행 |
| 유연한 차원 선택 | 256차원 기본 사용으로 속도·메모리 최적화 |

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 기술 스택 / 최적화 내역

| 항목 | 내용 |
|---|---|
| 베이스 모델 | google/embedding-gemma-300m |
| 원본 형식 | FP32 PyTorch |
| 양자화 방식 | Static INT8 (활성화 범위 사전 측정) |
| 칼리브레이션 | 팔만대장경 불경 텍스트 코퍼스 |
| 변환 도구 | ONNX 변환 + onnxruntime.quantization |
| 어휘 보존 | 한국어 / 영어 / 팔리어 |
| 임베딩 차원 | 768 / 512 / 256 / 128d (MRL) |
| 앱 기본 차원 | 256d |
| 추론 런타임 | onnxruntime_v2 (Flutter/Android) |

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 아키텍처 / 구현 원리

```
[Embedding Gemma 300M ONNX Static INT8 아키텍처]

입력 텍스트 (한국어 / 팔리어)
    │
    ▼
┌─────────────────────────────────────┐
│  SentencePiece 토크나이저            │
│  - 한/영/팔리어 어휘 보존             │
│  - 서브워드 분리                      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  ONNX Gemma 인코더 (Static INT8)    │
│  - 가중치·활성화 모두 INT8 양자화     │
│  - 불경 코퍼스 칼리브레이션 적용      │
│  - 300M 파라미터 경량 Gemma          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  MRL 임베딩 출력                      │
│  - 768d 전체                          │
│  - 512d / 256d / 128d 서브셋 사용    │
│  - 앱 기본: 앞 256d                   │
└─────────────────────────────────────┘
```

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 데이터 흐름

```
원본 google/embeddinggemma-300m (FP32)
      │
      ▼
불경 텍스트 칼리브레이션 데이터셋 수집
      │
      ▼
Static INT8 양자화 (활성화 범위 사전 측정)
      │
      ▼
어휘 프루닝 (한/영/팔리어 보존)
      │
      ▼
ONNX 변환 + MRL 지원 (768/512/256/128d)
      │
      ▼
Flutter onnxruntime_v2 통합 → Android 배포
```

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 설치 및 사용

**요구사항**

- Python 3.8+
- `onnxruntime >= 1.16`
- `transformers >= 4.35`
- `optimum[onnxruntime]`

```bash
# HuggingFace Hub에서 모델 다운로드
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="jaehyun-kim/gemma-300m-onnx-quantized",
    local_dir="./model"
)
EOF
```

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model")
session = ort.InferenceSession("./model/model.onnx")

text = "반야바라밀다심경 관자재보살"
inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)

outputs = session.run(None, {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
})

embedding_768 = outputs[0][0]           # 768차원 전체
embedding_256 = embedding_768[:256]     # 앞 256차원 (앱 기본 설정)
embedding_128 = embedding_768[:128]     # 앞 128차원 (최경량)
```

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## Flutter 통합

Flutter Android 앱에서 `onnxruntime_v2` 패키지로 온디바이스 추론하는 방법입니다. 앱은 기본적으로 256차원을 사용합니다.

```yaml
# pubspec.yaml
dependencies:
  onnxruntime_v2: ^최신버전
```

```dart
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

// 모델 로드
final session = OrtSession.fromFile(
  'assets/model.onnx',
  OrtSessionOptions(),
);

// 토큰화된 입력 준비
final List<int> tokenIds = /* 토크나이저 출력 */;
final int seqLen = tokenIds.length;

final inputs = {
  'input_ids': OrtValueTensor.createTensorWithDataList(
    tokenIds,
    [1, seqLen],
  ),
  'attention_mask': OrtValueTensor.createTensorWithDataList(
    List.filled(seqLen, 1),
    [1, seqLen],
  ),
};

// 추론 실행
final outputs = session.run(null, inputs);
final embedding = outputs[0]?.value as List<List<double>>;

// MRL: 256차원만 사용하여 속도·메모리 최적화
final embedding256 = embedding[0].sublist(0, 256);

// 필요 시 더 낮은 차원 선택
final embedding128 = embedding[0].sublist(0, 128);
```

> MRL 덕분에 앱에서 정확도와 속도 사이의 균형을 런타임에 선택할 수 있습니다. 256차원은 충분한 의미 표현력과 빠른 코사인 유사도 계산을 동시에 제공합니다.

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## Roadmap

- [x] Static INT8 양자화 (불경 칼리브레이션)
- [x] 불경 텍스트 칼리브레이션 데이터셋 구축
- [x] MRL 지원 (768/512/256/128d)
- [x] 어휘 프루닝 (한/영/팔리어 보존)
- [x] Flutter onnxruntime_v2 온디바이스 통합
- [ ] Dynamic INT8 버전 비교 (Static vs Dynamic 성능 벤치마크)
- [ ] FP16 버전 추가
- [ ] 벤치마크 문서 (속도·정확도·MRL 차원별 성능)
- [ ] 범용 한국어 도메인 파인튜닝

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 라이선스

MIT License

Copyright (c) 2026 kimdzhekhon

본 저장소의 최적화 코드 및 변환 스크립트는 MIT 라이선스로 배포됩니다. 베이스 모델(google/embedding-gemma-300m)의 라이선스는 [HuggingFace 모델 카드](https://huggingface.co/jaehyun-kim/gemma-300m-onnx-quantized)를 확인하세요.

<div align="right"><a href="#목차">↑ 맨 위로</a></div>
