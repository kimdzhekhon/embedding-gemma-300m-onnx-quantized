<div align="center">

# 🧠 EmbeddingGemma-300M ONNX Quantized

**불경 시맨틱 검색을 위한 경량화 임베딩 모델** — INT8 정적 양자화 ONNX 모델

[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/jaehyun-kim/gemma-300m-onnx-quantized)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

</div>

---

## 🌟 개요

[google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)을 팔만대장경(Tripitaka) 시맨틱 검색에 최적화한 INT8 정적 양자화 ONNX 모델입니다. Flutter 앱 온디바이스 추론에 사용됩니다.

**HuggingFace:** https://huggingface.co/jaehyun-kim/gemma-300m-onnx-quantized

## 🛠 기술 스택 / 최적화 내용

| 항목 | 원본 | 최적화 후 |
|------|------|---------|
| **포맷** | PyTorch | ONNX |
| **양자화** | FP32 | INT8 정적 양자화 |
| **칼리브레이션** | - | 불경 텍스트 기반 |
| **토크나이저** | 원본 어휘 | 불경 특화 프루닝 |
| **지원 차원** | 768d | 768/512/256/128d (MRL) |

## 🔍 핵심 기술 상세

### INT8 정적 양자화
동적 양자화와 달리, 불경 칼리브레이션 데이터셋으로 활성화 범위를 사전 측정하여 정적으로 양자화합니다. 도메인 분포를 반영한 양자화로 동적 대비 높은 품질을 유지합니다.

### Matryoshka Representation Learning (MRL)
768d 전체 임베딩의 앞부분 N차원만 잘라도 의미 있는 표현이 되는 중첩 임베딩 구조. 앱에서 256d를 사용하여 메모리와 검색 속도를 최적화합니다.

### 토크나이저 프루닝
원본 어휘 중 한국어·영어·팔리어에 불필요한 토큰을 제거하여 토크나이저를 경량화, 임베딩 행렬 크기를 줄입니다.

### 온디바이스 추론
Flutter `onnxruntime_v2` 패키지로 Android/iOS에서 서버 없이 직접 추론합니다.