# EmbeddingGemma-300M ONNX Quantized

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/jaehyun-kim/gemma-300m-onnx-quantized)

[google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) INT8 정적 양자화 ONNX 모델.  
불경(삼장) 시맨틱 검색을 위해 칼리브레이션 + 토크나이저 프루닝을 적용한 경량 모델입니다.

> INT8 static quantized ONNX model with Buddhist text calibration and pruned tokenizer for Tripitaka semantic search.
>
> **HuggingFace:** https://huggingface.co/jaehyun-kim/gemma-300m-onnx-quantized

---

## Overview / 개요

팔리어 삼장경(Tipitaka)을 한국어로 검색하기 위한 임베딩 모델입니다.  
[google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)을 ONNX로 변환하고, **불경 칼리브레이션 데이터**를 활용한 INT8 정적 양자화로 경량화했습니다.

MMTEB 벤치마크 500M 파라미터 이하 모델 중 **1위**이며, Matryoshka Representation Learning(MRL)을 지원하여 768d/512d/256d/128d 차원을 선택할 수 있습니다.

**주요 특징:**
- 한국어 질문 -> 영문/팔리어 경전 크로스링구얼 검색
- Matryoshka 지원: 768d / 512d / **256d** / 128d 선택 가능
- 불경 텍스트 칼리브레이션 기반 정적 양자화 (동적 양자화 대비 높은 품질)
- Flutter 앱 온디바이스 추론 가능 (onnxruntime)

---

## Changes / 변경 사항

| | 원본 (Original) | 양자화 (Quantized) |
|---|---|---|
| 포맷 | PyTorch | ONNX INT8 |
| 모델 크기 | ~1.2GB | ~200MB |
| 토크나이저 어휘 | 256,000 | ~120,000 |
| 임베딩 차원 | 768 (MRL: 256) | 768 (MRL: 256) |

### Quantization / 양자화
- **정적 양자화** (Static Quantization) with QDQ format
- 불경 텍스트 100건 칼리브레이션 (팔리 삼장 원문 기반)
- INT8 per-channel 가중치 양자화
- `onnxruntime.quantization.quantize_static` 사용

### Tokenizer Pruning / 토크나이저 프루닝
불경 검색에 불필요한 언어 토큰을 제거하여 토크나이저 크기를 축소했습니다.

**유지한 언어:**
| 언어 | 범위 | 용도 |
|---|---|---|
| Latin + diacritics | U+0000-U+036F, U+1E00-U+1EFF | 영어, 팔리어 로마자 표기 |
| 한글 | U+AC00-U+D7AF, U+3130-U+318F | 한국어 질문/검색 |
| CJK | U+4E00-U+9FFF, U+3400-U+4DBF | 한문 불경 |
| Devanagari | U+0900-U+097F | 산스크리트/팔리어 원문 |

**제거된 언어:** 아랍어, 태국어, 일본어(가나), 키릴 문자, 그루지아어 등 ~136,000 토큰

---

## Why EmbeddingGemma? / 왜 EmbeddingGemma인가?

| 모델 | 파라미터 | INT8 크기 | 차원 | MMTEB 순위 | 모바일 적합 |
|------|---------|----------|------|-----------|-----------|
| BGE-M3 | 568M | ~600MB | 1024 | - | X (너무 큼) |
| multilingual-e5-small | 118M | ~113MB | 384 | - | O |
| **EmbeddingGemma-300M** | **308M** | **~200MB** | **768 (MRL: 256)** | **500M 이하 1위** | **O** |

- 500M 이하 모델 중 **MMTEB 1위**
- **Matryoshka**: 256d로 쓰면 DB 용량 1/3, 검색 품질 유지
- **크로스링구얼**: 한국어 → 영어/팔리어 검색 최상급
- **모바일 최적화**: Google이 EdgeTPU/모바일 타겟으로 설계

---

## Usage / 사용법

### Python (onnxruntime)
```python
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

sess = ort.InferenceSession("model_quantized.onnx")
tok = AutoTokenizer.from_pretrained("./")

def embed(texts, dim=256):
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    inp = {n.name: enc[n.name].astype(np.int64) for n in sess.get_inputs() if n.name in enc}
    out = sess.run(None, inp)[0]
    mask = enc["attention_mask"][..., np.newaxis]
    pooled = (out * mask).sum(1) / mask.sum(1)
    # Matryoshka truncate
    pooled = pooled[:, :dim]
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

# 한국어 질문 -> 영문 경전 크로스링구얼 검색
query = embed(["고통의 원인은 무엇인가"])
passage = embed(["[SN56.11] Origin of Suffering: craving (tanha) leads to renewed existence."])
print(f"similarity: {np.dot(query[0], passage[0]):.4f}")
```

### Flutter (on-device)
```dart
// onnxruntime_flutter 패키지 사용
final session = await OrtSession.fromAsset('assets/models/gemma/model_quantized.onnx');
```

---

## Files / 파일 구성

| 파일 | 설명 |
|---|---|
| `model_quantized.onnx` | INT8 정적 양자화된 ONNX 모델 |
| `tokenizer.json` | 프루닝된 토크나이저 (라틴/한글/한자/데바나가리만 포함) |
| `tokenizer_config.json` | 토크나이저 설정 |
| `special_tokens_map.json` | 특수 토큰 매핑 |
| `config.json` | 모델 설정 |

---

## Reproduce / 재현 방법

Google Colab (T4 GPU)에서 실행:

```python
# 1. 설치
!pip install -q onnxruntime-gpu transformers huggingface_hub

# 2. ONNX 모델 다운로드 (PyTorch 없이)
from huggingface_hub import snapshot_download
snapshot_download('onnx-community/embeddinggemma-300m-ONNX', local_dir='./gemma-onnx')

# 3. 칼리브레이션 데이터 준비
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import onnxruntime as ort, numpy as np, json
from transformers import AutoTokenizer

with open('calibration_data.json', 'r') as f:
    calib = json.load(f)

tok = AutoTokenizer.from_pretrained('./gemma-onnx')
sess = ort.InferenceSession('./gemma-onnx/onnx/model.onnx')
input_names = [i.name for i in sess.get_inputs()]

class CalibReader(CalibrationDataReader):
    def __init__(self, texts, tokenizer, input_names):
        self.data = []
        for t in texts:
            enc = tokenizer(t, padding='max_length', truncation=True, max_length=512, return_tensors='np')
            self.data.append({n: enc[n].astype(np.int64) for n in input_names if n in enc})
        self.idx = 0
    def get_next(self):
        if self.idx >= len(self.data): return None
        d = self.data[self.idx]; self.idx += 1; return d
    def rewind(self):
        self.idx = 0

# 4. INT8 정적 양자화 (불경 칼리브레이션)
reader = CalibReader(calib[:100], tok, input_names)
quantize_static(
    model_input='./gemma-onnx/onnx/model.onnx',
    model_output='./quantized/model_quantized.onnx',
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    weight_type=QuantType.QInt8,
)

# 5. 토크나이저 프루닝
with open('./quantized/tokenizer.json', 'r') as f:
    t = json.load(f)
t['model']['vocab'] = [v for v in t['model']['vocab'] if
    not v[0].replace('\u2581','').strip() or v[0].startswith('<') or
    all(ord(c)<0x0370 or 0xAC00<=ord(c)<=0xD7AF or 0x3130<=ord(c)<=0x318F or
        0x4E00<=ord(c)<=0x9FFF or 0x3400<=ord(c)<=0x4DBF or 0x0900<=ord(c)<=0x097F
        for c in v[0].replace('\u2581','').strip())]
with open('./quantized/tokenizer.json', 'w') as f:
    json.dump(t, f, ensure_ascii=False)
```

---

## Use Case / 활용 사례

이 모델은 [구다경(Guda Sutra)](https://github.com/kimdzhekhon/Guda-Sutra) Flutter 앱에서 사용됩니다.

- 사용자가 한국어로 질문 (예: "고통의 원인은 무엇인가")
- 온디바이스 EmbeddingGemma 양자화 모델로 질문 임베딩 생성 (256d)
- Supabase pgvector에서 유사 경전 검색 (32,535개 팔리 삼장 청크)
- 검색된 경전을 컨텍스트로 LLM 답변 생성

## Base Model / 베이스 모델

- **모델:** [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
- **아키텍처:** Gemma 기반 308M 파라미터
- **학습:** Matryoshka Representation Learning (MRL)
- **ONNX 변환:** [onnx-community/embeddinggemma-300m-ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)

## License / 라이선스

[google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)과 동일 (Gemma License).

## Acknowledgements / 감사

- [Google](https://huggingface.co/google/embeddinggemma-300m) - EmbeddingGemma 베이스 모델
- [ONNX Community](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) - ONNX 변환
- [ONNX Runtime](https://onnxruntime.ai/) - 양자화 프레임워크
- [SuttaCentral](https://suttacentral.net/) - 팔리 삼장 원문 데이터
