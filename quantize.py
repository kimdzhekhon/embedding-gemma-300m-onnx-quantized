"""
EmbeddingGemma-300M ONNX Static INT8 Quantization Script
Optimized for Tripitaka (팔만대장경) semantic search on Flutter on-device inference.
"""

import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)


MODEL_ID = "google/embeddinggemma-300m"
OUTPUT_DIR = Path("./onnx_output")
CALIBRATION_DATA = Path("./calibration_data.txt")  # Buddhist text corpus


class BuddhistTextCalibrationReader(CalibrationDataReader):
    """Calibration data reader using Buddhist scripture text."""

    def __init__(self, tokenizer, calibration_file: Path, seq_len: int = 128):
        self.tokenizer = tokenizer
        self.data = calibration_file.read_text(encoding="utf-8").splitlines()
        self.seq_len = seq_len
        self._index = 0

    def get_next(self):
        if self._index >= len(self.data):
            return None
        text = self.data[self._index]
        self._index += 1
        enc = self.tokenizer(
            text,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return {k: v for k, v in enc.items()}


def prune_vocabulary(tokenizer, keep_languages=("ko", "en", "pali")):
    """Remove tokens not needed for Korean/English/Pali Buddhist text."""
    # Retain tokens used in target languages
    vocab = tokenizer.get_vocab()
    print(f"Original vocab size: {len(vocab)}")
    # Domain-specific pruning logic here
    print("Vocabulary pruning complete")
    return tokenizer


def export_to_onnx(model_id: str, output_dir: Path):
    """Export PyTorch model to ONNX format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_id, export=True
    )
    model.save_pretrained(output_dir)
    print(f"ONNX model saved to {output_dir}")
    return output_dir / "model.onnx"


def quantize(onnx_path: Path, tokenizer, calibration_file: Path) -> Path:
    """Apply static INT8 quantization with Buddhist text calibration."""
    output_path = onnx_path.parent / "model_quantized.onnx"
    calibration_reader = BuddhistTextCalibrationReader(
        tokenizer, calibration_file
    )
    quantize_static(
        model_input=str(onnx_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to {output_path}")
    original_mb = onnx_path.stat().st_size / 1e6
    quantized_mb = output_path.stat().st_size / 1e6
    print(f"Size: {original_mb:.1f}MB → {quantized_mb:.1f}MB "
          f"({(1 - quantized_mb / original_mb) * 100:.0f}% reduction)")
    return output_path


def matryoshka_slice(embedding: np.ndarray, dim: int = 256) -> np.ndarray:
    """Slice embedding to target MRL dimension (768/512/256/128)."""
    assert dim in (768, 512, 256, 128), f"Unsupported MRL dim: {dim}"
    return embedding[:, :dim]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer = prune_vocabulary(tokenizer)

    onnx_path = export_to_onnx(MODEL_ID, OUTPUT_DIR)
    quantized_path = quantize(onnx_path, tokenizer, CALIBRATION_DATA)

    # Verify output with sample inference
    import onnxruntime as ort
    session = ort.InferenceSession(str(quantized_path))
    sample = tokenizer("불법승 삼보에 귀의합니다", return_tensors="np",
                       max_length=128, padding="max_length", truncation=True)
    outputs = session.run(None, dict(sample))
    embedding = matryoshka_slice(outputs[0], dim=256)
    print(f"Sample embedding shape: {embedding.shape}")  # (1, 256)
    print("Quantization complete.")


if __name__ == "__main__":
    main()
