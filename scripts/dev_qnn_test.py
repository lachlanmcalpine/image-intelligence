"""First real test of the QNN execution provider against the Hexagon NPU.
Registers the QNN EP library, creates a session for the quantized image
tower with the HTP (NPU) backend, and compares output + latency against the
CPU execution provider.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn
from PIL import Image
from transformers import SiglipProcessor

ONNX_DIR = Path(__file__).resolve().parent.parent / "models" / "onnx"

print("registering QNN execution provider library...")
print("  library path:", onnxruntime_qnn.get_library_path())
print("  HTP backend path:", onnxruntime_qnn.get_qnn_htp_path())
ort.register_execution_provider_library("QNNExecutionProvider", onnxruntime_qnn.get_library_path())

print("\navailable providers after registration:", ort.get_available_providers())

processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
image = Image.new("RGB", (224, 224), (128, 64, 200))
pixel_values = processor(images=image, return_tensors="np")["pixel_values"]

model_path = str(ONNX_DIR / "siglip_image_int8.onnx")

print("\n=== CPU execution provider (baseline) ===")
cpu_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
for _ in range(3):
    t0 = time.perf_counter()
    cpu_out = cpu_sess.run(None, {"pixel_values": pixel_values})[0]
    print(f"  {(time.perf_counter()-t0)*1000:.1f}ms")

print("\n=== QNN execution provider (Hexagon HTP / NPU) ===")
so = ort.SessionOptions()
try:
    qnn_sess = ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=["QNNExecutionProvider"],
        provider_options=[{"backend_path": onnxruntime_qnn.get_qnn_htp_path()}],
    )
    print("session created successfully with QNN EP")
    for _ in range(5):
        t0 = time.perf_counter()
        qnn_out = qnn_sess.run(None, {"pixel_values": pixel_values})[0]
        print(f"  {(time.perf_counter()-t0)*1000:.1f}ms")

    diff = np.max(np.abs(cpu_out - qnn_out))
    print(f"\nmax abs diff (CPU EP vs QNN/NPU EP): {diff:.6f}")
except Exception as e:
    print(f"QNN EP FAILED: {type(e).__name__}: {e}")
