import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

def export_and_quantize():
    # Define paths
    model_id = r"../models/sentimentv1"
    onnx_path = Path(r"../onnx_model")
    quant_path = Path(r"../onnx_model_int8")
    
    # Ensure output directories exist
    onnx_path.mkdir(parents=True, exist_ok=True)
    quant_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Equivalent to: optimum-cli export onnx
    # ---------------------------------------------------------
    print("Exporting base model to ONNX (FP32)...")
    
    # Load and save the model in ONNX format for text-classification
    model = ORTModelForSequenceClassification.from_pretrained(
        model_id, 
        export=True,
        task="text-classification"
    )
    model.save_pretrained(onnx_path)
    
    # Save the tokenizer alongside the ONNX model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(onnx_path)
    print(f"Base ONNX model successfully saved to: {onnx_path}")

    # ---------------------------------------------------------
    # 2. INT8 Dynamic Quantization
    # ---------------------------------------------------------
    print("\nQuantizing ONNX model to INT8...")
    
    # Explicitly point to the exported onnx model file name inside the path
    quantizer = ORTQuantizer.from_pretrained(onnx_path, file_name="model.onnx")
    
    # Use standard dynamic configuration to prevent avx512 attribute errors
    dq_config = AutoQuantizationConfig.arm64(
        is_static=False, 
        per_channel=False
    )
    
    # Export the quantized model and target a distinct file name
    quantizer.quantize(
        quantization_config=dq_config,
        save_dir=quant_path,
        file_suffix="quantized"
    )
    
    # Copy tokenizer to the quantized folder for deployment readiness
    tokenizer.save_pretrained(quant_path)
    print(f"INT8 Quantized model successfully saved to: {quant_path}")

if __name__ == "__main__":
    export_and_quantize()
