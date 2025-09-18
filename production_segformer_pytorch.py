#!/usr/bin/env python3
"""
Production SegFormer B3 model using original PyTorch .pth file
Direct loading without TensorRT/ONNX optimization
"""

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import os


class ProductionSegFormerPyTorch:
    """
    Production SegFormer B3 model using original .pth weights
    - Direct PyTorch loading for debugging
    - Compatible with original TheDistanceAssessor interface
    """

    def __init__(self, model_path="/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth"):
        self.model_path = model_path
        self.num_labels = 13
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_pytorch_model()

    def _load_pytorch_model(self):
        """Load PyTorch model directly"""
        print(f"🚀 Loading PyTorch model: {self.model_path}")

        if os.path.exists(self.model_path):
            # Check if it's original model or our trained model
            if "SegFormer_B3_1024_finetuned" in self.model_path:
                # Original model - load complete model
                self.model = torch.load(self.model_path, map_location='cpu')
                print(f"✅ Loaded complete original model from {self.model_path}")
            else:
                # Our trained model - load state dict
                from transformers import SegformerForSemanticSegmentation, SegformerConfig

                # Create model with correct number of classes
                config = SegformerConfig.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
                config.num_labels = 13  # Our transfer learning model has 13 classes
                self.model = SegformerForSemanticSegmentation(config)

                # Load trained weights
                checkpoint = torch.load(self.model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                self.model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded trained model weights from {self.model_path}")
        else:
            print(f"❌ Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded on device: {self.device}")
        print(f"📊 Model config: num_labels = {self.model.config.num_labels if hasattr(self.model, 'config') else 'Unknown'}")

    def __call__(self, pixel_values):
        """Forward pass compatible with original SegFormer"""
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(self.device)
        else:
            pixel_values = torch.from_numpy(pixel_values).to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        return outputs

    def eval(self):
        """Compatibility method"""
        self.model.eval()
        return self

    @property
    def config(self):
        """Config compatibility"""
        return self.model.config


def load_model(model_path=None):
    """Load the PyTorch production model"""
    if model_path is None:
        # Use the new optimized model by default
        model_path = "/home/mmc-server4/RailSafeNet/models/segformer_b3_production_optimized_rail_0.7500.pth"
    return ProductionSegFormerPyTorch(model_path)


def load_pytorch_model():
    """Load PyTorch model"""
    return load_model()


if __name__ == "__main__":
    # Test the model
    print("🧪 Testing PyTorch production model...")
    model = load_model()

    # Test inference
    dummy_input = torch.randn(1, 3, 1024, 1024)
    output = model(dummy_input)

    print(f"✅ Output shape: {output.logits.shape}")
    print("🎯 PyTorch model ready for testing!")