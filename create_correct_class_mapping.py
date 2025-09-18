#!/usr/bin/env python3
"""
Create correct class mapping based on original model analysis
原제작자 모델 분석을 통한 올바른 클래스 매핑 생성
"""

import numpy as np

def create_railsem19_to_original_mapping():
    """
    Create mapping from RailSem19 classes to original 13-class system
    Based on analysis of original model output
    """

    # RailSem19 classes (19 classes total)
    # From research: key classes are [3, 12, 17, 18] for rails
    railsem19_classes = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic_light",
        7: "traffic_sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle"
    }

    # Original model output analysis:
    # Class 4: Main rail tracks (빨간색 - 가장 큰 영역)
    # Class 9: Secondary rail tracks (빨간색 - 보조 영역)
    # Class 1: Rail road/platform (회색/연보라색)
    # Class 6: Vegetation/Sky (초록색)
    # Class 12: Background (분홍색 - 배경)

    # Create mapping: RailSem19 -> Original 13-class
    class_mapping = {
        # Rails mapping - most important
        3: 4,    # wall -> main rail track (Class 4)
        12: 9,   # rider -> secondary rail track (Class 9)
        17: 4,   # motorcycle -> main rail track (Class 4)
        18: 9,   # bicycle -> secondary rail track (Class 9)

        # Infrastructure
        0: 1,    # road -> rail road/platform (Class 1)
        1: 1,    # sidewalk -> rail road/platform (Class 1)
        9: 1,    # terrain -> rail road/platform (Class 1)

        # Environment
        8: 6,    # vegetation -> vegetation (Class 6)
        10: 6,   # sky -> sky/vegetation (Class 6)

        # Objects/Background
        2: 12,   # building -> background (Class 12)
        4: 12,   # fence -> background (Class 12)
        5: 12,   # pole -> background (Class 12)
        6: 12,   # traffic_light -> background (Class 12)
        7: 12,   # traffic_sign -> background (Class 12)
        11: 12,  # person -> background (Class 12)
        13: 12,  # car -> background (Class 12)
        14: 12,  # truck -> background (Class 12)
        15: 12,  # bus -> background (Class 12)
        16: 12,  # train -> background (Class 12)

        # Ignore class
        255: 255
    }

    print("🗺️  RailSem19 to Original 13-Class Mapping:")
    print("=" * 50)

    # Group by target class
    target_classes = {}
    for src, tgt in class_mapping.items():
        if tgt not in target_classes:
            target_classes[tgt] = []
        if src != 255:
            target_classes[tgt].append(src)

    class_names = {
        1: "Rail Road/Platform",
        4: "Main Rail Track",
        6: "Vegetation/Sky",
        9: "Secondary Rail Track",
        12: "Background"
    }

    for tgt_class in sorted(target_classes.keys()):
        if tgt_class != 255:
            src_classes = target_classes[tgt_class]
            class_name = class_names.get(tgt_class, f"Class {tgt_class}")
            print(f"Class {tgt_class:2d} ({class_name:20s}): {src_classes}")

    return class_mapping

def apply_class_mapping(mask, class_mapping):
    """Apply class mapping to mask"""
    mapped_mask = np.full_like(mask, 255, dtype=np.uint8)

    for original_class, new_class in class_mapping.items():
        if original_class != 255:
            mapped_mask[mask == original_class] = new_class

    return mapped_mask

if __name__ == "__main__":
    mapping = create_railsem19_to_original_mapping()
    print(f"\n📋 Total mappings: {len(mapping)}")
    print("✅ Class mapping created successfully!")