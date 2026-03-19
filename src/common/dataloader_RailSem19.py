"""RailSem19 계열 마스크를 DeepLabv3 학습용으로 읽어오는 데이터로더.

이 데이터로더는 원본 클래스 체계를 직접 사용하지 않고, 일부 클래스는 무시한 뒤
남는 클래스를 다시 번호 매겨 학습 입력을 구성한다. 이 재매핑 규칙은 학습/평가
스크립트와 결합되어 있으므로, 클래스 체계를 변경할 때는 관련 스크립트를 함께
검토해야 한다.
"""

from pathlib import Path
from torchvision.datasets.vision import VisionDataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import cv2
import re

class CustomDataset(VisionDataset):
    """DeepLabv3 경로에서 사용하는 이미지/마스크 쌍 데이터셋."""

    def __init__(self, image_folder, mask_folder, image_size, subset, val_fraction=0.1):
        self.image_folder = Path(image_folder)
        self.mask_folder = Path(mask_folder)
        self.val_fraction = val_fraction
        
        # Define data transformations using Albumentations
        if subset == 'Train': 
            self.transform_base = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                            A.OneOf([
                                A.RandomBrightnessContrast(p=1),
                                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
                            ], p=0.3),
                            A.OneOf([
                                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
                                A.RandomSunFlare(flare_roi=(0.5, 0.6, 0.7, 0.7), angle_lower=0.5, src_radius=350, p=1),
                                A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.2, snow_point_upper=0.8, p=1),
                                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.3, alpha_coef=0.3, p=1),
                            ], p=0.3),
                                A.OneOf([
                                    A.CoarseDropout(max_holes=100, max_height=30, max_width=30, min_height=10, min_width=10, min_holes=50, fill_value=0, p=1),
                                    A.GaussianBlur(blur_limit=(11, 21), p=1),
                                    A.GaussNoise(var_limit=(300.0, 650.0), mean=0, per_channel=True, p=1),
                                    A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.6, 0.9), p=1),
                                ], p=0.5),
                            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1.0)),
                            ])
        elif subset == 'Valid':
            self.transform_base = A.Compose([
                            A.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_NEAREST),
                            ])
        
        self.transform_img = A.Compose([
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                            ToTensorV2(p=1.0),
                            ])
        self.transform_mask = A.Compose([
                            ToTensorV2(p=1.0),
                            ])
        # all files
        self.image_list = np.array(sorted(Path(self.image_folder).glob("*")))
        self.mask_list = np.array(sorted(Path(self.mask_folder).glob("*")))

        for file_path in self.image_list:
            if 'desktop.ini' in file_path.name:
                file_path.unlink()
        for file_path in self.mask_list:
            if 'desktop.ini' in file_path.name:
                file_path.unlink()

        self.mask_list = np.array(sorted(self.mask_list, key=lambda path: int(re.findall(r'\d+', path.stem)[0]) if re.findall(r'\d+', path.stem) else 0))

        if subset == 'Train':  # split dataset to 1-fraction of train data, default fraction == 0.1
            self.image_names = self.image_list[:int(np.ceil(len(self.image_list) * (1 - self.val_fraction)))]
            self.mask_names = self.mask_list[:int(np.ceil(len(self.mask_list) * (1 - self.val_fraction)))]
        elif subset == 'Valid':  # val data - data of length fraction
            self.image_names = self.image_list[int(np.ceil(len(self.image_list) * (1 - self.val_fraction))):]
            self.mask_names = self.mask_list[int(np.ceil(len(self.mask_list) * (1 - self.val_fraction))):]
        else:
            print('Invalid data subset.')

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx):
        """증강된 이미지와 재매핑된 마스크를 반환한다.

        `ignore_list`에 포함된 클래스는 학습 대상에서 제외하며, 남은 클래스만
        0부터 다시 번호를 부여한다. 제외 클래스는 최종적으로 background
        역할의 `12`로 합친다.
        """
        image_path = self.image_names[idx]
        mask_path = self.mask_names[idx]

        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:

            image = cv2.imread(image_file.name)
            mask = cv2.imread(mask_file.name, cv2.IMREAD_GRAYSCALE)
        
            transformed = self.transform_base(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
            
            image = self.transform_img(image=transformed_image)['image']
            mask = self.transform_mask(image=transformed_mask)['image']

            # ignore not well segmented classes
            ignore = True
            if ignore:
                ignore_list = [0,1,2,6,8,9,15,16,19,20]
                for cls in ignore_list:
                    mask[mask==cls] = 255

                ignore_set = set(ignore_list)
                cls_remaining = [num for num in range(0, 22) if num not in ignore_set]

                # renumber the remaining classes 0-number of remaining classes
                for idx, cls in enumerate(cls_remaining):
                    mask[mask==cls] = idx

                mask[mask==255] = 12 # background

            sample = [image, mask.squeeze().long()]
            return sample
