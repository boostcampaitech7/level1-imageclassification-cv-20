import albumentations as A

from albumentations.pytorch import ToTensorV2

class SketchImageAugmentation:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]

        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 색상 조정 추가
            self.transform = A.Compose(
                [# Geometric transformations
                    A.Rotate(limit=10, p=0.5),
                    A.Affine(scale=(0.8, 1.2), shear=(-10, 10), p=0.5),
                    A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10, p=0.5),

                    # Morphological transformations
                    A.Erosion(kernel=(1, 2), p=0.5),
                    A.Dilation(kernel=(1, 2), p=0.5),

                    # Noise and blur
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.MotionBlur(blur_limit=(3, 7), p=0.5),

                    # Sketch-specific augmentations
                    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=255, p=0.5),

                    # Advanced techniques
                    A.OneOf([
                        A.AutoContrast(p=0.5),
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                    ], p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)