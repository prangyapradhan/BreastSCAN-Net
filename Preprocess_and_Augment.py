"""
BreastFusion-Net
Data Preprocessing and Augmentation

Pipeline:
1. N4 bias-field correction
2. Otsu-based breast ROI extraction
3. Non-Local Means (NLM) denoising
4. Percentile intensity normalization
5. CLAHE contrast enhancement
6. Resize to 224 x 224
7. Training-only data augmentation

Note:
Augmentation must be applied only after patient-wise train/validation
partitioning to prevent data leakage.
"""

import cv2
import numpy as np
from pathlib import Path
from skimage import exposure, morphology
from skimage.restoration import denoise_nl_means, estimate_sigma
import SimpleITK as sitk

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# 1. N4 BIAS FIELD CORRECTION
# ============================================================

def n4_bias_correction(image):
    """
    Apply N4 bias-field correction to a grayscale MRI image.
    """

    image = image.astype(np.float32)

    # Normalize to 0-255 for conversion
    image_norm = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    itk_image = sitk.GetImageFromArray(image_norm)

    # Mask for foreground region
    mask = sitk.OtsuThreshold(
        itk_image,
        0,
        1,
        200
    )

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    corrected = corrector.Execute(
        sitk.Cast(itk_image, sitk.sitkFloat32),
        mask
    )

    corrected = sitk.GetArrayFromImage(corrected)

    return corrected


# ============================================================
# 2. OTSU-BASED BREAST ROI EXTRACTION
# ============================================================

def extract_breast_roi(image):
    """
    Extract the foreground breast region using Otsu thresholding
    and morphological refinement.
    """

    image_uint8 = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Otsu threshold
    _, mask = cv2.threshold(
        image_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological refinement
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:
        largest_component = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        mask = np.where(
            labels == largest_component,
            255,
            0
        ).astype(np.uint8)

    roi = cv2.bitwise_and(
        image_uint8,
        image_uint8,
        mask=mask
    )

    return roi


# ============================================================
# 3. NON-LOCAL MEANS DENOISING
# ============================================================

def nlm_denoising(image):
    """
    Apply Non-Local Means denoising while preserving lesion boundaries.
    """

    image_float = image.astype(np.float32) / 255.0

    sigma_est = np.mean(
        estimate_sigma(
            image_float,
            channel_axis=None
        )
    )

    denoised = denoise_nl_means(
        image_float,
        h=1.15 * sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        channel_axis=None
    )

    denoised = np.clip(
        denoised * 255,
        0,
        255
    ).astype(np.uint8)

    return denoised


# ============================================================
# 4. PERCENTILE INTENSITY NORMALIZATION
# ============================================================

def percentile_normalization(image, lower=1, upper=99):
    """
    Normalize MRI intensity using percentile-based scaling.
    """

    image = image.astype(np.float32)

    low_value = np.percentile(
        image,
        lower
    )

    high_value = np.percentile(
        image,
        upper
    )

    image = np.clip(
        image,
        low_value,
        high_value
    )

    normalized = (
        image - low_value
    ) / (
        high_value - low_value + 1e-8
    )

    normalized = (
        normalized * 255
    ).astype(np.uint8)

    return normalized


# ============================================================
# 5. CLAHE CONTRAST ENHANCEMENT
# ============================================================

def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    return enhanced


# ============================================================
# 6. RESIZE TO 224 x 224
# ============================================================

def resize_image(image, size=(224, 224)):
    """
    Resize the preprocessed MRI image.
    """

    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_mri(image):
    """
    Complete BreastFusion-Net MRI preprocessing pipeline.

    Raw MRI
       ↓
    N4 correction
       ↓
    Otsu ROI extraction
       ↓
    NLM denoising
       ↓
    Percentile normalization
       ↓
    CLAHE
       ↓
    224 x 224
    """

    # Step 1: N4 bias correction
    image = n4_bias_correction(image)

    # Step 2: Breast ROI extraction
    image = extract_breast_roi(image)

    # Step 3: NLM denoising
    image = nlm_denoising(image)

    # Step 4: Percentile normalization
    image = percentile_normalization(image)

    # Step 5: CLAHE
    image = apply_clahe(image)

    # Step 6: Resize
    image = resize_image(image)

    return image


# ============================================================
# SAVE PREPROCESSED IMAGE
# ============================================================

def preprocess_file(input_path, output_path):

    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {input_path}"
        )

    processed = preprocess_mri(image)

    cv2.imwrite(
        str(output_path),
        processed
    )


# ============================================================
# TRAINING-ONLY AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(

    # Controlled rotation
    rotation_range=15,

    # Horizontal flipping
    horizontal_flip=True,

    # Zoom-based scaling
    zoom_range=0.15,

    # Brightness variation
    brightness_range=(0.85, 1.15),

    # Width/height translation
    width_shift_range=0.05,
    height_shift_range=0.05,

    # Contrast-related preprocessing
    preprocessing_function=None
)


# ============================================================
# GAUSSIAN NOISE
# ============================================================

def add_gaussian_noise(image, mean=0.0, std=0.02):
    """
    Add controlled Gaussian noise for training augmentation.
    """

    image = image.astype(np.float32) / 255.0

    noise = np.random.normal(
        mean,
        std,
        image.shape
    )

    noisy = image + noise

    noisy = np.clip(
        noisy,
        0.0,
        1.0
    )

    return (
        noisy * 255
    ).astype(np.uint8)


# ============================================================
# AUGMENTATION FUNCTION
# ============================================================

def augment_training_image(image):

    # Keras augmentation
    image = train_datagen.random_transform(
        image
    )

    # Gaussian noise
    image = add_gaussian_noise(
        image
    )

    return image


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    input_image = "data/raw/sample_mri.png"
    output_image = "data/preprocessed/sample_mri.png"

    Path(output_image).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    preprocess_file(
        input_image,
        output_image
    )

    print(
        "MRI preprocessing completed successfully."
    )
