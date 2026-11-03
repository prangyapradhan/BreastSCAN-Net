
"""
BreastSCAN-Net: MRI-only breast cancer detection
======================================================

Pipeline:
MRI image -> ResNet50 spatial features -> S-Transformer contextual learning
-> SE channel refinement -> global image embedding -> classification.

Includes:
- Stratified K-fold evaluation
- Grey Wolf Optimizer (GWO) hyperparameter search
- Grad-CAM heatmap generation
- Out-of-fold prediction and fold-metric saving

Important:
1. No clinical data, CFLANN, clinical embedding, or multimodal fusion.
2. TAG-GAN generation is not implemented because its generator and
   discriminator training code were not supplied. Connect the actual
   TAG-GAN pipeline through the training augmentation hook.
3. StratifiedKFold is image-level, not patient-level.
4. Input images must be RGB arrays in [0, 255].
5. Labels: 0 = benign, 1 = malignant.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    jaccard_score,
)


# ============================================================
# 1. REPRODUCIBILITY AND GLOBAL CONFIGURATION
# ============================================================

SEED = 42
IMG_SIZE = 224
IMAGE_CHANNELS = 3
IMAGE_EMBED_DIM = 512
NUM_CLASSES = 1

os.environ.setdefault("PYTHONHASHSEED", str(SEED))


def set_seed(seed: int = SEED) -> None:
    """Set Python, NumPy, and TensorFlow random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ============================================================
# 2. S-TRANSFORMER CONTEXTUAL FEATURE LEARNING
# ============================================================

class STransformer(layers.Layer):
    """
    Transformer encoder operating on spatial tokens from ResNet50.

    Input:  [batch, height, width, channels]
    Output: [batch, height, width, channels]
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        num_heads: int = 8,
        ff_dim: int = 4096,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name="spatial_multihead_attention",
        )

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation=tf.nn.gelu),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
            ],
            name="spatial_feed_forward",
        )

        self.dropout = layers.Dropout(dropout)

    def call(self, feature_map, training=False):
        shape = tf.shape(feature_map)
        batch_size, height, width = shape[0], shape[1], shape[2]

        tokens = tf.reshape(
            feature_map,
            [batch_size, height * width, self.embed_dim],
        )

        attention_output = self.attention(
            tokens, tokens, training=training
        )

        tokens = self.norm1(
            tokens + self.dropout(attention_output, training=training)
        )

        ffn_output = self.ffn(tokens, training=training)

        tokens = self.norm2(
            tokens + self.dropout(ffn_output, training=training)
        )

        return tf.reshape(
            tokens,
            [batch_size, height, width, self.embed_dim],
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config


# ============================================================
# 3. SQUEEZE-AND-EXCITATION CHANNEL REFINEMENT
# ============================================================

class SEBlock(layers.Layer):
    """Squeeze-and-excitation channel attention for spatial features."""

    def __init__(self, reduction: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden_dim = max(channels // self.reduction, 1)

        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(hidden_dim, activation="relu")
        self.fc2 = layers.Dense(channels, activation="sigmoid")

        super().build(input_shape)

    def call(self, feature_map):
        channel_descriptor = self.global_pool(feature_map)
        channel_weights = self.fc1(channel_descriptor)
        channel_weights = self.fc2(channel_weights)

        channel_weights = tf.reshape(
            channel_weights,
            [-1, 1, 1, tf.shape(feature_map)[-1]],
        )

        return feature_map * channel_weights

    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config


# ============================================================
# 4. IMAGE-ONLY BREASTFUSION-NET MODEL
# ============================================================

def build_breastfusion_net(
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.30,
    weight_decay: float = 1e-5,
    backbone_trainable: bool = False,
    transformer_dropout: float = 0.10,
) -> Model:
    """
    Build the image-only BreastFusion-Net.

    Input:
        MRI image tensor (224, 224, 3), RGB, range [0, 255].

    Output:
        Sigmoid probability P(malignant | MRI).
    """

    image_input = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS),
        name="mri_input",
    )

    # ResNet50 ImageNet preprocessing
    x = layers.Lambda(
        lambda tensor: preprocess_input(tensor),
        name="resnet50_preprocessing",
    )(image_input)

    # ResNet50 spatial feature extraction
    backbone = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS),
        name="resnet50_backbone",
    )

    backbone.trainable = backbone_trainable

    spatial_features = backbone(x)

    spatial_features = layers.Activation(
        "linear",
        name="resnet50_spatial_features",
    )(spatial_features)

    # S-Transformer contextual feature learning
    contextual_features = STransformer(
        embed_dim=2048,
        num_heads=8,
        ff_dim=4096,
        dropout=transformer_dropout,
        name="s_transformer",
    )(spatial_features)

    # SE channel refinement
    refined_features = SEBlock(
        reduction=8,
        name="se_channel_refinement",
    )(contextual_features)

    # Image feature embedding
    image_embedding = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(refined_features)

    image_embedding = layers.Dense(
        IMAGE_EMBED_DIM,
        activation="relu",
        name="image_feature_projection",
    )(image_embedding)

    image_embedding = layers.Dropout(
        dropout_rate,
        name="classifier_dropout_1",
    )(image_embedding)

    x = layers.Dense(
        256,
        activation="relu",
        name="classification_dense",
    )(image_embedding)

    x = layers.Dropout(
        dropout_rate * 0.67,
        name="classifier_dropout_2",
    )(x)

    output = layers.Dense(
        NUM_CLASSES,
        activation="sigmoid",
        name="malignancy_probability",
    )(x)

    model = Model(
        inputs=image_input,
        outputs=output,
        name="BreastFusionNet_MRI_Only",
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


# ============================================================
# 5. GREY WOLF OPTIMIZATION
# ============================================================

class GreyWolfOptimizer:
    """
    Generic Grey Wolf Optimizer.

    The objective function must return a scalar validation loss.
    The validation data must come from the development partition,
    never from the final independent test set.
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: Sequence[Tuple[float, float]],
        population_size: int = 5,
        iterations: int = 5,
        seed: int = SEED,
    ):
        self.objective_function = objective_function
        self.bounds = np.asarray(bounds, dtype=np.float32)
        self.population_size = int(population_size)
        self.iterations = int(iterations)
        self.dimension = len(self.bounds)
        self.rng = np.random.default_rng(seed)

        if self.population_size < 3:
            raise ValueError("GWO population_size must be at least 3.")

        if self.iterations < 1:
            raise ValueError("GWO iterations must be at least 1.")

        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError(
                "Each lower bound must be below its upper bound."
            )

    def initialize_population(self):
        return self.rng.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(self.population_size, self.dimension),
        ).astype(np.float32)

    def optimize(self):
        population = self.initialize_population()

        alpha = beta = delta = None
        alpha_score = beta_score = delta_score = np.inf

        for iteration in range(self.iterations):

            for wolf in population:
                score = float(self.objective_function(wolf.copy()))

                if not np.isfinite(score):
                    score = np.inf

                if score < alpha_score:
                    delta, delta_score = beta, beta_score
                    beta, beta_score = alpha, alpha_score
                    alpha, alpha_score = wolf.copy(), score

                elif score < beta_score:
                    delta, delta_score = beta, beta_score
                    beta, beta_score = wolf.copy(), score

                elif score < delta_score:
                    delta, delta_score = wolf.copy(), score

            if alpha is None:
                raise RuntimeError(
                    "GWO objective did not return a finite score."
                )

            a = 2.0 - 2.0 * iteration / max(self.iterations - 1, 1)

            for i in range(self.population_size):
                candidate_positions = []

                for leader in (alpha, beta, delta):

                    if leader is None:
                        leader = alpha

                    r1 = self.rng.random(self.dimension)
                    r2 = self.rng.random(self.dimension)

                    A = 2.0 * a * r1 - a
                    C = 2.0 * r2

                    D = np.abs(C * leader - population[i])

                    candidate_positions.append(leader - A * D)

                population[i] = np.mean(
                    candidate_positions,
                    axis=0,
                )

                population[i] = np.clip(
                    population[i],
                    self.bounds[:, 0],
                    self.bounds[:, 1],
                )

            print(
                f"GWO iteration {iteration + 1}/{self.iterations} | "
                f"best validation loss: {alpha_score:.6f}"
            )

        return alpha, alpha_score


def gwo_hyperparameter_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    population_size: int = 5,
    iterations: int = 5,
    search_epochs: int = 5,
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    GWO hyperparameter search:

    wolf[0] = log10(learning rate), range [-5, -3]
    wolf[1] = dropout rate,          range [0.10, 0.50]
    wolf[2] = log10(weight decay),   range [-7, -3]

    Keep the final independent test set outside this optimization.
    """

    def objective(wolf: np.ndarray) -> float:

        learning_rate = float(10 ** wolf[0])
        dropout_rate = float(wolf[1])
        weight_decay = float(10 ** wolf[2])

        candidate = build_breastfusion_net(
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
        )

        history = candidate.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=search_epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                )
            ],
            verbose=0,
        )

        best_val_loss = float(
            np.min(history.history["val_loss"])
        )

        tf.keras.backend.clear_session()

        return best_val_loss

    optimizer = GreyWolfOptimizer(
        objective_function=objective,
        bounds=[
            (-5.0, -3.0),
            (0.10, 0.50),
            (-7.0, -3.0),
        ],
        population_size=population_size,
        iterations=iterations,
        seed=SEED,
    )

    best_wolf, best_loss = optimizer.optimize()

    return {
        "learning_rate": float(10 ** best_wolf[0]),
        "dropout_rate": float(best_wolf[1]),
        "weight_decay": float(10 ** best_wolf[2]),
        "validation_loss": float(best_loss),
    }


# ============================================================
# 6. METRIC COMPUTATION
# ============================================================

def calculate_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Calculate binary classification metrics from probabilities."""

    y_true = np.asarray(y_true).astype(np.int32).ravel()

    probabilities = np.asarray(
        probabilities
    ).astype(np.float32).ravel()

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) else 0.0
    )

    try:
        auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(y_true, predictions),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall_sensitivity": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "specificity": specificity,

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "auc": auc,

        "mcc": matthews_corrcoef(
            y_true,
            predictions,
        ),

        "cohen_kappa": cohen_kappa_score(
            y_true,
            predictions,
        ),

        "jaccard": jaccard_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "confusion_matrix": np.array(
            [[tn, fp], [fn, tp]]
        ),

        "y_pred": predictions,
    }


# ============================================================
# 7. STRATIFIED K-FOLD EVALUATION
# ============================================================

def stratified_kfold_evaluation(
    images: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.30,
    weight_decay: float = 1e-5,
    backbone_trainable: bool = False,
    train_augmentation_fn: Optional[
        Callable[
            [np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray],
        ]
    ] = None,
    callbacks: Optional[
        Sequence[tf.keras.callbacks.Callback]
    ] = None,
    verbose: int = 1,
) -> Dict[str, object]:
    """
    Run stratified image-level K-fold evaluation.

    train_augmentation_fn is called separately on each training fold.
    It can connect a TAG-GAN augmentation pipeline.

    It must return:
        augmented_training_images,
        augmented_training_labels

    It must NOT modify validation images.

    This function does not guarantee patient-level separation.
    """

    images = np.asarray(images)
    labels = np.asarray(labels).astype(np.int32).ravel()

    if images.ndim != 4 or images.shape[1:] != (
        IMG_SIZE,
        IMG_SIZE,
        IMAGE_CHANNELS,
    ):
        raise ValueError(
            f"images must have shape "
            f"(N, {IMG_SIZE}, {IMG_SIZE}, {IMAGE_CHANNELS}); "
            f"received {images.shape}"
        )

    if len(images) != len(labels):
        raise ValueError(
            "The number of images and labels must match."
        )

    unique_labels = np.unique(labels)

    if not np.array_equal(
        unique_labels,
        np.array([0, 1]),
    ):
        raise ValueError(
            "Expected labels: 0=benign and 1=malignant."
        )

    if np.min(images) < 0 or np.max(images) > 255:
        raise ValueError(
            "Input images must be RGB pixel arrays in [0, 255]."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    oof_true = np.zeros(
        len(labels),
        dtype=np.int32,
    )

    oof_prob = np.zeros(
        len(labels),
        dtype=np.float32,
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(images, labels),
        start=1,
    ):

        print(
            f"\n========== Fold {fold}/{n_splits} =========="
        )

        x_train = images[train_idx]
        y_train = labels[train_idx]

        x_val = images[val_idx]
        y_val = labels[val_idx]

        # Optional TAG-GAN augmentation: training fold only.
        if train_augmentation_fn is not None:

            x_train, y_train = train_augmentation_fn(
                x_train,
                y_train,
            )

            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train).astype(
                np.int32
            ).ravel()

        model = build_breastfusion_net(
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            backbone_trainable=backbone_trainable,
        )

        fold_callbacks = (
            list(callbacks)
            if callbacks is not None
            else []
        )

        fold_callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        )

        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=fold_callbacks,
            verbose=verbose,
        )

        probabilities = model.predict(
            x_val,
            batch_size=batch_size,
            verbose=0,
        ).ravel()

        oof_true[val_idx] = y_val
        oof_prob[val_idx] = probabilities

        fold_metrics = calculate_binary_metrics(
            y_val,
            probabilities,
        )

        fold_metrics.pop("y_pred")

        fold_metrics["fold"] = fold

        fold_results.append(fold_metrics)

        print(
            f"Fold {fold}: "
            f"Accuracy={fold_metrics['accuracy']:.4f}, "
            f"F1={fold_metrics['f1']:.4f}, "
            f"AUC={fold_metrics['auc']:.4f}"
        )

        tf.keras.backend.clear_session()

    pooled_metrics = calculate_binary_metrics(
        oof_true,
        oof_prob,
    )

    print(
        "\n========== POOLED OUT-OF-FOLD RESULTS =========="
    )

    for metric_name in (
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "auc",
        "mcc",
        "cohen_kappa",
        "jaccard",
    ):
        print(
            f"{metric_name:20s}: "
            f"{pooled_metrics[metric_name]:.4f}"
        )

    print(
        "Confusion matrix:\n",
        pooled_metrics["confusion_matrix"],
    )

    return {
        "fold_results": fold_results,
        "y_true": oof_true,
        "y_probability": oof_prob,
        "y_pred": pooled_metrics["y_pred"],
        "pooled_metrics": pooled_metrics,
    }


# ============================================================
# 8. GRAD-CAM FOR THE MRI IMAGE BRANCH
# ============================================================

def make_gradcam_heatmap(
    image: np.ndarray,
    model: Model,
    conv_layer_name: str = "conv5_block3_out",
) -> np.ndarray:
    """
    Generate Grad-CAM for the final ResNet50 convolutional feature map.

    image:
        RGB MRI image with shape (224, 224, 3), range [0, 255].
    """

    backbone = model.get_layer(
        "resnet50_backbone"
    )

    conv_layer = backbone.get_layer(
        conv_layer_name
    )

    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            conv_layer.output,
            model.output,
        ],
    )

    image_tensor = tf.convert_to_tensor(
        np.expand_dims(image, axis=0),
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:

        conv_outputs, prediction = grad_model(
            image_tensor,
            training=False,
        )

        target_score = prediction[:, 0]

    gradients = tape.gradient(
        target_score,
        conv_outputs,
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients could not be computed for "
            "the selected convolutional layer."
        )

    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    conv_outputs = conv_outputs[0]
    channel_weights = channel_weights[0]

    heatmap = tf.reduce_sum(
        conv_outputs
        * channel_weights[tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


def overlay_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an RGB MRI image."""

    import cv2

    image_uint8 = np.clip(
        np.asarray(image),
        0,
        255,
    ).astype(np.uint8)

    resized_heatmap = cv2.resize(
        heatmap,
        (
            image_uint8.shape[1],
            image_uint8.shape[0],
        ),
    )

    heatmap_uint8 = np.uint8(
        255 * resized_heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        image_uint8,
        1.0 - alpha,
        colored_heatmap,
        alpha,
        0,
    )

    return overlay


# ============================================================
# 9. SAVE CROSS-VALIDATION RESULTS
# ============================================================

def save_oof_predictions(
    results: Dict[str, object],
    output_path: str = "results/oof_predictions.npz",
) -> None:
    """Save out-of-fold labels, predictions, and probabilities."""

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    np.savez_compressed(
        output_path,
        y_true=results["y_true"],
        y_pred=results["y_pred"],
        y_probability=results["y_probability"],
    )

    print(
        f"OOF predictions saved to: {output_path}"
    )


def save_fold_metrics(
    results: Dict[str, object],
    output_path: str = "results/fold_metrics.csv",
) -> None:
    """Save fold-wise metrics as a CSV file."""

    import csv

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    rows = results["fold_results"]

    fieldnames = [
        "fold",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "auc",
        "mcc",
        "cohen_kappa",
        "jaccard",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                }
            )

    print(
        f"Fold metrics saved to: {output_path}"
    )


# ============================================================
# 10. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    set_seed(SEED)

    print(
        "BreastFusion-Net MRI-only implementation loaded."
    )

    print(
        "Expected input: RGB MRI images, "
        "shape (N, 224, 224, 3), range [0,255]."
    )

    print(
        "Labels: 0=benign, 1=malignant."
    )

    print(
        "Prepare your dataset arrays and call "
        "stratified_kfold_evaluation(images, labels)."
    )

    # Example:
    #
    # images = np.load("data/mri_images.npy")
    # labels = np.load("data/labels.npy")
    #
    # results = stratified_kfold_evaluation(
    #     images,
    #     labels,
    #     n_splits=5,
    #     epochs=30,
    #     batch_size=16,
    # )
    #
    # save_oof_predictions(results)
    # save_fold_metrics(results)
