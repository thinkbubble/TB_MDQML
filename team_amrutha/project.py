import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.utils import to_categorical
from models import build_cnn_model


def build_image_label_mapping(raw_data_folder_path):
    image_to_label = {}

    for split_folder in ["seg_train", "seg_test"]:
        split_path = os.path.join(raw_data_folder_path, split_folder, split_folder)

        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)

            if not os.path.isdir(class_path):
                continue

            for filename in os.listdir(class_path):
                if filename.startswith("."):
                    continue
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_to_label[filename] = class_name

    return image_to_label


def load_cleaned_images_and_labels(cleaned_data_folder_path, image_to_label):
    X = []
    y = []

    for filename in os.listdir(cleaned_data_folder_path):
        if not filename.endswith(".npz"):
            continue
        if not filename.startswith("img_"):
            continue

        original_stem = filename.replace("img_", "").replace("_cleaned.npz", "")
        possible_names = [
            f"{original_stem}.jpg",
            f"{original_stem}.jpeg",
            f"{original_stem}.png",
        ]

        matched_label = None
        for possible_name in possible_names:
            if possible_name in image_to_label:
                matched_label = image_to_label[possible_name]
                break

        if matched_label is None:
            continue

        file_path = os.path.join(cleaned_data_folder_path, filename)
        data = np.load(file_path)
        image = data["image"]

        X.append(image)
        y.append(matched_label)

    return np.array(X, dtype="float32"), np.array(y)


def run_initial_classification_pipeline(raw_data_folder_path, cleaned_data_folder_path):
    image_to_label = build_image_label_mapping(raw_data_folder_path)
    X, y = load_cleaned_images_and_labels(cleaned_data_folder_path, image_to_label)

    print("Loaded samples:", len(X))

    if len(X) == 0:
        raise ValueError("No labeled cleaned images were loaded.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    model = build_cnn_model(X_train.shape[1:], y_train.shape[1])

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=32,
        verbose=1
    )

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    print("\nTest Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))
    # Save model
    model.save("team_amrutha/model_v1.h5")
    print("\nModel saved successfully!")