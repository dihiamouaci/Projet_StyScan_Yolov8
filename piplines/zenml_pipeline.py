import subprocess
import sys
from pathlib import Path

from zenml import pipeline, step

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@step
def prepare_dataset_step() -> None:
    """
    Étape 1 : préparation du dataset DeepFashion2.

    """
    print("Étape ZenML : Préparation du dataset...")

    subprocess.run(
        [sys.executable, "scripts/prepare_deepfashion.py"],
        cwd=PROJECT_ROOT,
        check=True
    )


@step
def train_model_step() -> None:
    """
    Étape 2 : entraînement du modèle YOLOv8.

    """
    print("Étape ZenML : Entraînement YOLOv8...")

    subprocess.run(
        [sys.executable, "scripts/train.py"],
        cwd=PROJECT_ROOT,
        check=True
    )


@step
def verify_outputs_step() -> None:
    """
    Étape 3 : vérification des fichiers générés.

    """
    print("Étape ZenML : Vérification des sorties...")

    runs_dir = PROJECT_ROOT / "runs" / "detect"

    if not runs_dir.exists():
        raise FileNotFoundError("Le dossier runs/detect est introuvable.")

    best_pt_files = list(runs_dir.rglob("best.pt"))
    best_onnx_files = list(runs_dir.rglob("best.onnx"))

    if not best_pt_files:
        raise FileNotFoundError("Aucun fichier best.pt trouvé.")

    print("Modèle PyTorch trouvé :", best_pt_files[-1])

    if best_onnx_files:
        print("Modèle ONNX trouvé :", best_onnx_files[-1])
    else:
        print("Aucun fichier ONNX trouvé pour le moment.")


@pipeline
def stylescan_training_pipeline():
    """
    Pipeline ZenML complet

    """
    prepare_dataset_step()
    train_model_step()
    verify_outputs_step()


if __name__ == "__main__":
    stylescan_training_pipeline()