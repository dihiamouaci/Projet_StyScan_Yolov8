# StyleScan par MOUACI DIHIA ET HADJEM SONIA

Projet de détection multi-classe de vêtements avec YOLOv8, DeepFashion2, FastAPI et webcam.

## Objectif

Le système détecte plusieurs catégories de vêtements :
- Dress
- Jacket
- Pants
- Skirt
- Tee

Pour chaque détection, il retourne :
- la catégorie
- la couleur dominante
- le score de confiance
- la bounding box

Pour démarrer le Projet y a 2 méthodes soit:
## Exécution de zen Ml seulement
```bash 
pyhton zenml_pipeline.py
```
Ou veuillez suivre Ces Etapes
## Etape 1
## Installation:

```bash
venv\Scripts\activate
pip install -r requirements.txt
```



## Étape 2:

```bash
python scripts/inspect_dataset.py
python scripts/prepare_deepfashion.py
python scripts/train.py
python utils/webcam.py
```

## API

```bash
uvicorn api.main:app --reload
```


