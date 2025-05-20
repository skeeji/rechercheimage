import os
import pickle
import json
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import tensorflow_hub as hub
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import webbrowser
import argparse

def main():
    parser = argparse.ArgumentParser(description='Recherche d\'images similaires basée sur des embeddings')
    parser.add_argument('--embeddings', required=True, type=str, help='Chemin vers le fichier d\'embeddings')
    parser.add_argument('--metadata', required=True, type=str, help='Chemin vers le fichier de métadonnées')
    parser.add_argument('--n_results', default=10, type=int, help='Nombre de résultats à afficher')
    parser.add_argument('--output', default='results.html', type=str, help='Chemin de sortie pour le fichier HTML des résultats')
    
    args = parser.parse_args()
    
    # === Chargement des données ===
    print("Chargement des embeddings...")
    with open(args.embeddings, "rb") as f:
        embeddings = pickle.load(f)

    print("Chargement des métadonnées...")
    with open(args.metadata, "r", encoding="utf-8") as f:
        luminaires = json.load(f)

    # === Chargement du modèle d'embedding ===
    print("Chargement du modèle...")
    # IMPORTANT: Utiliser le même modèle que celui qui a généré vos embeddings
    model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2"
    embedding_model = hub.KerasLayer(model_url)

    # === Sélection d'une image utilisateur ===
    root = tk.Tk()
    root.withdraw()
    query_image_path = filedialog.askopenfilename(title="Choisissez une image de luminaire", 
                                                 filetypes=[("Images", "*.jpg *.jpeg *.png *.gif *.bmp"),
                                                            ("Tous les fichiers", "*.*")])

    if not query_image_path:
        print("Aucune image sélectionnée.")
        return

    print(f"Image sélectionnée: {query_image_path}")

    def load_image(image_path, target_size=(224, 224)):
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(target_size)
            return np.array(img) / 255.0
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {image_path}: {e}")
            return None

    query_img = load_image(query_image_path)
    if query_img is None:
        print("Impossible de traiter l'image sélectionnée.")
        return
    
    query_img_batch = np.expand_dims(query_img, axis=0)
    
    # Créer un modèle complet pour prédiction
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
    hub_layer = embedding_model(input_layer)
    model = tf.keras.Model(input_layer, hub_layer)
    
    print("Extraction des caractéristiques de l'image requête...")
    query_features = model.predict(query_img_batch).flatten()

    # === Recherche des plus proches voisins ===
    print("Recherche des images similaires...")
    nn_model = NearestNeighbors(n_neighbors=args.n_results, metric="cosine")
    nn_model.fit(embeddings)
    distances, indices = nn_model.kneighbors([query_features])

    # === Génération HTML ===
    print("Génération des résultats...")
    html_path = args.output
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Résultats de recherche d'images similaires</title>
<style>
    body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
    h1, h2 { color: #333; }
    .query-image { margin-bottom: 20px; }
    .results { display: flex; flex-wrap: wrap; justify-content: center; }
    .result-item { margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; width: 200px; text-align: center; }
    img { max-height: 150px; max-width: 150px; }
</style>
</head>
<body>
""")
        f.write("<h1>Résultats de recherche d'images similaires</h1>\n")
        f.write("<div class='query-image'>\n")
        f.write("<h2>Image requête:</h2>\n")
        f.write(f'<img src="{query_image_path}" style="max-height:300px;"><br>\n')
        f.write("</div>\n")
        f.write("<h2>Images similaires:</h2>\n")
        f.write("<div class='results'>\n")
        
        for i, idx in enumerate(indices[0]):
            if idx < len(luminaires):  # Vérifier que l'indice est valide
                result = luminaires[idx]
                image_path = result.get("image_path", "")
                name = result.get("name", f"Image {idx}")
                
                if os.path.exists(image_path):
                    sim_score = (1 - distances[0][i]) * 100  # Convertir en pourcentage
                    f.write(f'<div class="result-item">')
                    f.write(f'<img src="{image_path}" style="max-height:200px;"><br>')
                    f.write(f"{name}<br>")
                    f.write(f"Similarité: {sim_score:.2f}%")
                    f.write("</div>\n")
                else:
                    print(f"Attention: L'image {image_path} n'existe pas.")
            else:
                print(f"Indice {idx} hors limites des métadonnées")
                
        f.write("</div>\n")
        f.write("</body></html>\n")

    # === Ouvrir automatiquement dans le navigateur ===
    print(f"Ouverture des résultats dans votre navigateur...")
    webbrowser.open('file://' + os.path.abspath(html_path))
    print(f"✅ Résultats enregistrés dans {html_path}")

if __name__ == "__main__":
    main()