import os
import pickle
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow as tf
import argparse
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration pour éviter les messages de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description='Génération des embeddings pour les luminaires')
    parser.add_argument('--image_dir', type=str, default='data/images', 
                        help='Dossier contenant les images de luminaires')
    parser.add_argument('--output_dir', type=str, default='models', 
                        help='Dossier de sortie pour les embeddings et métadonnées')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Taille des batchs pour le traitement')
    parser.add_argument('--workers', type=int, default=8,
                        help='Nombre de workers pour charger les images')
    return parser.parse_args()

def load_image(image_path, target_size=(224, 224)):
    """Charge et prétraite une image pour l'inférence"""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        return np.array(img) / 255.0
    except Exception as e:
        print(f"⚠️ Erreur chargement {image_path}: {e}")
        return None

def process_batch(image_paths, image_processor, model):
    """Traite un batch d'images et retourne leurs embeddings"""
    valid_images = []
    valid_paths = []
    
    # Prétraiter les images
    for path in image_paths:
        img = image_processor(path)
        if img is not None:
            valid_images.append(img)
            valid_paths.append(path)
    
    if not valid_images:
        return [], []
    
    # Convertir en batch et prédire
    image_batch = np.array(valid_images)
    embeddings = model(image_batch).numpy()
    
    # Normaliser les vecteurs
    normalized_embeddings = []
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        if norm > 0:
            normalized_embeddings.append(emb / norm)
        else:
            normalized_embeddings.append(emb)
    
    return normalized_embeddings, valid_paths

def extract_metadata(image_path):
    """Extrait les métadonnées à partir du chemin de l'image"""
    file_name = os.path.basename(image_path)
    image_id = os.path.splitext(file_name)[0]
    
    # Simuler un prix basé sur l'ID (pour exemple)
    # Dans un cas réel, vous extrairiez les données d'une base de données
    price_seed = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 1000
    price = 50 + (price_seed / 10)  # Prix entre 50€ et 150€
    
    # Générer des catégories et styles simulés
    categories = ["Suspension", "Lampadaire", "Applique", "Lampe de table", "Plafonnier"]
    styles = ["Moderne", "Industriel", "Scandinave", "Art déco", "Vintage", "Minimaliste"]
    materials = ["Métal", "Bois", "Verre", "Tissu", "Céramique", "Plastique"]
    
    category_idx = int(hashlib.md5((image_id + "cat").encode()).hexdigest(), 16) % len(categories)
    style_idx = int(hashlib.md5((image_id + "style").encode()).hexdigest(), 16) % len(styles)
    material_idx = int(hashlib.md5((image_id + "mat").encode()).hexdigest(), 16) % len(materials)
    
    return {
        "id": image_id,
        "name": f"Luminaire {categories[category_idx]} {styles[style_idx]}",
        "description": f"Luminaire {categories[category_idx]} de style {styles[style_idx]} en {materials[material_idx].lower()}",
        "category": categories[category_idx].lower(),
        "style": styles[style_idx].lower(),
        "material": materials[material_idx].lower(),
        "image_path": image_path,
        "price": round(price, 2)
    }

def main():
    start_time = time.time()
    args = parse_args()
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paramètres de sortie
    embeddings_file = os.path.join(args.output_dir, "embeddings.pkl")
    metadata_file = os.path.join(args.output_dir, "luminaires.json")
    
    print("🔄 Chargement du modèle EfficientNet V2 B3...")
    model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2"
    embedding_model = hub.KerasLayer(model_url)
    print("✅ Modèle EfficientNet V2 B3 chargé")
    
    # Listing des fichiers images
    print(f"🔍 Analyse du dossier {args.image_dir}...")
    file_list = os.listdir(args.image_dir)
    image_files = [os.path.join(args.image_dir, f) for f in file_list 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    print(f"📊 {len(image_files)} images trouvées")
    
    # Préparation des listes pour stocker les résultats
    all_embeddings = []
    all_metadata = []
    processed_files = 0
    
    # Traitement par batchs
    batches = [image_files[i:i + args.batch_size] for i in range(0, len(image_files), args.batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="⚙️ Traitement des batchs")):
        embeddings, valid_paths = process_batch(batch, load_image, embedding_model)
        
        for emb, path in zip(embeddings, valid_paths):
            all_embeddings.append(emb)
            all_metadata.append(extract_metadata(path))
            processed_files += 1
        
        # Sauvegarde intermédiaire tous les 50 batchs
        if (batch_idx + 1) % 50 == 0:
            print(f"💾 Sauvegarde intermédiaire après {processed_files} images...")
            with open(embeddings_file, "wb") as f:
                pickle.dump(np.array(all_embeddings), f)
            
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    # Sauvegarde finale
    print(f"🏁 Traitement terminé! Génération des fichiers finaux...")
    print(f"   - {processed_files} images traitées sur {len(image_files)} disponibles")
    
    with open(embeddings_file, "wb") as f:
        pickle.dump(np.array(all_embeddings), f)
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    duration = time.time() - start_time
    print(f"✅ Temps total: {duration:.1f} secondes ({duration/60:.1f} minutes)")
    print(f"✅ Embeddings enregistrés dans {embeddings_file}")
    print(f"✅ Métadonnées enregistrées dans {metadata_file}")

if __name__ == "__main__":
    main()
