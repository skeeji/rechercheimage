import os
import tensorflow as tf
import tensorflow_hub as hub

# Création du dossier pour le modèle
os.makedirs('models', exist_ok=True)

# URL du modèle - exactement le même que dans votre app
model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2"

print("Téléchargement du modèle EfficientNet...")
model = hub.KerasLayer(model_url)

# Création d'un modèle simple pour sauvegarder
full_model = tf.keras.Sequential([model])
full_model.build([None, 224, 224, 3])

# Sauvegarde du modèle localement
save_path = 'models/efficientnet_v2_model'
print(f"Sauvegarde du modèle dans {save_path}...")
full_model.save(save_path)
print("Modèle sauvegardé avec succès!")

# Conversion des embeddings en format NPY (optionnel)
import pickle
import numpy as np

if os.path.exists('data/embeddings.pkl'):
    print("Conversion des embeddings en format NPY...")
    with open('data/embeddings.pkl', 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Sauvegarde au format NPY (plus rapide à charger)
    np.save('data/embeddings.npy', embeddings_data)
    print("Embeddings convertis avec succès!")
