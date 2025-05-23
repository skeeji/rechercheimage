from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import io
import base64
import json
import logging
import threading
import time
import traceback
from flask_cors import CORS

# Configuration
app = Flask(__name__)
CORS(app)

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
embedding_model = None
luminaire_embeddings = []
luminaire_metadata = []
is_loaded = False
is_loading = False

# Configuration TensorFlow optimisée pour Render
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_soft_device_placement(True)
if tf.config.list_physical_devices('GPU'):
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

def load_model():
    """Charge le modèle MobileNet avec optimisations mémoire"""
    global embedding_model
    try:
        logger.info("📱 Chargement du modèle MobileNet...")
        
        # Modèle plus léger et optimisé
        embedding_model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/5")
        
        # Test rapide du modèle
        test_input = tf.zeros((1, 224, 224, 3))
        test_output = embedding_model(test_input)
        logger.info(f"✅ Modèle chargé! Dimension: {test_output.shape}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
        logger.error(traceback.format_exc())
        return False

def load_data():
    """Charge les données des luminaires"""
    global luminaire_embeddings, luminaire_metadata
    
    try:
        logger.info("📂 Chargement des données...")
        
        # Chargement des embeddings
        embeddings_path = 'luminaire_embeddings.npy'
        if os.path.exists(embeddings_path):
            luminaire_embeddings = np.load(embeddings_path)
            logger.info(f"✅ Embeddings chargés: {luminaire_embeddings.shape}")
        else:
            logger.error(f"❌ Fichier embeddings non trouvé: {embeddings_path}")
            return False
        
        # Chargement des métadonnées
        metadata_path = 'luminaire_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                luminaire_metadata = json.load(f)
            logger.info(f"✅ Métadonnées chargées: {len(luminaire_metadata)} luminaires")
        else:
            logger.error(f"❌ Fichier métadonnées non trouvé: {metadata_path}")
            return False
        
        # Vérification cohérence
        if len(luminaire_embeddings) != len(luminaire_metadata):
            logger.error(f"❌ Incohérence: {len(luminaire_embeddings)} embeddings vs {len(luminaire_metadata)} métadonnées")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur chargement données: {e}")
        logger.error(traceback.format_exc())
        return False

def background_loading():
    """Charge les données en arrière-plan"""
    global is_loaded, is_loading
    logger.info("🚀 Démarrage du chargement en arrière-plan...")
    
    try:
        # Chargement du modèle
        model_success = load_model()
        if not model_success:
            is_loading = False
            is_loaded = False
            logger.error("❌ Échec du chargement modèle")
            return
        
        # Chargement des données
        data_success = load_data()
        if not data_success:
            is_loading = False
            is_loaded = False
            logger.error("❌ Échec du chargement données")
            return
        
        # Succès complet
        is_loaded = True
        is_loading = False
        logger.info(f"🎉 Chargement terminé avec succès! is_loaded={is_loaded}")
        
    except Exception as e:
        is_loading = False
        is_loaded = False
        logger.error(f"❌ Erreur chargement: {e}")
        logger.error(traceback.format_exc())

def preprocess_image(image):
    """Prétraite l'image pour le modèle"""
    try:
        # Redimensionner à 224x224
        image = image.convert('RGB')
        image = image.resize((224, 224))
        
        # Convertir en array et normaliser
        image_array = np.array(image)
        image_array = image_array.astype(np.float32)
        image_array = image_array / 255.0
        
        # Ajouter dimension batch
        image_array = np.expand_dims(image_array, 0)
        
        return image_array
    except Exception as e:
        logger.error(f"❌ Erreur preprocessing: {e}")
        raise

def get_image_embedding(image):
    """Calcule l'embedding d'une image"""
    try:
        if embedding_model is None:
            raise ValueError("Modèle non chargé")
        
        preprocessed = preprocess_image(image)
        embedding = embedding_model(preprocessed)
        embedding = embedding.numpy().flatten()
        
        # Normalisation
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    except Exception as e:
        logger.error(f"❌ Erreur calcul embedding: {e}")
        raise

def find_similar_luminaires(query_embedding, top_k=6):
    """Trouve les luminaires similaires"""
    try:
        if len(luminaire_embeddings) == 0:
            return []
        
        # Calcul des similarités cosinus
        similarities = np.dot(luminaire_embeddings, query_embedding)
        
        # Top-k résultats
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(luminaire_metadata):
                result = luminaire_metadata[idx].copy()
                result['similarity'] = float(similarities[idx])
                results.append(result)
        
        return results
    except Exception as e:
        logger.error(f"❌ Erreur recherche: {e}")
        return []

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Endpoint de santé du système"""
    return jsonify({
        'status': 'healthy',
        'loaded': is_loaded,
        'loading': is_loading,
        'model_ready': embedding_model is not None,
        'embeddings_count': len(luminaire_embeddings),
        'metadata_count': len(luminaire_metadata)
    })

@app.route('/api/force-load')
def force_load():
    """Force le rechargement du système"""
    global is_loading, is_loaded
    
    if is_loading:
        return jsonify({'message': 'Chargement déjà en cours...', 'loading': True})
    
    # Reset et relance
    is_loading = True
    is_loaded = False
    
    def reload_system():
        background_loading()
    
    thread = threading.Thread(target=reload_system, daemon=True)
    thread.start()
    
    return jsonify({
        'message': 'Rechargement forcé démarré',
        'loading': True,
        'estimated_time': '2-3 minutes'
    })

@app.route('/api/search', methods=['POST'])
def search():
    """Recherche par image"""
    logger.info(f"🔍 Recherche - is_loaded: {is_loaded}, modèle: {embedding_model is not None}")
    
    if not is_loaded:
        return jsonify({'error': 'Système en cours de chargement, veuillez patienter...'}), 503
    
    if embedding_model is None:
        return jsonify({'error': 'Modèle non disponible'}), 500
    
    try:
        # Récupération image
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Chargement et traitement
        image = Image.open(file.stream)
        logger.info(f"📸 Image reçue: {image.size}")
        
        # Calcul embedding
        query_embedding = get_image_embedding(image)
        logger.info(f"🧮 Embedding calculé: {query_embedding.shape}")
        
        # Recherche
        results = find_similar_luminaires(query_embedding)
        logger.info(f"🎯 {len(results)} résultats trouvés")
        
        return jsonify({
            'results': results,
            'count': len(results),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur recherche: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Erreur de traitement: {str(e)}'}), 500

# Démarrage du chargement en arrière-plan
def start_background_loading():
    global is_loading
    if not is_loading and not is_loaded:
        is_loading = True
        thread = threading.Thread(target=background_loading, daemon=True)
        thread.start()
        logger.info("🔄 Chargement en arrière-plan démarré")

if __name__ == '__main__':
    # Démarrage immédiat du chargement
    start_background_loading()
    
    # Lancement du serveur
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # Mode production (gunicorn)
    start_background_loading()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
