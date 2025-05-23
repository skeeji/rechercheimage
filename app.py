import os
import pickle
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
import time
import uuid
import traceback
import logging
from dotenv import load_dotenv
import gc
import threading

# Configuration TensorFlow optimisée
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
stored_embeddings = None
luminaires_data = None
embedding_model = None
is_loaded = False
is_loading = False

def load_embedding_model():
    """Charge le modèle d'embedding (version allégée)"""
    global embedding_model
    
    try:
        logger.info("Chargement du modèle MobileNetV2...")
        
        # Utiliser MobileNetV2 au lieu de TensorFlow Hub (plus léger)
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        
        embedding_model = base_model
        logger.info("Modèle MobileNetV2 chargé avec succès")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        embedding_model = None
        return False

def load_data():
    """Charge les embeddings et métadonnées"""
    global stored_embeddings, luminaires_data, is_loaded, is_loading
    
    if is_loading:
        return False
        
    is_loading = True
    
    try:
        logger.info("Chargement des données...")
        
        # Charger les embeddings
        embeddings_path = 'data/embeddings.pkl'
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                stored_embeddings = pickle.load(f)
            stored_embeddings = np.array(stored_embeddings)
            logger.info(f"Embeddings chargés: {stored_embeddings.shape}")
        else:
            # Essayer le format .npy
            embeddings_npy_path = 'data/embeddings.npy'
            if os.path.exists(embeddings_npy_path):
                stored_embeddings = np.load(embeddings_npy_path)
                logger.info(f"Embeddings NPY chargés: {stored_embeddings.shape}")
            else:
                logger.error("Aucun fichier d'embeddings trouvé")
                return False
        
        # Charger les métadonnées
        metadata_path = 'data/luminaires.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                luminaires_data = json.load(f)
            logger.info(f"Métadonnées chargées: {len(luminaires_data)} éléments")
        else:
            logger.error("Fichier de métadonnées non trouvé")
            return False
        
        # Vérifier la cohérence
        if len(stored_embeddings) != len(luminaires_data):
            logger.warning(f"Incohérence détectée: {len(stored_embeddings)} embeddings vs {len(luminaires_data)} métadonnées")
            min_len = min(len(stored_embeddings), len(luminaires_data))
            stored_embeddings = stored_embeddings[:min_len]
            luminaires_data = luminaires_data[:min_len]
            logger.info(f"Données ajustées à {min_len} éléments")
        
        # Charger le modèle
        load_embedding_model()
        
        is_loaded = True
        logger.info("Toutes les données chargées avec succès")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        is_loading = False
        gc.collect()

def extract_features(image):
    """Extrait les caractéristiques de l'image"""
    global embedding_model
    
    try:
        if embedding_model is None:
            # Méthode de fallback simple
            return extract_simple_features(image)
        
        # Préprocessing
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.cast(img_array, tf.float32) / 255.0
        
        # Extraction des caractéristiques
        features = embedding_model.predict(img_array, verbose=0)
        return features.flatten()
        
    except Exception as e:
        logger.error(f"Erreur extraction features: {e}")
        return extract_simple_features(image)

def extract_simple_features(image):
    """Méthode de fallback pour l'extraction de caractéristiques"""
    try:
        # Caractéristiques basiques comme fallback
        image = image.resize((64, 64)).convert('RGB')
        pixels = np.array(image)
        
        # Moyennes par canal
        mean_r = np.mean(pixels[:,:,0])
        mean_g = np.mean(pixels[:,:,1]) 
        mean_b = np.mean(pixels[:,:,2])
        
        # Écarts-types
        std_r = np.std(pixels[:,:,0])
        std_g = np.std(pixels[:,:,1])
        std_b = np.std(pixels[:,:,2])
        
        # Histogrammes simplifiés
        hist_r, _ = np.histogram(pixels[:,:,0], bins=8, range=(0, 255))
        hist_g, _ = np.histogram(pixels[:,:,1], bins=8, range=(0, 255))
        hist_b, _ = np.histogram(pixels[:,:,2], bins=8, range=(0, 255))
        
        # Normaliser les histogrammes
        hist_r = hist_r / np.sum(hist_r)
        hist_g = hist_g / np.sum(hist_g)
        hist_b = hist_b / np.sum(hist_b)
        
        # Combiner toutes les caractéristiques
        features = np.concatenate([
            [mean_r, mean_g, mean_b, std_r, std_g, std_b],
            hist_r, hist_g, hist_b
        ])
        
        # Étendre à 1280 dimensions pour compatibilité
        if len(features) < 1280:
            padding = np.zeros(1280 - len(features))
            features = np.concatenate([features, padding])
        
        return features
        
    except Exception:
        # En dernier recours
        return np.random.random(1280)

# Chargement en arrière-plan
def background_loading():
    """Charge les données en arrière-plan"""
    logger.info("Démarrage du chargement en arrière-plan...")
    load_data()

# Démarrer le chargement en arrière-plan
threading.Thread(target=background_loading, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Endpoint de santé de l'application"""
    return jsonify({
        'status': 'ok',
        'data_loaded': is_loaded,
        'model_loaded': embedding_model is not None,
        'embeddings_count': len(stored_embeddings) if stored_embeddings is not None else 0,
        'products_count': len(luminaires_data) if luminaires_data is not None else 0
    })

@app.route('/api/search', methods=['POST'])
def search_similar():
    start_time = time.time()
    
    try:
        logger.info("Début de la recherche...")
        
        # Vérifier si les données sont chargées
        if not is_loaded:
            if is_loading:
                return jsonify({
                    'error': 'Système en cours de chargement, veuillez patienter...'
                }), 503
            else:
                # Essayer de charger maintenant
                logger.info("Tentative de chargement des données...")
                if not load_data():
                    return jsonify({
                        'error': 'Impossible de charger les données'
                    }), 503
        
        # Vérifier la présence du fichier
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Traitement de l'image
        try:
            image = Image.open(file.stream)
            logger.info(f"Image reçue: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"Erreur ouverture image: {e}")
            return jsonify({'error': 'Format d\'image invalide'}), 400
        
        # Extraction des caractéristiques
        logger.info("Extraction des caractéristiques...")
        query_embedding = extract_features(image)
        logger.info(f"Caractéristiques extraites: {len(query_embedding)} dimensions")
        
        # Calcul de similarité
        logger.info("Calcul des similarités...")
        similarities = cosine_similarity([query_embedding], stored_embeddings)[0]
        
        # Obtenir les Top-K résultats
        num_results = min(int(request.form.get('num_results', 6)), 20)
        top_indices = np.argsort(similarities)[::-1][:num_results]
        
        # Construire la réponse
        results = []
        for idx in top_indices:
            if idx < len(luminaires_data):
                item = luminaires_data[idx].copy()
                item['similarity'] = float(similarities[idx])
                results.append(item)
        
        processing_time = time.time() - start_time
        
        response = {
            'results': results,
            'count': len(results),
            'processing_time': processing_time
        }
        
        logger.info(f"Recherche terminée en {processing_time:.2f}s - {len(results)} résultats")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Erreur interne du serveur'
        }), 500
    finally:
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
