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
    """Charge le modèle d'embedding"""
    global embedding_model
    
    try:
        logger.info("Chargement du modèle MobileNetV2...")
        
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
    global stored_embeddings, luminaires_data
    
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
        model_success = load_embedding_model()
        
        logger.info("✅ Toutes les données chargées avec succès")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        gc.collect()

def background_loading():
    """Charge les données en arrière-plan"""
    global is_loaded, is_loading
    logger.info("Démarrage du chargement en arrière-plan...")
    
    try:
        success = load_data()
        if success:
            is_loaded = True
            is_loading = False
            logger.info("✅ Chargement terminé avec succès!")
        else:
            is_loading = False
            logger.error("❌ Échec du chargement")
    except Exception as e:
        is_loading = False
        logger.error(f"❌ Erreur chargement: {e}")

def preprocess_image(image):
    """Préprocesse l'image pour le modèle"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        
        return image_array
    except Exception as e:
        logger.error(f"Erreur preprocessing: {e}")
        raise

def extract_features(image):
    """Extrait les caractéristiques d'une image"""
    global embedding_model
    
    if embedding_model is None:
        raise Exception("Modèle d'embedding non chargé")
    
    try:
        processed_image = preprocess_image(image)
        features = embedding_model.predict(processed_image, verbose=0)
        return features[0]
    except Exception as e:
        logger.error(f"Erreur extraction features: {e}")
        raise

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Endpoint de santé"""
    return jsonify({
        'status': 'healthy',
        'is_loaded': is_loaded,
        'is_loading': is_loading,
        'has_embeddings': stored_embeddings is not None,
        'has_metadata': luminaires_data is not None,
        'has_model': embedding_model is not None,
        'embeddings_count': len(stored_embeddings) if stored_embeddings is not None else 0
    })

@app.route('/api/force-load')
def force_load():
    """Force le chargement des données"""
    global is_loaded, is_loading
    
    if is_loading:
        return jsonify({'status': 'already_loading'})
    
    is_loading = True
    success = load_data()
    
    if success:
        is_loaded = True
        is_loading = False
        return jsonify({'status': 'success', 'loaded': True})
    else:
        is_loading = False
        return jsonify({'status': 'error', 'loaded': False})

@app.route('/api/search', methods=['POST'])
def search():
    if not is_loaded:
        if is_loading:
            return jsonify({'error': 'Système en cours de chargement, veuillez patienter...'}), 503
        else:
            return jsonify({'error': 'Système non initialisé'}), 503
    
    start_time = time.time()
    
    try:
        logger.info("Début de la recherche...")
        
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

# Initialisation au démarrage
if not is_loading and not is_loaded:
    is_loading = True
    thread = threading.Thread(target=background_loading)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
