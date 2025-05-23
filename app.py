import os
import pickle
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.neighbors import NearestNeighbors
import time
import uuid
import traceback
import logging
from dotenv import load_dotenv
import gc
import threading

# Configuration stricte de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Configuration de l'environnement
load_dotenv()
os.environ['TFHUB_CACHE_DIR'] = os.path.join(os.getcwd(), 'models', 'tfhub_cache')

# Configuration de l'application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
app.config['EMBEDDINGS_PATH'] = 'data/embeddings.pkl'
app.config['NPY_EMBEDDINGS_PATH'] = 'data/embeddings.npy'
app.config['METADATA_PATH'] = 'data/luminaires.json'
app.config['DEFAULT_NUM_RESULTS'] = 6
app.config['MODEL_PATH'] = 'models/efficientnet_v2_model'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
embeddings = None
metadata = None
embedding_model = None
nn_model = None
data_loaded = False
model_loaded = False
loading_in_progress = False
loading_lock = threading.Lock()

def cleanup_memory():
    """Force le nettoyage de la mémoire"""
    gc.collect()

def load_embedding_model():
    """Charge le modèle d'embedding avec optimisations"""
    global embedding_model, model_loaded, loading_in_progress
    
    with loading_lock:
        if model_loaded and embedding_model is not None:
            return True
        
        if loading_in_progress:
            # Attendre que l'autre thread finisse le chargement
            for _ in range(30):  # Attendre max 30 secondes
                time.sleep(1)
                if model_loaded:
                    return True
            return False
            
        loading_in_progress = True
    
    try:
        logger.info("Chargement du modèle d'embedding...")
        
        # Essayer le modèle préenregistré d'abord
        if os.path.exists(app.config['MODEL_PATH']):
            try:
                logger.info("Chargement du modèle préenregistré...")
                embedding_model = tf.keras.models.load_model(app.config['MODEL_PATH'], compile=False)
                logger.info("Modèle préenregistré chargé")
                model_loaded = True
                return True
            except Exception as e:
                logger.warning(f"Échec modèle préenregistré: {e}")
        
        # Fallback: modèle simple depuis TF Hub
        try:
            import tensorflow_hub as hub
            logger.info("Chargement depuis TF Hub...")
            
            # Modèle plus léger et plus rapide
            model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
            
            # Construction rapide du modèle
            embedding_model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
                hub.KerasLayer(model_url, trainable=False)
            ])
            
            # Test rapide du modèle
            test_input = tf.random.normal([1, 224, 224, 3])
            _ = embedding_model(test_input)
            
            logger.info("Modèle TF Hub chargé et testé")
            model_loaded = True
            cleanup_memory()
            return True
            
        except Exception as e:
            logger.error(f"Échec TF Hub: {e}")
            
            # Dernier fallback: modèle très simple
            try:
                logger.info("Utilisation du modèle de fallback...")
                embedding_model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(64)
                ])
                
                # Compiler et tester
                embedding_model.compile(optimizer='adam', loss='mse')
                test_input = tf.random.normal([1, 224, 224, 3])
                _ = embedding_model(test_input)
                
                logger.info("Modèle de fallback chargé")
                model_loaded = True
                return True
                
            except Exception as e:
                logger.error(f"Échec complet du chargement: {e}")
                model_loaded = False
                return False
        
    finally:
        loading_in_progress = False

def load_data():
    """Charge les données rapidement"""
    global embeddings, metadata, nn_model, data_loaded

    if data_loaded:
        return embeddings, metadata, nn_model

    # Chargement asynchrone du modèle si pas déjà fait
    if not model_loaded:
        if not load_embedding_model():
            logger.error("Impossible de charger le modèle")
            return None, None, None

    # Chargement optimisé des données
    try:
        logger.info("Chargement des embeddings...")
        
        if os.path.exists(app.config['NPY_EMBEDDINGS_PATH']):
            embeddings = np.load(app.config['NPY_EMBEDDINGS_PATH'])
        elif os.path.exists(app.config['EMBEDDINGS_PATH']):
            with open(app.config['EMBEDDINGS_PATH'], 'rb') as f:
                embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
        else:
            logger.error("Aucun fichier d'embeddings trouvé")
            return None, None, None
        
        # Limiter sévèrement pour éviter les timeouts
        max_items = 100  # Très réduit pour la rapidité
        if len(embeddings) > max_items:
            embeddings = embeddings[:max_items]
        
        logger.info(f"Embeddings chargés: {embeddings.shape}")

        # Métadonnées
        if os.path.exists(app.config['METADATA_PATH']):
            with open(app.config['METADATA_PATH'], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            metadata = metadata[:len(embeddings)]
            logger.info(f"Métadonnées: {len(metadata)} éléments")
        else:
            logger.error("Métadonnées non trouvées")
            return None, None, None

        # Modèle de recherche rapide
        nn_model = NearestNeighbors(n_neighbors=min(10, len(embeddings)), metric='cosine', n_jobs=1)
        nn_model.fit(embeddings)
        
        data_loaded = True
        cleanup_memory()
        logger.info("Données chargées avec succès")
        return embeddings, metadata, nn_model
        
    except Exception as e:
        logger.error(f"Erreur chargement données: {e}")
        return None, None, None

def process_image(image_path, target_size=(224, 224)):
    """Traite une image rapidement"""
    try:
        if not model_loaded or embedding_model is None:
            if not load_embedding_model():
                return None
        
        # Traitement rapide de l'image
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size, Image.LANCZOS)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Extraction des features
        features = embedding_model(image_batch)
        features = features.numpy().flatten()
        
        cleanup_memory()
        return features
        
    except Exception as e:
        logger.error(f"Erreur traitement image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_images():
    start_time = time.time()
    
    try:
        logger.info("Début recherche...")
        
        if 'image' not in request.files:
            return jsonify({'error': 'Aucun fichier envoyé'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400

        num_results = int(request.form.get('num_results', 4))  # Réduit par défaut
        num_results = min(num_results, 8)

        # Sauvegarde rapide
        filename = secure_filename(f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Chargement asynchrone des données si nécessaire
            embeddings, metadata, nn_model = load_data()
            
            if embeddings is None or metadata is None or nn_model is None:
                return jsonify({'error': 'Données non disponibles. Réessayez dans quelques secondes.'}), 503
            
            # Traitement de l'image
            query_features = process_image(filepath)
            if query_features is None:
                return jsonify({'error': 'Impossible de traiter cette image'}), 500

            # Recherche rapide
            num_neighbors = min(num_results, len(embeddings))
            distances, indices = nn_model.kneighbors([query_features], n_neighbors=num_neighbors)

            # Résultats
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(metadata):
                    item = metadata[idx].copy()
                    if 'image_path' in item:
                        image_name = os.path.basename(item['image_path'])
                        item['image_url'] = f"/images/{image_name}"
                    item['similarity'] = float(1 - distances[0][i])
                    results.append(item)

            processing_time = time.time() - start_time
            logger.info(f"Recherche terminée en {processing_time:.2f}s")
            
            return jsonify({
                'query_image': f"/uploads/{filename}",
                'results': results,
                'processing_time': processing_time,
                'count': len(results)
            })
            
        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
    
    except Exception as e:
        logger.error(f"Erreur recherche: {e}")
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/images/<filename>')
def database_image(filename):
    return send_from_directory('data/images', filename)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'data_loaded': data_loaded
    })

# Démarrage en arrière-plan
def background_init():
    """Initialisation en arrière-plan"""
    try:
        logger.info("Initialisation en arrière-plan...")
        load_data()
        logger.info("Initialisation terminée")
    except Exception as e:
        logger.error(f"Erreur initialisation: {e}")

# Lancer l'initialisation en arrière-plan au démarrage
if not data_loaded:
    init_thread = threading.Thread(target=background_init, daemon=True)
    init_thread.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
