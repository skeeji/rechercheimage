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

# Configuration stricte de TensorFlow pour économiser la mémoire
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Désactiver l'optimisation JIT qui consomme de la mémoire
tf.config.optimizer.set_jit(False)

# Configuration de l'environnement
load_dotenv()

# Configuration de l'application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # Réduit à 8 MB
app.config['EMBEDDINGS_PATH'] = 'data/embeddings.pkl'
app.config['NPY_EMBEDDINGS_PATH'] = 'data/embeddings.npy'
app.config['METADATA_PATH'] = 'data/luminaires.json'
app.config['DEFAULT_NUM_RESULTS'] = 6  # Réduit de 12 à 6
app.config['MODEL_PATH'] = 'models/efficientnet_v2_model'

# Création du dossier d'upload s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration du logging
logging.basicConfig(level=logging.ERROR)  # Changé à ERROR pour réduire les logs
logger = logging.getLogger(__name__)

# Variables globales pour les modèles et données
embeddings = None
metadata = None
embedding_model = None
nn_model = None
data_loaded = False

def cleanup_memory():
    """Force le nettoyage de la mémoire"""
    gc.collect()
    if hasattr(tf.keras.backend, 'clear_session'):
        tf.keras.backend.clear_session()

def load_model_lightweight():
    """Charge un modèle plus léger"""
    global embedding_model
    
    if embedding_model is not None:
        return embedding_model
    
    try:
        # Utiliser un modèle plus léger - MobileNetV2
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        
        logger.info("Chargement de MobileNetV2 (léger)...")
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        
        # Geler les couches pour économiser la mémoire
        base_model.trainable = False
        
        embedding_model = base_model
        logger.info("Modèle léger chargé avec succès")
        
        # Nettoyer immédiatement
        cleanup_memory()
        
        return embedding_model
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise

def load_data_optimized():
    """Chargement optimisé des données"""
    global embeddings, metadata, nn_model, data_loaded
    
    if data_loaded:
        return
    
    try:
        logger.info("Chargement des données...")
        
        # Charger les embeddings (limiter la taille)
        if os.path.exists(app.config['NPY_EMBEDDINGS_PATH']):
            embeddings = np.load(app.config['NPY_EMBEDDINGS_PATH'])
        else:
            with open(app.config['EMBEDDINGS_PATH'], 'rb') as f:
                embeddings = pickle.load(f)
        
        # Limiter le nombre d'embeddings si trop important
        if len(embeddings) > 1000:
            logger.info("Limitation du dataset pour économiser la mémoire")
            embeddings = embeddings[:1000]
        
        # Charger les métadonnées correspondantes
        with open(app.config['METADATA_PATH'], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Limiter les métadonnées aussi
        if len(metadata) > len(embeddings):
            metadata = metadata[:len(embeddings)]
        
        # Modèle NN avec moins de voisins
        nn_model = NearestNeighbors(metric='cosine', n_jobs=1)
        nn_model.fit(embeddings)
        
        data_loaded = True
        logger.info(f"Données chargées: {len(embeddings)} éléments")
        
        # Nettoyer la mémoire
        cleanup_memory()
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        raise

def process_image_lightweight(image_path, target_size=(224, 224)):
    """Version optimisée du traitement d'image"""
    try:
        # Charger le modèle à la demande
        model = load_model_lightweight()
        
        # Traitement de l'image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Preprocessing pour MobileNetV2
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img_array = preprocess_input(img_array)
        
        # Prédiction
        features = model.predict(img_array, verbose=0, batch_size=1)
        features = features.flatten()
        
        # Normalisation
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        # Nettoyer
        del img, img_array
        cleanup_memory()
        
        return features
        
    except Exception as e:
        logger.error(f"Erreur de traitement d'image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    try:
        start_time = time.time()
        
        # Vérifications basiques
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nom de fichier vide'}), 400
        
        # Limiter le nombre de résultats
        try:
            num_results = min(int(request.form.get('num_results', 6)), 6)
        except:
            num_results = 6
        
        # Sauvegarder l'image
        filename = secure_filename(f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Charger les données si nécessaire
            load_data_optimized()
            
            # Traitement de l'image
            query_features = process_image_lightweight(filepath)
            if query_features is None:
                return jsonify({'error': 'Impossible de traiter cette image'}), 400
            
            # Recherche
            num_neighbors = min(num_results, len(embeddings))
            distances, indices = nn_model.kneighbors([query_features], n_neighbors=num_neighbors)
            
            # Résultats
            results = []
            for i, idx in enumerate(indices[0]):
                item = metadata[idx].copy()
                if 'image_path' in item:
                    image_name = os.path.basename(item['image_path'])
                    item['image_url'] = f"/images/{image_name}"
                item['similarity'] = float(1 - distances[0][i])
                results.append(item)
            
            query_image_url = f"/uploads/{filename}"
            processing_time = time.time() - start_time
            
            return jsonify({
                'query_image': query_image_url,
                'results': results,
                'processing_time': processing_time,
                'count': len(results)
            })
            
        finally:
            # Nettoyer le fichier temporaire
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
            cleanup_memory()
    
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {e}")
        return jsonify({'error': f'Erreur de traitement: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/images/<filename>')
def database_image(filename):
    return send_from_directory('data/images', filename)

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
