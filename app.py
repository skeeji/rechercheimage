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

# Création du dossier d'upload s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales pour les modèles et données
embeddings = None
metadata = None
embedding_model = None
nn_model = None
data_loaded = False
model_loaded = False

def cleanup_memory():
    """Force le nettoyage de la mémoire"""
    gc.collect()

def load_embedding_model():
    """Charge le modèle d'embedding de manière robuste"""
    global embedding_model, model_loaded
    
    if model_loaded and embedding_model is not None:
        return True
        
    logger.info("Tentative de chargement du modèle d'embedding...")
    
    try:
        # D'abord essayer de charger le modèle préenregistré
        if os.path.exists(app.config['MODEL_PATH']):
            logger.info("Chargement du modèle préenregistré...")
            embedding_model = tf.keras.models.load_model(app.config['MODEL_PATH'])
            logger.info("Modèle préenregistré chargé avec succès")
            model_loaded = True
            return True
            
    except Exception as e:
        logger.warning(f"Échec du chargement du modèle préenregistré: {e}")
        embedding_model = None
    
    # Fallback: charger depuis TF Hub
    try:
        logger.info("Téléchargement du modèle depuis TF Hub...")
        import tensorflow_hub as hub
        
        # Utiliser une URL plus simple et plus fiable
        model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2"
        
        # Créer le modèle avec une couche hub
        embedding_model = tf.keras.Sequential([
            hub.KerasLayer(model_url, trainable=False)
        ])
        
        # Compiler le modèle pour s'assurer qu'il fonctionne
        embedding_model.build([None, 224, 224, 3])
        
        logger.info("Modèle TF Hub chargé et compilé avec succès")
        model_loaded = True
        cleanup_memory()
        return True
        
    except Exception as e:
        logger.error(f"Échec du chargement depuis TF Hub: {e}")
        logger.error(traceback.format_exc())
        embedding_model = None
        model_loaded = False
        return False

def load_data():
    """Charge toutes les données nécessaires"""
    global embeddings, metadata, nn_model, data_loaded

    if data_loaded:
        return embeddings, metadata, nn_model

    # Chargement du modèle d'embedding
    if not load_embedding_model():
        logger.error("Impossible de charger le modèle d'embedding")
        return None, None, None

    # Chargement des embeddings
    logger.info("Chargement des embeddings...")
    try:
        if os.path.exists(app.config['NPY_EMBEDDINGS_PATH']):
            embeddings = np.load(app.config['NPY_EMBEDDINGS_PATH'])
            logger.info(f"Embeddings .npy chargés: {embeddings.shape}")
        elif os.path.exists(app.config['EMBEDDINGS_PATH']):
            with open(app.config['EMBEDDINGS_PATH'], 'rb') as f:
                embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            logger.info(f"Embeddings .pkl chargés: {embeddings.shape}")
        else:
            logger.error("Aucun fichier d'embeddings trouvé")
            return None, None, None
            
        # Limiter le dataset pour économiser la mémoire
        if len(embeddings) > 500:
            embeddings = embeddings[:500]
            logger.info(f"Dataset limité à {len(embeddings)} éléments")
            
    except Exception as e:
        logger.error(f"Erreur lors du chargement des embeddings: {e}")
        return None, None, None

    # Chargement des métadonnées
    logger.info("Chargement des métadonnées...")
    try:
        if os.path.exists(app.config['METADATA_PATH']):
            with open(app.config['METADATA_PATH'], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Synchroniser avec les embeddings limités
            if len(metadata) > len(embeddings):
                metadata = metadata[:len(embeddings)]
                
            logger.info(f"Métadonnées chargées: {len(metadata)} éléments")
        else:
            logger.error("Fichier de métadonnées non trouvé")
            return None, None, None
    except Exception as e:
        logger.error(f"Erreur lors du chargement des métadonnées: {e}")
        return None, None, None

    # Création du modèle de recherche
    try:
        nn_model = NearestNeighbors(n_neighbors=min(20, len(embeddings)), metric='cosine')
        nn_model.fit(embeddings)
        logger.info("Modèle de recherche créé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de la création du modèle de recherche: {e}")
        return None, None, None

    data_loaded = True
    cleanup_memory()
    logger.info("Chargement des données terminé avec succès")
    return embeddings, metadata, nn_model

def get_data():
    """Fonction utilitaire pour obtenir les données chargées"""
    return load_data()

def process_image(image_path, target_size=(224, 224)):
    """
    Traite une image et extrait ses caractéristiques avec le modèle d'embedding
    """
    try:
        # Vérifier que le modèle est chargé
        if embedding_model is None or not model_loaded:
            logger.error("Modèle d'embedding non chargé")
            # Tenter de recharger le modèle
            if not load_embedding_model():
                return None
        
        # Charger et préprocesser l'image
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Obtenir les embeddings
        features = embedding_model(image_batch)
        features = features.numpy().flatten()
        
        # Nettoyage
        cleanup_memory()
        
        return features
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image {image_path}: {e}")
        logger.error(traceback.format_exc())
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_images():
    start_time = time.time()
    
    try:
        logger.info("Début de la recherche...")
        
        # Vérifier qu'un fichier a été envoyé
        if 'image' not in request.files:
            return jsonify({'error': 'Aucun fichier envoyé'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400

        # Obtenir le nombre de résultats demandés
        num_results = int(request.form.get('num_results', app.config['DEFAULT_NUM_RESULTS']))
        num_results = min(num_results, 12)

        # Sauvegarder l'image temporairement
        filename = secure_filename(f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Traiter l'image et extraire les caractéristiques
            query_features = process_image(filepath)
            if query_features is None:
                return jsonify({'error': 'Impossible de traiter cette image. Modèle non disponible.'}), 500

            # Obtenir les données nécessaires pour la recherche
            embeddings, metadata, nn_model = get_data()
            
            if embeddings is None or metadata is None or nn_model is None:
                return jsonify({'error': 'Les données de recherche ne sont pas disponibles'}), 500
            
            # Recherche des plus proches voisins
            num_neighbors = min(num_results, len(embeddings))
            distances, indices = nn_model.kneighbors([query_features], n_neighbors=num_neighbors)

            # Préparer les résultats
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(metadata):
                    item = metadata[idx].copy()

                    if 'image_path' in item:
                        image_name = os.path.basename(item['image_path'])
                        item['image_url'] = f"/images/{image_name}"
                    
                    item['similarity'] = float(1 - distances[0][i])
                    results.append(item)

            query_image_url = f"/uploads/{filename}"
            processing_time = time.time() - start_time
            
            logger.info(f"Recherche effectuée en {processing_time:.2f}s avec {len(results)} résultats")
            
            return jsonify({
                'query_image': query_image_url,
                'results': results,
                'processing_time': processing_time,
                'count': len(results)
            })
            
        finally:
            # Nettoyage du fichier temporaire
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
    
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Erreur interne du serveur: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/images/<filename>')
def database_image(filename):
    return send_from_directory('data/images', filename)

@app.route('/health')
def health_check():
    """Endpoint de santé pour le monitoring"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loaded,
            'data_loaded': data_loaded,
            'embedding_model_available': embedding_model is not None
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# Initialisation au démarrage pour Gunicorn
def init_app():
    """Initialise l'application au démarrage"""
    try:
        logger.info("Initialisation de l'application...")
        # Charger les données au démarrage
        load_data()
        logger.info("Application initialisée avec succès!")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")

if __name__ == '__main__':
    init_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
