import os
import pickle
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
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

# Configuration TensorFlow ULTRA optimisée pour Render
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Réduire l'utilisation mémoire TF
tf.config.experimental.enable_tensor_float_32_execution(False)

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
    """Charge le modèle TensorFlow Hub optimisé"""
    global embedding_model
    
    try:
        logger.info("Chargement du modèle TensorFlow Hub...")
        
        # URL du modèle léger et rapide
        model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
        
        # Charger avec optimisations mémoire
        embedding_model = hub.load(model_url)
        
        logger.info("✅ Modèle TensorFlow Hub chargé avec succès")
        
        # Test rapide pour réchauffer le modèle
        test_input = tf.zeros((1, 224, 224, 3))
        _ = embedding_model(test_input)
        logger.info("✅ Modèle réchauffé")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
        embedding_model = None
        return False

def load_data():
    """Charge les embeddings et métadonnées"""
    global stored_embeddings, luminaires_data
    
    try:
        logger.info("📦 Chargement des données...")
        
        # Charger les embeddings
        embeddings_path = 'data/embeddings.pkl'
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                stored_embeddings = pickle.load(f)
            stored_embeddings = np.array(stored_embeddings)
            logger.info(f"✅ Embeddings chargés: {stored_embeddings.shape}")
        else:
            logger.error("❌ Fichier embeddings.pkl non trouvé")
            return False
        
        # Charger les métadonnées
        metadata_path = 'data/luminaires.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                luminaires_data = json.load(f)
            logger.info(f"✅ Métadonnées chargées: {len(luminaires_data)} éléments")
        else:
            logger.error("❌ Fichier luminaires.json non trouvé")
            return False
        
        # Vérifier cohérence
        if len(stored_embeddings) != len(luminaires_data):
            logger.warning(f"⚠️ Taille différente: {len(stored_embeddings)} embeddings vs {len(luminaires_data)} métadonnées")
            min_len = min(len(stored_embeddings), len(luminaires_data))
            stored_embeddings = stored_embeddings[:min_len]
            luminaires_data = luminaires_data[:min_len]
            logger.info(f"✅ Données ajustées à {min_len} éléments")
        
        # Charger le modèle
        model_success = load_embedding_model()
        if not model_success:
            return False
        
        logger.info("🎉 Toutes les données chargées avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des données: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        gc.collect()

def background_loading():
    """Charge les données en arrière-plan"""
    global is_loaded, is_loading
    logger.info("🚀 Démarrage du chargement en arrière-plan...")
    
    try:
        success = load_data()
        if success:
            is_loaded = True
            is_loading = False
            logger.info("🎉 Chargement terminé avec succès!")
        else:
            is_loading = False
            logger.error("❌ Échec du chargement")
    except Exception as e:
        is_loading = False
        logger.error(f"❌ Erreur chargement: {e}")

def extract_features(image):
    """Extrait les caractéristiques d'une image avec TensorFlow Hub"""
    try:
        # Préprocessing optimisé
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0  # Normalisation
        
        # Extraction avec le modèle TensorFlow Hub
        image_tensor = tf.constant(image_array)
        features = embedding_model(image_tensor)
        
        return features.numpy()[0]
        
    except Exception as e:
        logger.error(f"❌ Erreur extraction features: {e}")
        raise e
    finally:
        gc.collect()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Endpoint de santé"""
    return jsonify({
        'status': 'healthy',
        'loaded': is_loaded,
        'loading': is_loading,
        'embeddings_count': len(stored_embeddings) if stored_embeddings is not None else 0,
        'metadata_count': len(luminaires_data) if luminaires_data is not None else 0,
        'model_ready': embedding_model is not None
    })

@app.route('/api/force-load')
def force_load():
    """Force le chargement"""
    global is_loaded, is_loading
    
    if is_loading:
        return jsonify({'status': 'already_loading'})
    
    is_loading = True
    thread = threading.Thread(target=background_loading)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'loading_started'})

@app.route('/api/search', methods=['POST'])
def search():
    """Recherche par image"""
    if not is_loaded:
        return jsonify({'error': 'Système en cours de chargement, veuillez patienter...'}), 503
    
    if embedding_model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    start_time = time.time()
    
    try:
        logger.info("🔍 Début de la recherche...")
        
        # Vérifier la présence de l'image
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Charger et valider l'image
        try:
            image = Image.open(file.stream)
            logger.info(f"📷 Image chargée: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"❌ Erreur ouverture image: {e}")
            return jsonify({'error': 'Format d\'image invalide'}), 400
        
        # Extraction des caractéristiques
        logger.info("🔬 Extraction des caractéristiques...")
        query_embedding = extract_features(image)
        logger.info(f"✅ Caractéristiques extraites: {len(query_embedding)} dimensions")
        
        # Calcul de similarité
        logger.info("📊 Calcul des similarités...")
        similarities = cosine_similarity([query_embedding], stored_embeddings)[0]
        
        # Stats de debug
        logger.info(f"📈 Similarités - Min: {similarities.min():.4f}, Max: {similarities.max():.4f}, Moyenne: {similarities.mean():.4f}")
        
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
        
        logger.info(f"🎯 Recherche terminée en {processing_time:.2f}s - {len(results)} résultats")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Erreur interne du serveur'
        }), 500
    finally:
        gc.collect()

@app.route('/api/debug', methods=['POST'])
def debug_search():
    """Route de debug"""
    if not is_loaded:
        return jsonify({'error': 'Système non chargé'}), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image'}), 400
        
        file = request.files['image']
        image = Image.open(file.stream)
        
        # Extraire les features
        query_embedding = extract_features(image)
        
        # Calculer similarités
        similarities = cosine_similarity([query_embedding], stored_embeddings)[0]
        
        # Stats de debug
        stats = {
            'query_embedding_shape': query_embedding.shape,
            'query_embedding_type': str(type(query_embedding)),
            'stored_embeddings_shape': stored_embeddings.shape,
            'similarity_stats': {
                'min': float(similarities.min()),
                'max': float(similarities.max()),
                'mean': float(similarities.mean()),
                'std': float(similarities.std())
            },
            'top_10_similarities': [float(x) for x in sorted(similarities, reverse=True)[:10]],
            'model_loaded': embedding_model is not None
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# Initialisation au démarrage
if not is_loading and not is_loaded:
    is_loading = True
    thread = threading.Thread(target=background_loading)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
