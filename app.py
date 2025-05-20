import os
import pickle
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.neighbors import NearestNeighbors
import time
import uuid
import traceback
import logging
from dotenv import load_dotenv

# Configuration de l'environnement
load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supprimer les messages TF

# Configuration de l'application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
app.config['EMBEDDINGS_PATH'] = 'data/embeddings.pkl'
app.config['METADATA_PATH'] = 'data/luminaires.json'
app.config['DEFAULT_NUM_RESULTS'] = 12

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

# Chargement des embeddings et des métadonnées
def load_data():
    global embeddings, metadata, embedding_model, nn_model

    logger.info("Chargement du modèle d'embedding...")
    try:
        # Utiliser EXACTEMENT le même modèle qui a généré les embeddings
        model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2"
        embedding_model = hub.KerasLayer(model_url)
        logger.info("Modèle d'embedding chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        logger.error(traceback.format_exc())
        raise

    logger.info("Chargement des features et metadata...")
    start_time = time.time()

    try:
        with open(app.config['EMBEDDINGS_PATH'], 'rb') as f:
            embeddings = pickle.load(f)
            logger.info(f"Nombre d'embeddings chargés: {len(embeddings)}")

        with open(app.config['METADATA_PATH'], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            logger.info(f"Nombre d'éléments dans metadata: {len(metadata)}")

        logger.info(f"Dimension des embeddings: {embeddings.shape}")

        # Préparation du modèle de recherche des plus proches voisins
        nn_model = NearestNeighbors(metric='cosine')
        nn_model.fit(embeddings)
        logger.info("Modèle de recherche initialisé")
        
        logger.info("Application démarrée avec succès!")

    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        logger.error(traceback.format_exc())
        embeddings = []
        metadata = []
        raise

@app.before_request
def before_request():
    global embedding_model, embeddings
    if embedding_model is None:
        load_data()

# Fonction pour traiter l'image et extraire l'embedding
def process_image(image_path, target_size=(224, 224)):
    try:
        # Utiliser la même méthode de traitement que dans votre script de génération
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        features = embedding_model(img_batch).numpy().flatten()
        # Normaliser le vecteur comme dans votre script de génération
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features
    except Exception as e:
        logger.error(f"Erreur de traitement d'image: {e}")
        logger.error(traceback.format_exc())
        return None

# Page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Route pour la recherche par image
@app.route('/api/search', methods=['POST'])
def search():
    try:
        start_time = time.time()

        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nom de fichier vide'}), 400

        try:
            num_results = int(request.form.get('num_results', app.config['DEFAULT_NUM_RESULTS']))
        except:
            num_results = app.config['DEFAULT_NUM_RESULTS']

        filename = secure_filename(f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        query_features = process_image(filepath)
        if query_features is None:
            return jsonify({'error': 'Impossible de traiter cette image'}), 400

        num_neighbors = min(num_results, len(embeddings))
        distances, indices = nn_model.kneighbors([query_features], n_neighbors=num_neighbors)

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
        logger.info(f"Recherche effectuée en {processing_time:.2f}s")

        return jsonify({
            'query_image': query_image_url,
            'results': results,
            'processing_time': processing_time,
            'count': len(results)
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/images/<filename>')
def database_image(filename):
    return send_from_directory('data/images', filename)

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
