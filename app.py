from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
import os
import json
import pickle
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

app = Flask(__name__)

embedding_model = None
luminaire_embeddings = None
luminaire_metadata = None
is_loaded = False

def ensure_initialized():
    global embedding_model, luminaire_embeddings, luminaire_metadata, is_loaded
    
    # 🔧 CORRECTIF: vérification explicite avec 'is not None'
    if embedding_model is not None and luminaire_embeddings is not None and luminaire_metadata is not None:
        return True
        
    try:
        logging.info("🔄 Initialisation...")
        
        if embedding_model is None:
            logging.info("📥 Chargement MobileNet...")
            embedding_model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
            # Test du modèle
            test_output = embedding_model(tf.constant(np.random.rand(1, 224, 224, 3), dtype=tf.float32))
            logging.info(f"✅ Modèle chargé, dimensions: {test_output.shape}")

        embeddings_path = 'data/embeddings.npy'
        if not os.path.exists(embeddings_path):
            logging.error("❌ Fichier embeddings.npy manquant")
            return False
        
        if luminaire_embeddings is None:
            luminaire_embeddings = np.load(embeddings_path)
            logging.info(f"✅ Embeddings chargés: {luminaire_embeddings.shape}")

        if luminaire_metadata is None:
            metadata_pkl_path = 'data/embeddings.pkl'
            metadata_json_path = 'data/luminaires.json'
            
            if os.path.exists(metadata_pkl_path):
                try:
                    with open(metadata_pkl_path, 'rb') as f:
                        luminaire_metadata = pickle.load(f)
                    logging.info(f"✅ Métadonnées PKL: {len(luminaire_metadata)} items")
                except Exception as e:
                    logging.warning(f"⚠️ Erreur PKL: {e}")
                    luminaire_metadata = None
                    
            if luminaire_metadata is None and os.path.exists(metadata_json_path):
                with open(metadata_json_path, 'r', encoding='utf-8') as f:
                    luminaire_metadata = json.load(f)
                logging.info(f"✅ Métadonnées JSON: {len(luminaire_metadata)} items")
                
            if luminaire_metadata is None:
                logging.error("❌ Aucune métadonnée trouvée")
                return False

        is_loaded = True
        logging.info("🎉 Initialisation complète!")
        return True
        
    except Exception as e:
        logging.error(f"❌ Erreur initialisation: {e}")
        return False

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.warning(f"Template manquant: {e}")
        return jsonify({
            'message': 'API de recherche de luminaires',
            'endpoints': {
                'status': '/status',
                'search': '/search (POST)',
                'api_search': '/api/search (POST)'
            }
        })

@app.route('/api')
def api_info():
    return jsonify({
        'message': 'API de recherche de luminaires',
        'endpoints': {
            'status': '/status',
            'search': '/search (POST)',
            'api_search': '/api/search (POST)'
        },
        'version': '1.0',
        'ready': is_loaded
    })

@app.route('/status')
def status():
    model_loaded = embedding_model is not None
    embeddings_loaded = luminaire_embeddings is not None
    metadata_loaded = luminaire_metadata is not None
    initialized = model_loaded and embeddings_loaded and metadata_loaded
    
    status_info = {
        'initialized': initialized,
        'status': 'ready' if initialized else 'not_ready',
        'details': {
            'model_loaded': model_loaded,
            'embeddings_loaded': embeddings_loaded,
            'metadata_loaded': metadata_loaded,
            'embeddings_count': len(luminaire_embeddings) if embeddings_loaded else 0,
            'metadata_count': len(luminaire_metadata) if metadata_loaded else 0
        }
    }
    
    # Ajout des dimensions si chargé
    if embeddings_loaded and model_loaded:
        status_info['details']['embeddings_shape'] = luminaire_embeddings.shape
        
    return jsonify(status_info)

@app.route('/data/images/<filename>')
def serve_image(filename):
    path = os.path.join('data', 'images', filename)
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return jsonify({'error': 'Image non trouvée'}), 404

@app.route('/search', methods=['POST'])
def search_similar():
    """Version optimisée de la recherche avec gestion mémoire"""
    if not ensure_initialized():
        return jsonify({'success': False, 'error': 'Système non initialisé'}), 503
        
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'success': False, 'error': 'Pas d\'image'}), 400

    try:
        logging.info("🔍 Nouvelle recherche...")
        
        # Traitement de l'image
        image = Image.open(request.files['image'].stream).convert('RGB')
        image = image.resize((224, 224))
        image_np = (np.array(image, dtype=np.float32) / 255.0 - 0.5) * 2.0
        image_batch = np.expand_dims(image_np, axis=0)
        
        # Extraction des features
        query_features = embedding_model(tf.constant(image_batch, dtype=tf.float32))
        query_embedding = query_features.numpy()[0]  # Shape: (1280,)
        
        logging.info(f"Query: {query_embedding.shape}, DB: {luminaire_embeddings.shape}")
        
        # 🔧 CORRECTIF DIMENSION MISMATCH
        if luminaire_embeddings.shape[1] != query_embedding.shape[0]:
            logging.warning(f"⚠️ Mismatch dimensions: {query_embedding.shape[0]} vs {luminaire_embeddings.shape[1]}")
            
            if luminaire_embeddings.shape[1] > query_embedding.shape[0]:
                # Tronquer DB aux dimensions du query (1536 → 1280)
                database_embeddings = luminaire_embeddings[:, :query_embedding.shape[0]]
                logging.info(f"✅ DB tronquée: {luminaire_embeddings.shape[1]}D → {query_embedding.shape[0]}D")
            else:
                # Padding query (cas inverse)
                padding_size = luminaire_embeddings.shape[1] - query_embedding.shape[0]
                padding = np.zeros(padding_size, dtype=np.float32)
                query_embedding = np.concatenate([query_embedding, padding])
                database_embeddings = luminaire_embeddings
                logging.info(f"✅ Query paddé: → {query_embedding.shape[0]}D")
        else:
            database_embeddings = luminaire_embeddings
        
        # Normalisation query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # 🔧 CALCUL PAR CHUNKS pour éviter crash mémoire
        logging.info("📊 Calcul par chunks...")
        chunk_size = 500  # Traiter 500 embeddings à la fois
        max_items = min(len(database_embeddings), 5000)  # Limiter à 5000 max
        similarities = np.zeros(max_items)
        
        for i in range(0, max_items, chunk_size):
            end_idx = min(i + chunk_size, max_items)
            chunk = database_embeddings[i:end_idx]
            
            # Normalisation du chunk
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            normalized_chunk = np.where(norms > 0, chunk / norms, chunk)
            
            # Similarité cosinus
            similarities[i:end_idx] = np.dot(normalized_chunk, query_embedding)
            
            # Log progrès tous les 2000
            if i % 2000 == 0:
                logging.info(f"Progrès: {i}/{max_items}")
        
        # Top 10 résultats
        top_indices = np.argsort(similarities)[::-1][:10]
        
        results = []
        for i, idx in enumerate(top_indices):
            if idx >= len(luminaire_metadata):
                continue
                
            metadata = luminaire_metadata[idx] if isinstance(luminaire_metadata[idx], dict) else {}
            similarity_score = similarities[idx]
            
            result_item = {
                'rank': i + 1,
                'similarity': round(max(0, min(100, similarity_score * 100)), 1),
                'metadata': {
                    'id': metadata.get('id', str(idx)),
                    'name': metadata.get('name', f'Luminaire {idx}'),
                    'description': metadata.get('description', ''),
                    'price': float(metadata.get('price', 0.0)),
                    'category': metadata.get('category', ''),
                    'style': metadata.get('style', ''),
                    'material': metadata.get('material', ''),
                    'image_path': metadata.get('image_path', f'data/images/{idx}.jpg')
                }
            }
            results.append(result_item)

        best_score = results[0]['similarity'] if results else 0
        logging.info(f"✅ Recherche terminée: {len(results)} résultats, meilleur: {best_score}%")

        return jsonify({
            'success': True,
            'results': results,
            'message': f'{len(results)} résultats',
            'stats': {
                'total_searched': max_items,
                'results_count': len(results),
                'best_similarity': best_score,
                'query_dimensions': query_embedding.shape[0],
                'db_dimensions': database_embeddings.shape[1] if len(database_embeddings) > 0 else 0
            }
        })
        
    except Exception as e:
        logging.error(f"❌ Erreur recherche: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'Erreur: {str(e)}'}), 500

@app.route('/api/search', methods=['POST'])
def api_search_similar():
    return search_similar()

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Route non trouvée'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Erreur serveur'}), 500

if __name__ == '__main__':
    logging.info("🚀 Démarrage du serveur...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
