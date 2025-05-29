from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
import os
import json
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

app = Flask(__name__)

embedding_model = None
luminaire_embeddings = None
luminaire_metadata = None
is_loaded = False

def ensure_initialized():
    global embedding_model, luminaire_embeddings, luminaire_metadata, is_loaded
    
    if embedding_model is not None and luminaire_embeddings is not None and luminaire_metadata is not None:
        return True
        
    try:
        logging.info("🔄 Initialisation...")
        
        if embedding_model is None:
            logging.info("📥 Chargement EfficientNet Lite...")
            # NOUVEAU MODÈLE PLUS PRÉCIS
            embedding_model = hub.load("https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2")
            test_output = embedding_model(tf.constant(np.random.rand(1, 224, 224, 3), dtype=tf.float32))
            logging.info(f"✅ Modèle chargé, dimensions: {test_output.shape}")

        embeddings_path = 'models/embeddings2.npy'
        if not os.path.exists(embeddings_path):
            logging.error("❌ Fichier embeddings.npy manquant")
            return False
        
        if luminaire_embeddings is None:
            luminaire_embeddings = np.load(embeddings_path)
            logging.info(f"✅ Embeddings chargés: {luminaire_embeddings.shape}")

        if luminaire_metadata is None:
            metadata_pkl_path = 'models/embeddings2.pkl'
            metadata_json_path = 'models/luminaires2.json'
            
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

# NOUVEAU PREPROCESSING OPTIMISÉ POUR LUMINAIRES
def preprocess_image_advanced(image):
    """Preprocessing avancé pour luminaires"""
    # Resize avec conservation des proportions
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Conversion en array
    image_np = np.array(image, dtype=np.float32)
    
    # Normalisation EfficientNet standard
    image_np = image_np / 255.0
    
    # Amélioration spécifique aux luminaires
    # Augmentation du contraste pour les détails métalliques
    image_np = np.clip(image_np * 1.3 - 0.15, 0.0, 1.0)
    
    # Amélioration des contours
    from scipy import ndimage
    try:
        # Filtre de netteté léger
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        
        for i in range(3):  # Pour chaque canal RGB
            image_np[:,:,i] = ndimage.convolve(image_np[:,:,i], kernel)
        
        image_np = np.clip(image_np, 0.0, 1.0)
    except:
        pass  # Si scipy pas disponible, on continue
    
    return np.expand_dims(image_np, axis=0)

# NOUVELLE FONCTION DE SIMILARITÉ HYBRIDE
def calculate_similarity_hybrid(query_embedding, database_embeddings):
    """Calcul de similarité hybride (cosine + euclidean)"""
    
    # Normalisation L2
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    db_norms = database_embeddings / (np.linalg.norm(database_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Similarité cosinus (0-1)
    cosine_sim = np.dot(db_norms, query_norm)
    
    # Distance euclidienne normalisée (0-1)
    euclidean_dist = np.linalg.norm(db_norms - query_norm, axis=1)
    euclidean_sim = 1 / (1 + euclidean_dist)
    
    # Combinaison pondérée (70% cosine, 30% euclidean)
    hybrid_similarity = 0.7 * cosine_sim + 0.3 * euclidean_sim
    
    return hybrid_similarity

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.warning(f"Template manquant: {e}")
        return jsonify({
            'message': 'API de recherche de luminaires AMÉLIORÉE',
            'model': 'EfficientNet Lite',
            'endpoints': {
                'status': '/status',
                'search': '/search (POST)',
                'api_search': '/api/search (POST)'
            }
        })

@app.route('/api')
def api_info():
    return jsonify({
        'message': 'API de recherche de luminaires AMÉLIORÉE',
        'model': 'EfficientNet Lite + Preprocessing Avancé',
        'improvements': [
            'Modèle EfficientNet plus précis',
            'Preprocessing optimisé luminaires',
            'Similarité hybride cosine+euclidean',
            'Seuil de qualité adaptatif'
        ],
        'endpoints': {
            'status': '/status',
            'search': '/search (POST)',
            'api_search': '/api/search (POST)'
        },
        'version': '2.0',
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
        'model_type': 'EfficientNet Lite',
        'version': '2.0 - Optimisé Luminaires',
        'details': {
            'model_loaded': model_loaded,
            'embeddings_loaded': embeddings_loaded,
            'metadata_loaded': metadata_loaded,
            'embeddings_count': len(luminaire_embeddings) if embeddings_loaded else 0,
            'metadata_count': len(luminaire_metadata) if metadata_loaded else 0
        }
    }
    
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
    if not ensure_initialized():
        return jsonify({'success': False, 'error': 'Système non initialisé'}), 503
        
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'success': False, 'error': 'Pas d\'image'}), 400

    try:
        logging.info("🔍 Nouvelle recherche AMÉLIORÉE...")
        
        # NOUVEAU PREPROCESSING
        image = Image.open(request.files['image'].stream).convert('RGB')
        image_batch = preprocess_image_advanced(image)
        
        # Extraction features avec EfficientNet
        query_features = embedding_model(tf.constant(image_batch, dtype=tf.float32))
        query_embedding = query_features.numpy()[0]
        
        logging.info(f"Query: {query_embedding.shape}, DB: {luminaire_embeddings.shape}")
        
        # Gestion des dimensions
        if luminaire_embeddings.shape[1] != query_embedding.shape[0]:
            logging.warning(f"⚠️ Mismatch dimensions: {query_embedding.shape[0]} vs {luminaire_embeddings.shape[1]}")
            
            if luminaire_embeddings.shape[1] > query_embedding.shape[0]:
                database_embeddings = luminaire_embeddings[:, :query_embedding.shape[0]]
                logging.info(f"✅ DB tronquée: {luminaire_embeddings.shape[1]}D → {query_embedding.shape[0]}D")
            else:
                padding_size = luminaire_embeddings.shape[1] - query_embedding.shape[0]
                padding = np.zeros(padding_size, dtype=np.float32)
                query_embedding = np.concatenate([query_embedding, padding])
                database_embeddings = luminaire_embeddings
                logging.info(f"✅ Query paddé: → {query_embedding.shape[0]}D")
        else:
            database_embeddings = luminaire_embeddings
        
        # NOUVEAU CALCUL DE SIMILARITÉ HYBRIDE
        logging.info("🧮 Calcul similarité hybride...")
        max_items = min(len(database_embeddings), 5000)
        
        if max_items <= 2000:
            # Calcul direct si petit dataset
            similarities = calculate_similarity_hybrid(query_embedding, database_embeddings[:max_items])
        else:
            # Calcul par chunks si gros dataset
            chunk_size = 500
            similarities = np.zeros(max_items)
            
            for i in range(0, max_items, chunk_size):
                end_idx = min(i + chunk_size, max_items)
                chunk = database_embeddings[i:end_idx]
                similarities[i:end_idx] = calculate_similarity_hybrid(query_embedding, chunk)
                
                if i % 2000 == 0:
                    logging.info(f"Progrès: {i}/{max_items}")
        
        # NOUVEAU SEUIL DE QUALITÉ ADAPTATIF
        top_indices = np.argsort(similarities)[::-1][:20]  # Top 20 d'abord
        
        # Filtrage par seuil de qualité
        quality_threshold = max(0.3, np.percentile(similarities, 95) * 0.6)
        logging.info(f"🎯 Seuil qualité: {quality_threshold:.3f}")
        
        filtered_indices = [idx for idx in top_indices if similarities[idx] >= quality_threshold][:10]
        
        results = []
        for i, idx in enumerate(filtered_indices):
            if idx >= len(luminaire_metadata):
                continue
                
            metadata = luminaire_metadata[idx] if isinstance(luminaire_metadata[idx], dict) else {}
            similarity_score = similarities[idx]
            
            # NOUVEAU SCORE DE CONFIANCE
            confidence_score = min(100, max(0, (similarity_score - 0.3) / 0.7 * 100))
            
            result_item = {
                'rank': i + 1,
                'similarity': round(similarity_score * 100, 1),
                'confidence': round(confidence_score, 1),
                'quality': 'excellent' if confidence_score > 80 else 'good' if confidence_score > 60 else 'fair',
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
        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        
        logging.info(f"✅ Recherche AMÉLIORÉE terminée: {len(results)} résultats, meilleur: {best_score}%, confiance moy: {avg_confidence:.1f}%")

        return jsonify({
            'success': True,
            'results': results,
            'message': f'{len(results)} résultats de qualité',
            'improvements': {
                'model': 'EfficientNet Lite',
                'preprocessing': 'Optimisé luminaires',
                'similarity': 'Hybride cosine+euclidean',
                'quality_filter': f'Seuil {quality_threshold:.2f}'
            },
            'stats': {
                'total_searched': max_items,
                'results_count': len(results),
                'best_similarity': best_score,
                'avg_confidence': round(avg_confidence, 1),
                'quality_threshold': round(quality_threshold * 100, 1),
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
    logging.info("🚀 Démarrage du serveur AMÉLIORÉ...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
