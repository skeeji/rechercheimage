from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
import os
import json
import pickle
import logging
import gc
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Configuration GPU/CPU optimisée
tf.config.set_soft_device_placement(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

embedding_model = None
luminaire_embeddings = None
luminaire_metadata = None
is_loaded = False

def cleanup_memory():
    """Nettoyage mémoire agressif"""
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except:
        pass

def preprocess_image_advanced(image):
    """Preprocessing AVANCÉ pour meilleure précision"""
    try:
        # Conversion RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionnement avec préservation du ratio + padding
        original_size = image.size
        target_size = (224, 224)
        
        # Calcul du ratio optimal
        ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        
        # Redimensionnement haute qualité
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Création d'un canvas avec padding centré
        canvas = Image.new('RGB', target_size, (128, 128, 128))  # Gris neutre
        paste_x = (target_size[0] - new_size[0]) // 2
        paste_y = (target_size[1] - new_size[1]) // 2
        canvas.paste(image, (paste_x, paste_y))
        
        # Conversion en array avec normalisation MobileNet
        image_array = np.array(canvas, dtype=np.float32)
        image_array = image_array / 255.0  # [0, 1]
        image_array = (image_array - 0.5) * 2.0  # [-1, 1] comme MobileNet attend
        
        return np.expand_dims(image_array, axis=0)
        
    except Exception as e:
        logging.error(f"❌ Erreur preprocessing avancé: {e}")
        raise

def normalize_embeddings(embeddings):
    """Normalisation L2 optimisée"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Éviter division par zéro
    norms = np.where(norms == 0, 1e-8, norms)
    return embeddings / norms

def calculate_similarity_advanced(query_embedding, database_embeddings, method='cosine_weighted'):
    """Calcul de similarité AVANCÉ avec plusieurs métriques"""
    
    if method == 'cosine_weighted':
        # Similarité cosine avec pondération par magnitude
        dot_products = np.dot(database_embeddings, query_embedding)
        query_norm = np.linalg.norm(query_embedding)
        db_norms = np.linalg.norm(database_embeddings, axis=1)
        
        # Éviter division par zéro
        valid_mask = (db_norms > 1e-8) & (query_norm > 1e-8)
        similarities = np.zeros(len(database_embeddings))
        
        if np.any(valid_mask) and query_norm > 1e-8:
            cosine_sim = dot_products[valid_mask] / (db_norms[valid_mask] * query_norm)
            
            # Pondération par magnitude (features plus "riches" = plus importantes)
            magnitude_weight = np.log1p(db_norms[valid_mask] * query_norm) / 10.0
            weighted_sim = cosine_sim * (1.0 + magnitude_weight)
            
            similarities[valid_mask] = weighted_sim
            
    elif method == 'euclidean_normalized':
        # Distance euclidienne normalisée inversée
        distances = np.linalg.norm(database_embeddings - query_embedding, axis=1)
        max_distance = np.max(distances)
        similarities = 1 - (distances / (max_distance + 1e-8))
        
    else:  # cosine classique
        similarities = cosine_similarity([query_embedding], database_embeddings)[0]
    
    return similarities

def ensure_initialized():
    global embedding_model, luminaire_embeddings, luminaire_metadata, is_loaded

    if embedding_model is not None and luminaire_embeddings is not None and luminaire_metadata is not None:
        return True

    try:
        logging.info("🔄 Initialisation HAUTE PRÉCISION...")

        if embedding_model is None:
            logging.info("📥 Chargement MobileNet V2 optimisé...")
            embedding_model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
            
            # Test avec preprocessing optimal
            test_image = np.random.rand(224, 224, 3) * 255
            test_pil = Image.fromarray(test_image.astype('uint8'))
            test_batch = preprocess_image_advanced(test_pil)
            test_output = embedding_model(tf.constant(test_batch, dtype=tf.float32))
            logging.info(f"✅ Modèle chargé et testé: {test_output.shape}")

        embeddings_path = 'models/embeddings2.npy'
        if not os.path.exists(embeddings_path):
            logging.error("❌ Fichier embeddings2.npy manquant")
            return False

        if luminaire_embeddings is None:
            raw_embeddings = np.load(embeddings_path)
            # Normalisation L2 des embeddings de la base
            luminaire_embeddings = normalize_embeddings(raw_embeddings)
            logging.info(f"✅ Embeddings normalisés: {luminaire_embeddings.shape}")

        if luminaire_metadata is None:
            metadata_pkl_path = 'models/embeddings2.pkl'
            metadata_json_path = 'models/luminaires2.json'

            if os.path.exists(metadata_pkl_path):
                try:
                    with open(metadata_pkl_path, 'rb') as f:
                        raw_metadata = pickle.load(f)
                    
                    # Si c'est un array d'embeddings, créer des métadonnées artificielles
                    if isinstance(raw_metadata, np.ndarray):
                        luminaire_metadata = []
                        for i in range(len(raw_metadata)):
                            luminaire_metadata.append({
                                'id': f'luminaire_{i:06d}',
                                'name': f'Luminaire {i+1}',
                                'image_path': f'data/images/luminaire_{i:06d}.jpg',
                                'index': i
                            })
                        logging.info(f"✅ Métadonnées générées: {len(luminaire_metadata)} items")
                    else:
                        luminaire_metadata = raw_metadata
                        logging.info(f"✅ Métadonnées PKL: {len(luminaire_metadata)} items")
                        
                except Exception as e:
                    logging.warning(f"⚠️ Erreur PKL: {e}")
                    luminaire_metadata = None

            if luminaire_metadata is None and os.path.exists(metadata_json_path):
                with open(metadata_json_path, 'r', encoding='utf-8') as f:
                    luminaire_metadata = json.load(f)
                logging.info(f"✅ Métadonnées JSON: {len(luminaire_metadata)} items")

            if luminaire_metadata is None:
                # Création de métadonnées par défaut
                luminaire_metadata = []
                for i in range(len(luminaire_embeddings)):
                    luminaire_metadata.append({
                        'id': f'luminaire_{i:06d}',
                        'name': f'Luminaire {i+1}',
                        'image_path': f'data/images/luminaire_{i:06d}.jpg',
                        'category': 'Luminaire',
                        'index': i
                    })
                logging.info(f"✅ Métadonnées par défaut: {len(luminaire_metadata)} items")

        is_loaded = True
        cleanup_memory()
        logging.info("🎉 Initialisation HAUTE PRÉCISION complète!")
        return True

    except Exception as e:
        logging.error(f"❌ Erreur initialisation: {e}")
        cleanup_memory()
        return False

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.warning(f"Template manquant: {e}")
        return jsonify({
            'message': 'API de recherche HAUTE PRÉCISION',
            'version': '2.0 - Ultra-optimisée',
            'endpoints': {
                'status': '/status',
                'search': '/search (POST)',
                'api_search': '/api/search (POST)'
            }
        })

@app.route('/api')
def api_info():
    return jsonify({
        'message': 'API de recherche de luminaires HAUTE PRÉCISION',
        'endpoints': {
            'status': '/status',
            'search': '/search (POST)',
            'api_search': '/api/search (POST)'
        },
        'version': '2.0',
        'features': [
            'Preprocessing avancé avec padding centré',
            'Similarité cosine pondérée par magnitude',
            'Normalisation L2 optimisée', 
            'Seuils adaptatifs intelligents',
            'Gestion mémoire optimisée'
        ],
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
        'precision_mode': 'HAUTE_PRECISION_V2',
        'details': {
            'model_loaded': model_loaded,
            'embeddings_loaded': embeddings_loaded,
            'embeddings_normalized': True if embeddings_loaded else False,
            'metadata_loaded': metadata_loaded,
            'embeddings_count': len(luminaire_embeddings) if embeddings_loaded else 0,
            'metadata_count': len(luminaire_metadata) if metadata_loaded else 0
        }
    }

    if embeddings_loaded and model_loaded:
        status_info['details']['embeddings_shape'] = luminaire_embeddings.shape
        # Test de qualité des embeddings
        sample_norm = np.linalg.norm(luminaire_embeddings[0]) if len(luminaire_embeddings) > 0 else 0
        status_info['details']['embeddings_quality'] = 'normalized' if 0.9 <= sample_norm <= 1.1 else 'raw'

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
        return jsonify({'success': False, 'error': 'Pas d\'image fournie'}), 400

    try:
        logging.info("🔍 RECHERCHE HAUTE PRÉCISION v2.0...")

        # Chargement et preprocessing avancé de l'image
        image = Image.open(request.files['image'].stream)
        logging.info(f"📸 Image originale: {image.size}, mode: {image.mode}")
        
        image_batch = preprocess_image_advanced(image)
        logging.info(f"✅ Preprocessing avancé terminé: {image_batch.shape}")

        # Extraction des features avec le modèle
        query_features = embedding_model(tf.constant(image_batch, dtype=tf.float32))
        query_embedding = query_features.numpy()[0]
        
        # Normalisation L2 de la requête
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 1e-8:
            query_embedding = query_embedding / query_norm
        
        logging.info(f"✅ Features extraites et normalisées: {query_embedding.shape}, norme: {np.linalg.norm(query_embedding):.3f}")

        # Vérification de compatibilité des dimensions
        if luminaire_embeddings.shape[1] != query_embedding.shape[0]:
            logging.warning(f"⚠️ Mismatch dimensions: {query_embedding.shape[0]} vs {luminaire_embeddings.shape[1]}")
            # Troncature ou padding intelligent
            min_dim = min(luminaire_embeddings.shape[1], query_embedding.shape[0])
            database_embeddings = luminaire_embeddings[:, :min_dim]
            query_embedding = query_embedding[:min_dim]
            logging.info(f"🔧 Ajustement dimensions: {min_dim}D")
        else:
            database_embeddings = luminaire_embeddings

        # Calcul de similarité AVANCÉ avec méthode pondérée
        logging.info("🎯 Calcul similarité haute précision...")
        similarities = calculate_similarity_advanced(query_embedding, database_embeddings, method='cosine_weighted')
        
        # Seuils adaptatifs intelligents
        valid_similarities = similarities[similarities > 0]
        if len(valid_similarities) > 0:
            mean_sim = np.mean(valid_similarities)
            std_sim = np.std(valid_similarities)
            q75 = np.percentile(valid_similarities, 75)
            
            # Seuil adaptatif sophistiqué
            adaptive_threshold = max(0.3, min(0.6, mean_sim + 0.5 * std_sim, q75 - 0.1))
        else:
            adaptive_threshold = 0.3
            
        logging.info(f"📊 Stats: mean={np.mean(valid_similarities):.3f}, std={np.std(valid_similarities):.3f}, seuil={adaptive_threshold:.3f}")

        # Sélection des TOP résultats avec seuil intelligent
        top_k = min(50, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        
        # Filtrage par seuil adaptatif
        filtered_indices = []
        for idx in top_indices:
            if similarities[idx] >= adaptive_threshold:
                filtered_indices.append(idx)
                if len(filtered_indices) >= 15:  # Limite intelligente
                    break

        # Construction des résultats avec scoring avancé
        results = []
        for i, idx in enumerate(filtered_indices):
            if idx >= len(luminaire_metadata):
                continue

            metadata = luminaire_metadata[idx] if isinstance(luminaire_metadata[idx], dict) else {}
            similarity_raw = similarities[idx]
            
            # Scoring sophistiqué
            confidence = min(100, max(0, int((similarity_raw - 0.2) * 125)))
            
            # Qualité basée sur quartiles
            if similarity_raw >= 0.8:
                quality, quality_score = 'excellent', 95
            elif similarity_raw >= 0.65:
                quality, quality_score = 'très_bon', 85
            elif similarity_raw >= 0.5:
                quality, quality_score = 'bon', 75
            elif similarity_raw >= 0.35:
                quality, quality_score = 'correct', 60
            else:
                quality, quality_score = 'faible', 40

            result_item = {
                'rank': i + 1,
                'similarity': round(similarity_raw, 4),
                'similarity_percent': round(similarity_raw * 100, 1),
                'confidence': confidence,
                'quality': quality,
                'quality_score': quality_score,
                'metadata': {
                    'id': metadata.get('id', f'luminaire_{idx:06d}'),
                    'name': metadata.get('name', f'Luminaire {idx+1}'),
                    'description': metadata.get('description', ''),
                    'price': float(metadata.get('price', 0.0)),
                    'category': metadata.get('category', 'Luminaire'),
                    'style': metadata.get('style', ''),
                    'material': metadata.get('material', ''),
                    'image_path': metadata.get('image_path', f'data/images/luminaire_{idx:06d}.jpg'),
                    'index': idx
                }
            }
            results.append(result_item)

        cleanup_memory()

        best_score = results[0]['similarity'] if results else 0
        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        
        logging.info(f"🎉 RECHERCHE TERMINÉE: {len(results)} résultats haute précision, meilleur: {best_score:.4f}")

        return jsonify({
            'success': True,
            'results': results,
            'message': f'{len(results)} résultats haute précision trouvés',
            'stats': {
                'total_searched': len(database_embeddings),
                'results_count': len(results),
                'best_similarity': round(best_score, 4),
                'avg_confidence': round(avg_confidence, 1),
                'adaptive_threshold': round(adaptive_threshold, 3),
                'query_dimensions': query_embedding.shape[0],
                'db_dimensions': database_embeddings.shape[1],
                'precision_mode': 'HAUTE_PRECISION_V2',
                'preprocessing': 'avancé_avec_padding',
                'similarity_method': 'cosine_pondéré'
            }
        })

    except Exception as e:
        cleanup_memory()
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
    cleanup_memory()
    return jsonify({'success': False, 'error': 'Erreur serveur'}), 500

if __name__ == '__main__':
    logging.info("🚀 Démarrage serveur HAUTE PRÉCISION v2.0...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
