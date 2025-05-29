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
            # NOUVEAU MODÈLE PLUS PRÉCIS (compatible TF 2.11)
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

# NOUVEAU PREPROCESSING OPTIMISÉ POUR LUMINAIRES (sans scipy)
def preprocess_image_advanced(image):
    """Preprocessing avancé pour luminaires - compatible toutes versions"""
    # Resize avec conservation des proportions
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Conversion en array
    image_np = np.array(image, dtype=np.float32)
    
    # Normalisation EfficientNet standard
    image_np = image_np / 255.0
    
    # Amélioration spécifique aux luminaires
    # 1. Augmentation du contraste pour les détails métalliques/formes
    image_np = np.clip(image_np * 1.4 - 0.2, 0.0, 1.0)
    
    # 2. Amélioration gamma pour better dynamic range
    gamma = 0.8
    image_np = np.power(image_np, gamma)
    
    # 3. Amélioration des contours (version simple sans scipy)
    try:
        # Filtre de netteté basique avec convolution manuelle
        h, w, c = image_np.shape
        sharpened = np.copy(image_np)
        
        # Kernel de netteté
        for i in range(1, h-1):
            for j in range(1, w-1):
                for ch in range(c):
                    # Filtre de netteté 3x3
                    center = image_np[i, j, ch]
                    neighbors = (image_np[i-1, j, ch] + image_np[i+1, j, ch] + 
                               image_np[i, j-1, ch] + image_np[i, j+1, ch]) / 4
                    sharpened[i, j, ch] = center + 0.5 * (center - neighbors)
        
        image_np = np.clip(sharpened, 0.0, 1.0)
    except:
        # Si erreur, on garde l'image sans sharpening
        pass
    
    # 4. Ajustement final de la saturation pour les couleurs métalliques
    # Conversion RGB vers HSV simplifiée
    max_vals = np.max(image_np, axis=2)
    min_vals = np.min(image_np, axis=2)
    diff = max_vals - min_vals
    saturation_boost = np.where(diff > 0.1, 1.2, 1.0)  # Boost si coloré
    
    # Application du boost
    for ch in range(3):
        image_np[:,:,ch] = np.clip(image_np[:,:,ch] * saturation_boost, 0.0, 1.0)
    
    return np.expand_dims(image_np, axis=0)

# NOUVELLE FONCTION DE SIMILARITÉ HYBRIDE OPTIMISÉE
def calculate_similarity_hybrid(query_embedding, database_embeddings):
    """Calcul de similarité hybride optimisé (cosine + euclidean + dot product)"""
    
    # Normalisation L2 robuste
    query_norm_val = np.linalg.norm(query_embedding)
    if query_norm_val > 0:
        query_normalized = query_embedding / query_norm_val
    else:
        query_normalized = query_embedding
    
    db_norms = np.linalg.norm(database_embeddings, axis=1)
    db_normalized = np.zeros_like(database_embeddings)
    
    # Normalisation robuste pour la DB
    for i in range(len(database_embeddings)):
        if db_norms[i] > 0:
            db_normalized[i] = database_embeddings[i] / db_norms[i]
        else:
            db_normalized[i] = database_embeddings[i]
    
    # 1. Similarité cosinus (principale pour luminaires)
    cosine_sim = np.dot(db_normalized, query_normalized)
    cosine_sim = np.clip(cosine_sim, -1, 1)  # Clamp pour stabilité
    
    # 2. Distance euclidienne normalisée
    euclidean_distances = np.linalg.norm(db_normalized - query_normalized, axis=1)
    euclidean_sim = 1.0 / (1.0 + euclidean_distances)
    
    # 3. Dot product brut (pour capturer magnitude)
    dot_products = np.dot(database_embeddings, query_embedding)
    max_dot = np.max(np.abs(dot_products))
    if max_dot > 0:
        dot_sim = np.abs(dot_products) / max_dot
    else:
        dot_sim = np.zeros_like(dot_products)
    
    # Combinaison pondérée optimisée pour luminaires
    # 60% cosine (forme), 25% euclidean (détails), 15% magnitude
    hybrid_similarity = (0.6 * cosine_sim + 
                        0.25 * euclidean_sim + 
                        0.15 * dot_sim)
    
    return hybrid_similarity

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.warning(f"Template manquant: {e}")
        return jsonify({
            'message': 'API de recherche de luminaires AMÉLIORÉE v2.0',
            'model': 'EfficientNet Lite + Preprocessing Avancé',
            'compatible': 'TensorFlow 2.11',
            'endpoints': {
                'status': '/status',
                'search': '/search (POST)',
                'api_search': '/api/search (POST)'
            }
        })

@app.route('/api')
def api_info():
    return jsonify({
        'message': 'API de recherche de luminaires AMÉLIORÉE v2.0',
        'model': 'EfficientNet Lite + Preprocessing Avancé',
        'improvements': [
            'EfficientNet Lite (vs MobileNet)',
            'Preprocessing optimisé luminaires',
            'Similarité hybride triple (cosine+euclidean+dot)',
            'Seuil de qualité adaptatif',
            'Amélioration contraste/netteté',
            'Compatible TF 2.11'
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
        'tensorflow_version': tf.__version__,
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
        logging.info("🔍 Nouvelle recherche AMÉLIORÉE v2.0...")
        
        # NOUVEAU PREPROCESSING AVANCÉ
        image = Image.open(request.files['image'].stream).convert('RGB')
        image_batch = preprocess_image_advanced(image)
        
        # Extraction features avec EfficientNet
        query_features = embedding_model(tf.constant(image_batch, dtype=tf.float32))
        query_embedding = query_features.numpy()[0]
        
        logging.info(f"Query: {query_embedding.shape}, DB: {luminaire_embeddings.shape}")
        
        # Gestion intelligente des dimensions
        if luminaire_embeddings.shape[1] != query_embedding.shape[0]:
            logging.warning(f"⚠️ Mismatch dimensions: {query_embedding.shape[0]} vs {luminaire_embeddings.shape[1]}")
            
            min_dim = min(luminaire_embeddings.shape[1], query_embedding.shape[0])
            database_embeddings = luminaire_embeddings[:, :min_dim]
            query_embedding = query_embedding[:min_dim]
            logging.info(f"✅ Dimensions alignées: {min_dim}D")
        else:
            database_embeddings = luminaire_embeddings
        
        # CALCUL SIMILARITÉ HYBRIDE OPTIMISÉ
        logging.info("🧮 Calcul similarité hybride triple...")
        max_items = min(len(database_embeddings), 5000)
        
        if max_items <= 1000:
            # Calcul direct pour petit dataset
            similarities = calculate_similarity_hybrid(query_embedding, database_embeddings[:max_items])
        else:
            # Calcul par chunks optimisé
            chunk_size = 500
            similarities = np.zeros(max_items)
            
            for i in range(0, max_items, chunk_size):
                end_idx = min(i + chunk_size, max_items)
                chunk = database_embeddings[i:end_idx]
                similarities[i:end_idx] = calculate_similarity_hybrid(query_embedding, chunk)
                
                if i % 1000 == 0:
                    logging.info(f"Progrès: {i}/{max_items}")
        
        # SEUIL DE QUALITÉ ADAPTATIF AMÉLIORÉ
        top_indices = np.argsort(similarities)[::-1][:30]  # Top 30 d'abord
        
        # Seuil dynamique basé sur la distribution
        top_scores = similarities[top_indices[:10]]
        base_threshold = 0.4  # Seuil minimum
        adaptive_threshold = max(base_threshold, np.percentile(similarities, 85) * 0.7)
        
        # Si le meilleur score est très bon, on monte le seuil
        if len(top_scores) > 0 and top_scores[0] > 0.8:
            adaptive_threshold = max(adaptive_threshold, top_scores[0] * 0.6)
        
        logging.info(f"🎯 Seuil adaptatif: {adaptive_threshold:.3f} (base: {base_threshold})")
        
        # Filtrage intelligent
        filtered_indices = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= adaptive_threshold and len(filtered_indices) < 10:
                filtered_indices.append(idx)
            elif len(filtered_indices) < 5 and score >= base_threshold:
                # Garde au moins 5 résultats même si scores plus bas
                filtered_indices.append(idx)
        
        # Si trop peu de résultats, on baisse le seuil
        if len(filtered_indices) < 3:
            filtered_indices = top_indices[:8]
            logging.info("🔽 Seuil abaissé pour garantir des résultats")
        
        results = []
        for i, idx in enumerate(filtered_indices):
            if idx >= len(luminaire_metadata):
                continue
                
            metadata = luminaire_metadata[idx] if isinstance(luminaire_metadata[idx], dict) else {}
            similarity_score = similarities[idx]
            
            # SCORING AMÉLIORÉ
            # Normalisation 0-100 avec courbe adaptée aux luminaires
            raw_confidence = (similarity_score - base_threshold) / (1.0 - base_threshold) * 100
            confidence_score = max(0, min(100, raw_confidence))
            
            # Ajustement courbe pour luminaires (boost scores moyens)
            if confidence_score > 20:
                confidence_score = min(100, confidence_score * 1.2)
            
            # Classification qualité
            if confidence_score > 75:
                quality = 'excellent'
            elif confidence_score > 55:
                quality = 'good'
            elif confidence_score > 35:
                quality = 'fair'
            else:
                quality = 'low'
            
            result_item = {
                'rank': i + 1,
                'similarity': round(similarity_score * 100, 1),
                'confidence': round(confidence_score, 1),
                'quality': quality,
                'metadata': {
                    'id': metadata.get('id', str(idx)),
                    'name': metadata.get('name', f'Luminaire {idx}'),
                    'description': metadata.get('description', ''),
                    'price': float(metadata.get('price', 0.0)) if metadata.get('price') else 0.0,
                    'category': metadata.get('category', ''),
                    'style': metadata.get('style', ''),
                    'material': metadata.get('material', ''),
                    'image_path': metadata.get('image_path', f'data/images/{idx}.jpg')
                }
            }
            results.append(result_item)

        # STATS FINALES
        best_score = results[0]['similarity'] if results else 0
        best_confidence = results[0]['confidence'] if results else 0
        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        excellent_count = len([r for r in results if r['quality'] == 'excellent'])
        
        logging.info(f"✅ Recherche AMÉLIORÉE terminée: {len(results)} résultats")
        logging.info(f"📊 Meilleur: {best_score:.1f}% (confiance {best_confidence:.1f}%)")
        logging.info(f"📊 {excellent_count} excellents, confiance moyenne: {avg_confidence:.1f}%")

        return jsonify({
            'success': True,
            'results': results,
            'message': f'{len(results)} résultats trouvés',
            'model_info': {
                'name': 'EfficientNet Lite',
                'version': '2.0',
                'tensorflow_version': tf.__version__
            },
            'improvements': {
                'preprocessing': 'Contraste/netteté optimisés luminaires',
                'similarity': 'Triple hybride (cosine+euclidean+dot)',
                'threshold': 'Adaptatif avec seuil minimum',
                'scoring': 'Courbe ajustée pour luminaires'
            },
            'quality_stats': {
                'excellent': len([r for r in results if r['quality'] == 'excellent']),
                'good': len([r for r in results if r['quality'] == 'good']),
                'fair': len([r for r in results if r['quality'] == 'fair']),
                'low': len([r for r in results if r['quality'] == 'low'])
            },
            'stats': {
                'total_searched': max_items,
                'results_count': len(results),
                'best_similarity': best_score,
                'best_confidence': round(best_confidence, 1),
                'avg_confidence': round(avg_confidence, 1),
                'adaptive_threshold': round(adaptive_threshold * 100, 1),
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
    logging.info("🚀 Démarrage serveur AMÉLIORÉ v2.0...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
