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
            logging.info("📥 Chargement MobileNet V2 (compatible)...")
            # RETOUR À MOBILENET MAIS OPTIMISÉ
            embedding_model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
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

# 🎯 PREPROCESSING OPTIMISÉ POUR LUMINAIRES
def enhanced_preprocess_image(image):
    """Preprocessing spécial pour luminaires avec MobileNet optimisé"""
    try:
        # Resize optimal
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize intelligent avec aspect ratio
        target_size = (224, 224)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Conversion en array
        img_array = np.array(image, dtype=np.float32)
        
        # 🔥 OPTIMISATIONS SPÉCIALES LUMINAIRES
        # 1. Normalisation améliorée
        img_array = img_array / 255.0
        
        # 2. Amélioration du contraste pour luminaires
        img_array = np.clip(img_array * 1.1 + 0.05, 0, 1)
        
        # 3. Correction gamma pour meilleure détection
        img_array = np.power(img_array, 0.9)
        
        # 4. Normalisation finale MobileNet
        img_array = (img_array - 0.5) * 2.0
        
        # Batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        logging.info(f"✅ Image preprocessée: {img_batch.shape}")
        return img_batch
        
    except Exception as e:
        logging.error(f"❌ Erreur preprocessing: {e}")
        return None

# 🔥 SIMILARITÉ HYBRIDE OPTIMISÉE
def calculate_enhanced_similarity(query_embedding, database_embeddings):
    """Calcul de similarité hybride optimisé pour luminaires"""
    try:
        # Normalisation L2
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        db_norms = database_embeddings / (np.linalg.norm(database_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # 1. Similarité cosine (poids principal)
        cosine_sim = np.dot(db_norms, query_norm.T).flatten()
        
        # 2. Similarité euclidienne inversée (pour distance)
        euclidean_dist = np.linalg.norm(db_norms - query_norm, axis=1)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # 3. Similarité dot product
        dot_sim = np.dot(database_embeddings, query_embedding.T).flatten()
        dot_sim = (dot_sim - np.min(dot_sim)) / (np.max(dot_sim) - np.min(dot_sim) + 1e-8)
        
        # 🎯 COMBINAISON OPTIMISÉE POUR LUMINAIRES
        # Cosine = 60%, Euclidean = 30%, Dot = 10%
        hybrid_similarity = (
            0.6 * cosine_sim + 
            0.3 * euclidean_sim + 
            0.1 * dot_sim
        )
        
        return hybrid_similarity
        
    except Exception as e:
        logging.error(f"❌ Erreur calcul similarité: {e}")
        return np.array([])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_similar():
    try:
        if not ensure_initialized():
            return jsonify({'success': False, 'error': 'Modèle non initialisé'}), 503

        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Aucune image fournie'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Aucun fichier sélectionné'}), 400

        logging.info("🔍 Nouvelle recherche OPTIMISÉE v2.0...")
        
        # Chargement et preprocessing de l'image
        try:
            image = Image.open(file.stream)
            logging.info(f"📸 Image chargée: {image.size}, mode: {image.mode}")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Image invalide: {str(e)}'}), 400

        # Preprocessing optimisé
        image_batch = enhanced_preprocess_image(image)
        if image_batch is None:
            return jsonify({'success': False, 'error': 'Erreur preprocessing image'}), 500

        # Extraction des features
        try:
            query_features = embedding_model(tf.constant(image_batch, dtype=tf.float32))
            query_embedding = query_features.numpy().flatten()
            logging.info(f"✅ Features extraites: {query_embedding.shape}")
        except Exception as e:
            logging.error(f"❌ Erreur extraction: {e}")
            return jsonify({'success': False, 'error': f'Erreur extraction features: {str(e)}'}), 500

        # Calcul similarités hybrides
        database_embeddings = luminaire_embeddings
        similarities = calculate_enhanced_similarity(query_embedding, database_embeddings)
        
        if len(similarities) == 0:
            return jsonify({'success': False, 'error': 'Erreur calcul similarités'}), 500

        # 🎯 SEUIL ADAPTATIF INTELLIGENT
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        max_sim = np.max(similarities)
        
        # Seuil adaptatif basé sur la distribution
        adaptive_threshold = max(
            0.15,  # Seuil minimum
            min(0.50, mean_sim + 0.5 * std_sim),  # Seuil adaptatif
            max_sim * 0.4  # Minimum 40% du meilleur score
        )
        
        logging.info(f"📊 Stats: mean={mean_sim:.3f}, std={std_sim:.3f}, max={max_sim:.3f}")
        logging.info(f"🎯 Seuil adaptatif: {adaptive_threshold:.3f}")

        # Sélection des meilleurs résultats
        top_indices = np.argsort(similarities)[::-1]
        max_results = 20
        max_items = min(len(similarities), 10000)

        results = []
        for i, idx in enumerate(top_indices[:max_results]):
            if idx >= max_items or idx >= len(luminaire_metadata):
                continue
                
            similarity_score = similarities[idx]
            
            # Filtre par seuil adaptatif
            if similarity_score < adaptive_threshold:
                continue
                
            metadata = luminaire_metadata[idx] if isinstance(luminaire_metadata[idx], dict) else {}
            
            # 🔥 SCORING OPTIMISÉ
            confidence_score = min(100, max(0, similarity_score * 120))
            
            # Qualité basée sur le score
            if confidence_score >= 80:
                quality = 'excellent'
            elif confidence_score >= 65:
                quality = 'good'
            elif confidence_score >= 45:
                quality = 'fair'
            else:
                quality = 'low'
            
            result_item = {
                'rank': i + 1,
                'similarity': round(max(0, min(100, similarity_score * 100)), 1),
                'confidence': round(confidence_score, 1),
                'quality': quality,
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
        best_confidence = results[0]['confidence'] if results else 0
        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        
        logging.info(f"✅ Recherche terminée: {len(results)} résultats, meilleur: {best_score}%")

        return jsonify({
            'success': True,
            'results': results,
            'message': f'{len(results)} résultats trouvés',
            'model_info': {
                'name': 'MobileNet V2 Optimisé',
                'version': '2.0',
                'features': 'Preprocessing + Similarité Hybride'
            },
            'improvements': {
                'preprocessing': 'Contraste + Gamma correction',
                'similarity': 'Hybrid (Cosine+Euclidean+Dot)',
                'threshold': 'Adaptatif intelligent',
                'scoring': 'Optimisé luminaires'
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
    logging.info("🚀 Démarrage serveur OPTIMISÉ v2.0...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)

