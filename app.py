from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
import os
import json
import pickle
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Configuration mémoire TensorFlow
tf.config.set_soft_device_placement(True)
try:
    tf.config.experimental.enable_memory_growth = True
except:
    pass

app = Flask(__name__)

# Variables globales optimisées
embedding_model = None
luminaire_embeddings = None
luminaire_metadata = None

def cleanup_memory():
    """Nettoyage mémoire forcé"""
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except:
        pass

def ensure_initialized():
    global embedding_model, luminaire_embeddings, luminaire_metadata
    
    if all(x is not None for x in [embedding_model, luminaire_embeddings, luminaire_metadata]):
        return True
        
    try:
        logging.info("🔄 Initialisation ALLÉGÉE...")
        
        # Modèle plus léger
        if embedding_model is None:
            logging.info("📥 Chargement MobileNet LÉGER...")
            embedding_model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
            # Test rapide
            test_output = embedding_model(tf.constant(np.random.rand(1, 224, 224, 3), dtype=tf.float32))
            logging.info(f"✅ Modèle OK: {test_output.shape}")
            
        # Embeddings avec memory mapping
        if luminaire_embeddings is None:
            embeddings_path = 'models/embeddings2.npy'
            if os.path.exists(embeddings_path):
                # Memory mapping pour économiser RAM
                luminaire_embeddings = np.load(embeddings_path, mmap_mode='r')
                logging.info(f"✅ Embeddings mappés: {luminaire_embeddings.shape}")
            else:
                logging.error("❌ Fichier embeddings2.npy introuvable")
                return False
            
        # Métadonnées
        if luminaire_metadata is None:
            metadata_path = 'models/embeddings2.pkl'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    luminaire_metadata = pickle.load(f)
                logging.info(f"✅ Métadonnées: {len(luminaire_metadata)} items")
            else:
                logging.error("❌ Fichier embeddings2.pkl introuvable")
                return False
        
        cleanup_memory()
        logging.info("🎉 Initialisation complète!")
        return True
        
    except Exception as e:
        logging.error(f"❌ Erreur initialisation: {e}")
        cleanup_memory()
        return False

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image_optimized(image):
    """Preprocessing allégé et optimisé"""
    try:
        # Resize efficace
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Conversion array optimisée
        image_array = np.array(image, dtype=np.float32)
        
        # Normalisation [0,1]
        image_array = image_array / 255.0
        
        # Normalisation MobileNet [-1,1]
        image_array = (image_array - 0.5) * 2.0
        
        # Ajout dimension batch
        return np.expand_dims(image_array, axis=0)
        
    except Exception as e:
        logging.error(f"❌ Erreur préprocessing: {e}")
        raise

@app.route('/search', methods=['POST'])
def search_similar():
    try:
        # Vérification initialisation
        if not ensure_initialized():
            return jsonify({'success': False, 'error': 'Modèle non initialisé'}), 503
            
        # Vérification fichier
        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'Aucune image fournie'}), 400
            
        logging.info("🔍 Nouvelle recherche OPTIMISÉE...")
        
        # Chargement et conversion image
        image = Image.open(file.stream)
        logging.info(f"📸 Image chargée: {image.size}, mode: {image.mode}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logging.info("🔄 Conversion en RGB")
            
        # Preprocessing optimisé
        image_batch = preprocess_image_optimized(image)
        logging.info(f"✅ Image préprocessée: {image_batch.shape}")
        
        # Extraction des features
        query_features = embedding_model(tf.constant(image_batch, dtype=tf.float32))
        query_embedding = query_features.numpy().flatten()
        logging.info(f"✅ Features extraites: {query_embedding.shape}")
        
        # Calcul de similarité cosine optimisé
        # Normalisation des vecteurs
        db_norms = np.linalg.norm(luminaire_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        # Filtrage des vecteurs nuls
        valid_indices = (db_norms > 1e-8) & (query_norm > 1e-8)
        
        if query_norm < 1e-8:
            return jsonify({'success': False, 'error': 'Features de requête invalides'}), 400
            
        # Calcul similarité cosine pour indices valides seulement
        similarities = np.zeros(len(luminaire_embeddings))
        if np.any(valid_indices):
            valid_embeddings = luminaire_embeddings[valid_indices]
            valid_norms = db_norms[valid_indices]
            
            # Produit scalaire normalisé
            dot_products = np.dot(valid_embeddings, query_embedding)
            similarities[valid_indices] = dot_products / (valid_norms * query_norm)
        
        # Seuil adaptatif intelligent
        valid_similarities = similarities[similarities > 0]
        if len(valid_similarities) > 0:
            mean_sim = np.mean(valid_similarities)
            std_sim = np.std(valid_similarities)
            adaptive_threshold = max(0.25, mean_sim + 0.3 * std_sim)
        else:
            adaptive_threshold = 0.25
            
        logging.info(f"🎯 Seuil adaptatif: {adaptive_threshold:.3f}")
        
        # Sélection des meilleurs résultats
        top_k = min(25, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        
        # Construction des résultats
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            
            if similarity >= adaptive_threshold:
                try:
                    metadata = luminaire_metadata[idx]
                    
                    # Calcul confidence amélioré
                    confidence = min(100, max(0, int((similarity - 0.2) * 125)))
                    
                    # Qualité basée sur similarity
                    if similarity >= 0.8:
                        quality = 'excellent'
                        quality_score = 95
                    elif similarity >= 0.65:
                        quality = 'very_good'
                        quality_score = 85
                    elif similarity >= 0.5:
                        quality = 'good'
                        quality_score = 75
                    elif similarity >= 0.35:
                        quality = 'fair'
                        quality_score = 60
                    else:
                        quality = 'low'
                        quality_score = 40
                    
                    result = {
                        'filename': metadata.get('filename', f'image_{idx}.jpg'),
                        'similarity': round(similarity, 4),
                        'confidence': confidence,
                        'quality': quality,
                        'quality_score': quality_score,
                        'url': f"/static/compressed/{metadata.get('filename', f'img_{idx}.jpg')}",
                        'index': int(idx)
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logging.warning(f"⚠️ Erreur métadonnées index {idx}: {e}")
                    continue
                    
            # Limite pour performance
            if len(results) >= 15:
                break
        
        # Tri final par similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Nettoyage mémoire
        cleanup_memory()
        
        # Statistiques
        best_score = max([r['similarity'] for r in results], default=0)
        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        
        logging.info(f"✅ Trouvé {len(results)} résultats, meilleur: {best_score:.3f}")
        
        return jsonify({
            'success': True,
            'results': results,
            'query_info': {
                'total_database': len(luminaire_embeddings),
                'valid_database': int(np.sum(valid_indices)),
                'results_found': len(results),
                'best_similarity': round(best_score, 4),
                'avg_confidence': round(avg_confidence, 1),
                'adaptive_threshold': round(adaptive_threshold, 3),
                'query_norm': round(float(query_norm), 3),
                'feature_dims': int(query_embedding.shape[0])
            }
        })
        
    except Exception as e:
        cleanup_memory()
        logging.error(f"❌ Erreur recherche: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'Erreur système: {str(e)}'}), 500

@app.route('/api/search', methods=['POST'])
def api_search_similar():
    """Endpoint API pour recherche"""
    return search_similar()

@app.route('/health')
def health_check():
    """Endpoint de santé"""
    try:
        is_ready = all(x is not None for x in [embedding_model, luminaire_embeddings, luminaire_metadata])
        return jsonify({
            'status': 'healthy' if is_ready else 'initializing',
            'model_loaded': embedding_model is not None,
            'embeddings_loaded': luminaire_embeddings is not None,
            'metadata_loaded': luminaire_metadata is not None,
            'database_size': len(luminaire_embeddings) if luminaire_embeddings is not None else 0
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint non trouvé'}), 404

@app.errorhandler(500)
def internal_error(error):
    cleanup_memory()
    return jsonify({'success': False, 'error': 'Erreur serveur interne'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'success': False, 'error': 'Image trop volumineuse (max 16MB)'}), 413

if __name__ == '__main__':
    logging.info("🚀 Démarrage serveur OPTIMISÉ v2.1...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
