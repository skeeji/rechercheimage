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
        logging.info("🔄 Initialisation CORRIGÉE...")
        
        # Modèle plus léger
        if embedding_model is None:
            logging.info("📥 Chargement MobileNet...")
            embedding_model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
            test_output = embedding_model(tf.constant(np.random.rand(1, 224, 224, 3), dtype=tf.float32))
            logging.info(f"✅ Modèle OK: {test_output.shape}")
            
        # Embeddings avec memory mapping
        if luminaire_embeddings is None:
            embeddings_path = 'models/embeddings2.npy'
            if os.path.exists(embeddings_path):
                luminaire_embeddings = np.load(embeddings_path, mmap_mode='r')
                logging.info(f"✅ Embeddings: {luminaire_embeddings.shape}")
            else:
                logging.error("❌ Fichier embeddings2.npy introuvable")
                return False
            
        # Métadonnées avec diagnostic
        if luminaire_metadata is None:
            metadata_path = 'models/embeddings2.pkl'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    luminaire_metadata = pickle.load(f)
                
                # DIAGNOSTIC du format des métadonnées
                logging.info(f"📊 Type métadonnées: {type(luminaire_metadata)}")
                if hasattr(luminaire_metadata, 'shape'):
                    logging.info(f"📊 Shape métadonnées: {luminaire_metadata.shape}")
                if isinstance(luminaire_metadata, (list, np.ndarray)) and len(luminaire_metadata) > 0:
                    sample = luminaire_metadata[0] if len(luminaire_metadata) > 0 else None
                    logging.info(f"📊 Échantillon métadonnée[0]: {type(sample)} = {sample}")
                
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
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        image_array = (image_array - 0.5) * 2.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        logging.error(f"❌ Erreur préprocessing: {e}")
        raise

@app.route('/search', methods=['POST'])
def search_similar():
    try:
        if not ensure_initialized():
            return jsonify({'success': False, 'error': 'Modèle non initialisé'}), 503
            
        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'Aucune image fournie'}), 400
            
        logging.info("🔍 Recherche avec métadonnées CORRIGÉES...")
        
        # Chargement image
        image = Image.open(file.stream)
        logging.info(f"📸 Image: {image.size}, mode: {image.mode}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Preprocessing
        image_batch = preprocess_image_optimized(image)
        logging.info(f"✅ Préprocessée: {image_batch.shape}")
        
        # Extraction features
        query_features = embedding_model(tf.constant(image_batch, dtype=tf.float32))
        query_embedding = query_features.numpy().flatten()
        logging.info(f"✅ Features: {query_embedding.shape}")
        
        # Similarité (version allégée)
        db_norms = np.linalg.norm(luminaire_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        valid_indices = (db_norms > 1e-8) & (query_norm > 1e-8)
        
        if query_norm < 1e-8:
            return jsonify({'success': False, 'error': 'Features invalides'}), 400
            
        similarities = np.zeros(len(luminaire_embeddings))
        if np.any(valid_indices):
            valid_embeddings = luminaire_embeddings[valid_indices]
            valid_norms = db_norms[valid_indices]
            dot_products = np.dot(valid_embeddings, query_embedding)
            similarities[valid_indices] = dot_products / (valid_norms * query_norm)
        
        # Seuil adaptatif plus permissif
        valid_sims = similarities[similarities > 0]
        if len(valid_sims) > 0:
            mean_sim = np.mean(valid_sims)
            # Seuil plus bas pour avoir des résultats
            adaptive_threshold = max(0.15, mean_sim - 0.1)
        else:
            adaptive_threshold = 0.15
            
        logging.info(f"🎯 Seuil: {adaptive_threshold:.3f}")
        
        # Top résultats
        top_k = min(30, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        
        # Construction résultats CORRIGÉE
        results = []
        for rank, idx in enumerate(top_indices):
            similarity = float(similarities[idx])
            
            if similarity >= adaptive_threshold:
                try:
                    # Génération nom de fichier générique
                    filename = f"luminaire_{idx:06d}.jpg"
                    
                    # Tentative d'extraction du vrai nom si possible
                    try:
                        if isinstance(luminaire_metadata, (list, np.ndarray)) and idx < len(luminaire_metadata):
                            meta_item = luminaire_metadata[idx]
                            if isinstance(meta_item, dict) and 'filename' in meta_item:
                                filename = meta_item['filename']
                            elif isinstance(meta_item, str):
                                filename = meta_item if meta_item.endswith(('.jpg', '.jpeg', '.png')) else f"{meta_item}.jpg"
                    except Exception as meta_error:
                        logging.debug(f"Métadonnée non accessible pour {idx}: {meta_error}")
                    
                    # Calculs
                    confidence = min(100, max(0, int((similarity - 0.1) * 111)))
                    
                    if similarity >= 0.7:
                        quality, quality_score = 'excellent', 95
                    elif similarity >= 0.5:
                        quality, quality_score = 'good', 75
                    elif similarity >= 0.3:
                        quality, quality_score = 'fair', 60
                    else:
                        quality, quality_score = 'low', 40
                    
                    # Construction URL
                    base_name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                    
                    result = {
                        'filename': filename,
                        'similarity': round(similarity, 4),
                        'confidence': confidence,
                        'quality': quality,
                        'quality_score': quality_score,
                        'url': f"/static/compressed/{base_name}.jpg",
                        'index': int(idx),
                        'rank': rank + 1
                    }
                    
                    results.append(result)
                    logging.info(f"✅ #{len(results)}: {filename} ({similarity:.3f})")
                    
                except Exception as e:
                    logging.warning(f"⚠️ Erreur index {idx}: {e}")
                    continue
                    
            if len(results) >= 12:  # Limite
                break
        
        cleanup_memory()
        
        best_score = max([r['similarity'] for r in results], default=0)
        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        
        logging.info(f"🎉 {len(results)} résultats, meilleur: {best_score:.3f}")
        
        return jsonify({
            'success': True,
            'results': results,
            'query_info': {
                'total_database': len(luminaire_embeddings),
                'results_found': len(results),
                'best_similarity': round(best_score, 4),
                'avg_confidence': round(avg_confidence, 1),
                'adaptive_threshold': round(adaptive_threshold, 3),
                'metadata_type': str(type(luminaire_metadata).__name__)
            }
        })
        
    except Exception as e:
        cleanup_memory()
        logging.error(f"❌ Erreur: {e}")
        return jsonify({'success': False, 'error': f'Erreur: {str(e)}'}), 500

@app.route('/health')
def health_check():
    try:
        is_ready = all(x is not None for x in [embedding_model, luminaire_embeddings, luminaire_metadata])
        return jsonify({
            'status': 'ready' if is_ready else 'loading',
            'database_size': len(luminaire_embeddings) if luminaire_embeddings is not None else 0
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("🚀 Version MÉTADONNÉES CORRIGÉES...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
