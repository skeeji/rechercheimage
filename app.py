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

def get_real_filenames():
    """Récupère les vrais noms de fichiers depuis le dossier static/compressed"""
    try:
        compressed_dir = 'static/compressed'
        if os.path.exists(compressed_dir):
            files = []
            for f in os.listdir(compressed_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    files.append(f)
            files.sort()  # Tri important pour correspondance avec embeddings
            logging.info(f"📁 Trouvé {len(files)} fichiers dans {compressed_dir}")
            return files
        else:
            logging.warning(f"⚠️ Dossier {compressed_dir} introuvable")
            return None
    except Exception as e:
        logging.error(f"❌ Erreur lecture fichiers: {e}")
        return None

def ensure_initialized():
    global embedding_model, luminaire_embeddings, luminaire_metadata
    
    if all(x is not None for x in [embedding_model, luminaire_embeddings, luminaire_metadata]):
        return True
        
    try:
        logging.info("🔄 Initialisation avec VRAIS NOMS...")
        
        # Modèle
        if embedding_model is None:
            logging.info("📥 Chargement MobileNet...")
            embedding_model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
            test_output = embedding_model(tf.constant(np.random.rand(1, 224, 224, 3), dtype=tf.float32))
            logging.info(f"✅ Modèle OK: {test_output.shape}")
            
        # Embeddings
        if luminaire_embeddings is None:
            embeddings_path = 'models/embeddings2.npy'
            if os.path.exists(embeddings_path):
                luminaire_embeddings = np.load(embeddings_path, mmap_mode='r')
                logging.info(f"✅ Embeddings: {luminaire_embeddings.shape}")
            else:
                logging.error("❌ Fichier embeddings2.npy introuvable")
                return False
            
        # Métadonnées = noms de fichiers réels
        if luminaire_metadata is None:
            real_files = get_real_filenames()
            if real_files and len(real_files) >= len(luminaire_embeddings):
                # Correspondance 1:1 entre embeddings et fichiers
                luminaire_metadata = real_files[:len(luminaire_embeddings)]
                logging.info(f"✅ Métadonnées réelles: {len(luminaire_metadata)} fichiers")
                logging.info(f"📋 Échantillon: {luminaire_metadata[:3]}")
            else:
                # Fallback avec noms génériques basés sur les fichiers existants
                luminaire_metadata = []
                if real_files:
                    # Utiliser les vrais noms disponibles
                    for i in range(len(luminaire_embeddings)):
                        if i < len(real_files):
                            luminaire_metadata.append(real_files[i])
                        else:
                            luminaire_metadata.append(f"luminaire_{i:06d}.jpg")
                else:
                    # Pur fallback
                    luminaire_metadata = [f"luminaire_{i:06d}.jpg" for i in range(len(luminaire_embeddings))]
                
                logging.warning(f"⚠️ Métadonnées fallback: {len(luminaire_metadata)} items")
        
        cleanup_memory()
        logging.info("🎉 Initialisation complète avec vrais fichiers!")
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
            
        logging.info("🔍 Recherche avec VRAIS NOMS DE FICHIERS...")
        
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
        
        # Similarité cosine optimisée
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
        
        # Seuil adaptatif
        valid_sims = similarities[similarities > 0]
        if len(valid_sims) > 0:
            mean_sim = np.mean(valid_sims)
            std_sim = np.std(valid_sims)
            adaptive_threshold = max(0.2, mean_sim - 0.5 * std_sim)
        else:
            adaptive_threshold = 0.2
            
        logging.info(f"🎯 Seuil: {adaptive_threshold:.3f}")
        
        # Top résultats
        top_k = min(25, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        
        # Construction résultats avec VRAIS NOMS
        results = []
        for rank, idx in enumerate(top_indices):
            similarity = float(similarities[idx])
            
            if similarity >= adaptive_threshold:
                try:
                    # Récupération du vrai nom de fichier
                    if idx < len(luminaire_metadata):
                        filename = luminaire_metadata[idx]
                    else:
                        filename = f"luminaire_{idx:06d}.jpg"
                    
                    # Vérification existence fichier
                    file_path = os.path.join('static/compressed', filename)
                    if not os.path.exists(file_path):
                        # Essai avec extensions alternatives
                        base_name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                            alt_path = os.path.join('static/compressed', f"{base_name}{ext}")
                            if os.path.exists(alt_path):
                                filename = f"{base_name}{ext}"
                                break
                        else:
                            logging.warning(f"⚠️ Fichier non trouvé: {file_path}")
                            # Utiliser quand même le nom pour le debug
                    
                    # Calculs de qualité
                    confidence = min(100, max(0, int((similarity - 0.1) * 111)))
                    
                    if similarity >= 0.7:
                        quality, quality_score = 'excellent', 95
                    elif similarity >= 0.55:
                        quality, quality_score = 'very_good', 85
                    elif similarity >= 0.4:
                        quality, quality_score = 'good', 75
                    elif similarity >= 0.25:
                        quality, quality_score = 'fair', 60
                    else:
                        quality, quality_score = 'low', 40
                    
                    # URL complète
                    url = f"/static/compressed/{filename}"
                    
                    result = {
                        'filename': filename,
                        'similarity': round(similarity, 4),
                        'confidence': confidence,
                        'quality': quality,
                        'quality_score': quality_score,
                        'url': url,
                        'index': int(idx),
                        'rank': rank + 1,
                        'exists': os.path.exists(os.path.join('static/compressed', filename))
                    }
                    
                    results.append(result)
                    status = "✅" if result['exists'] else "⚠️"
                    logging.info(f"{status} #{len(results)}: {filename} ({similarity:.3f})")
                    
                except Exception as e:
                    logging.warning(f"⚠️ Erreur index {idx}: {e}")
                    continue
                    
            if len(results) >= 12:  # Limite
                break
        
        cleanup_memory()
        
        best_score = max([r['similarity'] for r in results], default=0)
        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        existing_files = sum(1 for r in results if r.get('exists', False))
        
        logging.info(f"🎉 {len(results)} résultats, {existing_files} fichiers existants, meilleur: {best_score:.3f}")
        
        return jsonify({
            'success': True,
            'results': results,
            'query_info': {
                'total_database': len(luminaire_embeddings),
                'results_found': len(results),
                'existing_files': existing_files,
                'best_similarity': round(best_score, 4),
                'avg_confidence': round(avg_confidence, 1),
                'adaptive_threshold': round(adaptive_threshold, 3),
                'sample_filename': luminaire_metadata[0] if luminaire_metadata else None
            }
        })
        
    except Exception as e:
        cleanup_memory()
        logging.error(f"❌ Erreur: {e}")
        return jsonify({'success': False, 'error': f'Erreur: {str(e)}'}), 500

@app.route('/api/files')
def list_files():
    """Debug: liste les fichiers disponibles"""
    try:
        files = get_real_filenames()
        return jsonify({
            'compressed_files': files[:10] if files else [],
            'total_files': len(files) if files else 0,
            'embeddings_count': len(luminaire_embeddings) if luminaire_embeddings is not None else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    try:
        is_ready = all(x is not None for x in [embedding_model, luminaire_embeddings, luminaire_metadata])
        files_count = len(get_real_filenames() or [])
        return jsonify({
            'status': 'ready' if is_ready else 'loading',
            'database_size': len(luminaire_embeddings) if luminaire_embeddings is not None else 0,
            'files_available': files_count,
            'sample_metadata': luminaire_metadata[:3] if luminaire_metadata else None
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("🚀 Version NOMS DE FICHIERS RÉELS...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
