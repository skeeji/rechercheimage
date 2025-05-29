from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from PIL import Image
import os
import json
import pickle
import logging
import gc

# CONFIGURATION MINIMALE
logging.basicConfig(level=logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Variables globales avec lazy loading
_embedding_model = None
_embeddings_mmap = None
_metadata_cache = None

def get_model():
    """Load model only when needed"""
    global _embedding_model
    if _embedding_model is None:
        import tensorflow_hub as hub
        import tensorflow as tf
        
        tf.config.experimental.enable_memory_growth = True
        tf.config.set_soft_device_placement(True)
        
        _embedding_model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
        logging.error("✅ Model loaded")
    return _embedding_model

def get_embeddings():
    """Memory-mapped embeddings"""
    global _embeddings_mmap
    if _embeddings_mmap is None:
        _embeddings_mmap = np.load('models/embeddings2.npy', mmap_mode='r')
        logging.error(f"✅ Embeddings mapped: {_embeddings_mmap.shape}")
    return _embeddings_mmap

def get_metadata():
    """Cached metadata avec VRAIS noms de fichiers GitHub"""
    global _metadata_cache
    if _metadata_cache is None:
        try:
            # Lire les vrais noms de fichiers du dossier
            images_dir = 'data/images'
            if os.path.exists(images_dir):
                # Lister tous les fichiers .jpg dans l'ordre
                real_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.lower().endswith('.jpg')])
                logging.error(f"✅ Fichiers trouvés: {len(real_files)}")
                
                # Créer métadonnées avec vrais noms
                _metadata_cache = []
                for i, filename in enumerate(real_files):
                    _metadata_cache.append({
                        'id': f'lum_{i:04d}',
                        'name': f'Luminaire {filename}',
                        'image_path': f'data/images/{filename}',  # VRAI nom
                        'filename': filename
                    })
                
                # Compléter si nécessaire (au cas où il y a plus d'embeddings que d'images)
                while len(_metadata_cache) < 9056:
                    i = len(_metadata_cache)
                    _metadata_cache.append({
                        'id': f'lum_{i:04d}',
                        'name': f'Luminaire {i}',
                        'image_path': f'data/images/placeholder_{i}.jpg',
                        'filename': f'placeholder_{i}.jpg'
                    })
            else:
                # Fallback si dossier non trouvé
                _metadata_cache = []
                for i in range(9056):
                    _metadata_cache.append({
                        'id': f'lum_{i:04d}',
                        'name': f'Luminaire {i}',
                        'image_path': f'data/images/image_{i}.jpg',
                        'filename': f'image_{i}.jpg'
                    })
        
        except Exception as e:
            logging.error(f"❌ Erreur metadata: {e}")
            # Super fallback
            _metadata_cache = []
            for i in range(9056):
                _metadata_cache.append({
                    'id': f'lum_{i:04d}',
                    'name': f'Luminaire {i}',
                    'image_path': f'data/images/fallback_{i}.jpg',
                    'filename': f'fallback_{i}.jpg'
                })
        
        logging.error(f"✅ Metadata: {len(_metadata_cache)}")
    return _metadata_cache

def preprocess_minimal(image):
    """Preprocessing minimal pour économiser RAM"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224), Image.LANCZOS)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def cleanup():
    """Nettoyage mémoire"""
    gc.collect()
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except:
        pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        logging.error("🔍 Search starting...")
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty file'}), 400
        
        # Prétraitement
        image = Image.open(file.stream)
        processed = preprocess_minimal(image)
        
        # Modèle et embedding
        model = get_model()
        import tensorflow as tf
        query_embedding = model(tf.constant(processed)).numpy()[0]
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Base de données
        database_embeddings = get_embeddings()
        metadata = get_metadata()
        
        # Recherche par chunks pour économiser RAM
        max_search = min(2000, len(database_embeddings))
        chunk_size = 100
        
        best_scores = []
        best_indices = []
        
        for start in range(0, max_search, chunk_size):
            end = min(start + chunk_size, max_search)
            chunk = np.array(database_embeddings[start:end])
            
            # Similarité cosinus
            similarities = np.dot(chunk, query_embedding)
            
            # Garder les meilleurs de ce chunk
            for i, sim in enumerate(similarities):
                if sim > 0.3:  # Seuil minimum
                    best_scores.append(sim)
                    best_indices.append(start + i)
            
            # Cleanup ce chunk
            del chunk, similarities
            gc.collect()
        
        if best_scores:
            # Trier et prendre le top 10
            sorted_pairs = sorted(zip(best_scores, best_indices), reverse=True)
            top_pairs = sorted_pairs[:10]
            
            results = []
            for rank, (score, idx) in enumerate(top_pairs):
                meta = metadata[idx] if idx < len(metadata) else {}
                
                results.append({
                    'rank': rank + 1,
                    'similarity': round(score * 100, 1),
                    'metadata': {
                        'id': meta.get('id', f'lum_{idx}'),
                        'name': meta.get('name', f'Luminaire {idx}'),
                        'image_path': meta.get('image_path', f'data/images/image_{idx}.jpg'),
                        'filename': meta.get('filename', f'image_{idx}.jpg')
                    }
                })
            
            cleanup()
            logging.error(f"✅ Found {len(results)} results")
            
            return jsonify({
                'success': True,
                'results': results,
                'stats': {
                    'searched': max_search,
                    'found': len(results),
                    'best': results[0]['similarity'] if results else 0
                }
            })
        else:
            cleanup()
            return jsonify({'success': True, 'results': [], 'message': 'No matches'})
            
    except Exception as e:
        cleanup()
        logging.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def api_search():
    return search()

@app.route('/data/images/<filename>')
def serve_image(filename):
    """Sert les images du dossier data/images"""
    path = os.path.join('data', 'images', filename)
    if os.path.exists(path):
        return send_file(path)
    
    logging.error(f"❌ Image not found: {filename} at {path}")
    return jsonify({'error': f'Image not found: {filename}'}), 404

@app.route('/debug/images')
def debug_images():
    """Debug: liste les vraies images disponibles"""
    try:
        images_dir = 'data/images'
        if os.path.exists(images_dir):
            files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
            return jsonify({
                'directory': images_dir,
                'exists': True,
                'count': len(files),
                'samples': sorted(files)[:20],  # 20 premiers exemples
                'total_files': len(files)
            })
        else:
            return jsonify({
                'directory': images_dir,
                'exists': False,
                'error': 'Directory not found'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.error("🚀 Ultra-light server starting...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
