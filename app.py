from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from PIL import Image
import os
import json
import pickle
import logging
import gc

# CONFIGURATION MINIMALE
logging.basicConfig(level=logging.ERROR)  # Moins de logs
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
        
        # Config mémoire TF ultra-strict
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
    """Cached metadata"""
    global _metadata_cache
    if _metadata_cache is None:
        try:
            with open('models/embeddings2.pkl', 'rb') as f:
                raw = pickle.load(f)
            if isinstance(raw, np.ndarray):
                _metadata_cache = [{'id': f'lum_{i:04d}', 'image_path': f'data/images/luminaire_{i:06d}.jpg'} 
                                 for i in range(len(raw))]
            else:
                _metadata_cache = raw
        except:
            _metadata_cache = [{'id': f'lum_{i:04d}'} for i in range(9056)]
        logging.error(f"✅ Metadata: {len(_metadata_cache)}")
    return _metadata_cache

def preprocess_minimal(image):
    """Preprocessing minimal pour économiser RAM"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize simple
    image = image.resize((224, 224), Image.LANCZOS)
    
    # Array minimal
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def cleanup():
    """Nettoyage agressif"""
    gc.collect()
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except:
        pass

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except:
        return jsonify({'status': 'API active', 'endpoint': '/search'})

@app.route('/status')
def status():
    try:
        embeddings = get_embeddings()
        metadata = get_metadata()
        return jsonify({
            'status': 'ready',
            'embeddings': embeddings.shape[0],
            'metadata': len(metadata),
            'memory_mode': 'ultra_light'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image'}), 400
    
    try:
        logging.error("🔍 Search starting...")
        
        # 1. Load image
        image = Image.open(request.files['image'].stream)
        image_batch = preprocess_minimal(image)
        
        # 2. Get model and extract features
        model = get_model()
        import tensorflow as tf
        features = model(tf.constant(image_batch)).numpy()[0]
        
        # Normalize query
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        # 3. Get embeddings (memory-mapped)
        embeddings = get_embeddings()
        
        # 4. Search in SMALL batches to save memory
        batch_size = 100
        max_search = min(2000, len(embeddings))  # Limite drastique
        
        best_scores = []
        best_indices = []
        
        for i in range(0, max_search, batch_size):
            end_idx = min(i + batch_size, max_search)
            
            # Load only this batch
            batch = embeddings[i:end_idx].copy()
            
            # Normalize batch
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            batch = np.where(norms > 0, batch / norms, batch)
            
            # Cosine similarity
            scores = np.dot(batch, features)
            
            # Keep only top from this batch
            top_k = min(5, len(scores))
            top_batch_idx = np.argpartition(scores, -top_k)[-top_k:]
            
            for idx in top_batch_idx:
                if scores[idx] > 0.3:  # Seuil strict
                    best_scores.append(scores[idx])
                    best_indices.append(i + idx)
            
            # Cleanup this batch
            del batch
            if i % 500 == 0:
                gc.collect()
        
        # 5. Final top results
        if best_scores:
            final_indices = np.argsort(best_scores)[::-1][:10]
            metadata = get_metadata()
            
            results = []
            for rank, idx in enumerate(final_indices):
                real_idx = best_indices[idx]
                score = best_scores[idx]
                
                meta = metadata[real_idx] if real_idx < len(metadata) else {}
                
                results.append({
                    'rank': rank + 1,
                    'similarity': round(score * 100, 1),
                    'metadata': {
                        'id': meta.get('id', f'lum_{real_idx}'),
                        'name': meta.get('name', f'Luminaire {real_idx}'),
                        'image_path': meta.get('image_path', f'data/images/luminaire_{real_idx:06d}.jpg')
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
    path = os.path.join('data', 'images', filename)
    if os.path.exists(path):
        return send_file(path)
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    logging.error("🚀 Ultra-light server starting...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
