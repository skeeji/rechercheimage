from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import json
import pickle
import logging
import gc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# CONFIGURATION MINIMALE
logging.basicConfig(level=logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Variables globales avec lazy loading
_embedding_model = None
_embeddings_mmap = None
_embeddings_normalized = None
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
    """Memory-mapped embeddings avec normalisation"""
    global _embeddings_mmap, _embeddings_normalized
    if _embeddings_mmap is None:
        _embeddings_mmap = np.load('models/embeddings2.npy', mmap_mode='r')
        # Pré-normaliser pour améliorer la précision
        _embeddings_normalized = normalize(_embeddings_mmap.copy(), norm='l2')
        logging.error(f"✅ Embeddings normalisés: {_embeddings_normalized.shape}")
    return _embeddings_normalized

def get_metadata():
    """Cached metadata avec VRAIS noms de fichiers GitHub"""
    global _metadata_cache
    if _metadata_cache is None:
        try:
            images_dir = 'data/images'
            if os.path.exists(images_dir):
                real_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.lower().endswith('.jpg')])
                logging.error(f"✅ Fichiers trouvés: {len(real_files)}")
                
                _metadata_cache = []
                for i, filename in enumerate(real_files):
                    _metadata_cache.append({
                        'id': f'lum_{i:04d}',
                        'name': f'Luminaire {filename.replace(".jpg", "")}',
                        'image_path': f'data/images/{filename}',
                        'filename': filename
                    })
                
                while len(_metadata_cache) < 9056:
                    i = len(_metadata_cache)
                    _metadata_cache.append({
                        'id': f'lum_{i:04d}',
                        'name': f'Luminaire {i}',
                        'image_path': f'data/images/placeholder_{i}.jpg',
                        'filename': f'placeholder_{i}.jpg'
                    })
            else:
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

def preprocess_advanced(image):
    """Preprocessing ULTRA-AVANCÉ pour maximum précision"""
    # Conversion RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 1. AMÉLIORATION CONTRASTE/LUMINOSITÉ
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)  # +20% contraste
    
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)  # +10% luminosité
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.15)  # +15% netteté
    
    # 2. FILTRE ANTI-BRUIT
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    # 3. REDIMENSIONNEMENT HAUTE QUALITÉ avec padding intelligent
    original_size = image.size
    target_size = 224
    
    # Calculer le ratio optimal
    ratio = target_size / max(original_size)
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    
    # Resize avec LANCZOS (meilleure qualité)
    image = image.resize(new_size, Image.LANCZOS)
    
    # Padding centré avec couleur dominante
    new_image = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    paste_x = (target_size - new_size[0]) // 2
    paste_y = (target_size - new_size[1]) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    # 4. NORMALISATION AVANCÉE
    arr = np.array(new_image, dtype=np.float32) / 255.0
    
    # Normalisation par canal (ImageNet style)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    
    # 5. AUGMENTATION DE DONNÉES (moyenne de plusieurs versions)
    variants = []
    
    # Version originale
    variants.append(np.expand_dims(arr, axis=0))
    
    # Version légèrement rotée
    rotated = new_image.rotate(2, expand=False)
    arr_rot = np.array(rotated, dtype=np.float32) / 255.0
    arr_rot = (arr_rot - mean) / std
    variants.append(np.expand_dims(arr_rot, axis=0))
    
    # Version avec contraste différent
    contrast_img = ImageEnhance.Contrast(new_image).enhance(0.9)
    arr_cont = np.array(contrast_img, dtype=np.float32) / 255.0
    arr_cont = (arr_cont - mean) / std
    variants.append(np.expand_dims(arr_cont, axis=0))
    
    return variants

def extract_embedding_ensemble(image_variants, model):
    """Extraction d'embeddings avec technique d'ensemble"""
    import tensorflow as tf
    
    embeddings = []
    
    for variant in image_variants:
        embedding = model(tf.constant(variant)).numpy()[0]
        # Normalisation L2
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        embeddings.append(embedding)
    
    # Moyenne pondérée des embeddings
    weights = [0.6, 0.25, 0.15]  # Plus de poids sur l'original
    final_embedding = np.average(embeddings, axis=0, weights=weights)
    
    # Re-normalisation finale
    norm = np.linalg.norm(final_embedding)
    if norm > 0:
        final_embedding = final_embedding / norm
    
    return final_embedding

def advanced_similarity_search(query_embedding, database_embeddings, top_k=50):
    """Recherche de similarité ultra-avancée"""
    
    # 1. SIMILARITÉ COSINUS HAUTE PRÉCISION
    cosine_scores = np.dot(database_embeddings, query_embedding)
    
    # 2. DISTANCE EUCLIDIENNE (pour affiner)
    euclidean_distances = np.linalg.norm(database_embeddings - query_embedding, axis=1)
    euclidean_scores = 1 / (1 + euclidean_distances)  # Convertir en score
    
    # 3. SIMILARITÉ HYBRIDE PONDÉRÉE
    hybrid_scores = 0.7 * cosine_scores + 0.3 * euclidean_scores
    
    # 4. FILTRAGE ADAPTATIF
    # Calculer quartiles pour seuil dynamique
    q75 = np.percentile(hybrid_scores, 75)
    q90 = np.percentile(hybrid_scores, 90)
    
    # Seuil adaptatif basé sur la distribution
    adaptive_threshold = max(0.4, q75 * 0.8)
    
    # 5. SÉLECTION DES MEILLEURS CANDIDATS
    valid_indices = np.where(hybrid_scores > adaptive_threshold)[0]
    
    if len(valid_indices) == 0:
        # Fallback : prendre les 10 meilleurs quand même
        valid_indices = np.argsort(hybrid_scores)[-10:]
    
    # Trier par score
    sorted_indices = valid_indices[np.argsort(hybrid_scores[valid_indices])[::-1]]
    
    # Prendre le top K
    final_indices = sorted_indices[:top_k]
    final_scores = hybrid_scores[final_indices]
    
    return list(zip(final_scores, final_indices))

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
        logging.error("🔍 Recherche HAUTE PRÉCISION...")
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty file'}), 400
        
        # 1. PRÉTRAITEMENT AVANCÉ
        image = Image.open(file.stream)
        image_variants = preprocess_advanced(image)
        
        # 2. EXTRACTION EMBEDDING ENSEMBLE
        model = get_model()
        query_embedding = extract_embedding_ensemble(image_variants, model)
        logging.error(f"✅ Embedding extrait avec {len(image_variants)} variants")
        
        # 3. BASE DE DONNÉES NORMALISÉE
        database_embeddings = get_embeddings()
        metadata = get_metadata()
        
        # 4. RECHERCHE ULTRA-PRÉCISE
        results_pairs = advanced_similarity_search(query_embedding, database_embeddings, top_k=15)
        logging.error(f"✅ Trouvé {len(results_pairs)} candidats")
        
        # 5. POST-TRAITEMENT ET RANKING
        final_results = []
        for rank, (score, idx) in enumerate(results_pairs[:10]):
            meta = metadata[idx] if idx < len(metadata) else {}
            
            # Score de confiance ajusté
            confidence = min(100, score * 120)  # Boost léger pour affichage
            
            # Classification qualitative
            if confidence > 85:
                quality = "Excellent"
            elif confidence > 70:
                quality = "Très bon"
            elif confidence > 55:
                quality = "Bon"
            elif confidence > 40:
                quality = "Moyen"
            else:
                quality = "Faible"
            
            final_results.append({
                'rank': rank + 1,
                'similarity': round(confidence, 1),
                'quality': quality,
                'metadata': {
                    'id': meta.get('id', f'lum_{idx}'),
                    'name': meta.get('name', f'Luminaire {idx}'),
                    'image_path': meta.get('image_path', f'data/images/image_{idx}.jpg'),
                    'filename': meta.get('filename', f'image_{idx}.jpg')
                }
            })
        
        cleanup()
        logging.error(f"✅ Résultats finaux: {len(final_results)}")
        
        return jsonify({
            'success': True,
            'results': final_results,
            'stats': {
                'searched': len(database_embeddings),
                'candidates': len(results_pairs),
                'final': len(final_results),
                'best_score': final_results[0]['similarity'] if final_results else 0,
                'precision_mode': 'ULTRA_ADVANCED'
            }
        })
            
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
                'samples': sorted(files)[:20],
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
    logging.error("🚀 Serveur HAUTE PRÉCISION démarré...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
