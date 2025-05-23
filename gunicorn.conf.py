import os

# Configuration optimisée pour Render.com gratuit
bind = f"0.0.0.0:{os.environ.get('PORT', 8000)}"
workers = 1  # Un seul worker pour économiser la mémoire
worker_class = "sync"
worker_connections = 50  # Réduit
timeout = 180  # Réduit mais suffisant
keepalive = 2
max_requests = 50  # Réduit pour forcer le recyclage
max_requests_jitter = 10
preload_app = False  # Éviter le préchargement
worker_tmp_dir = "/dev/shm"  # Utiliser la RAM pour les fichiers temporaires

# Limites mémoire
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
