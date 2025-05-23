import os

# Configuration pour Render.com gratuit avec contraintes mémoire strictes
bind = f"0.0.0.0:{os.environ.get('PORT', 8000)}"
workers = 1  # Un seul worker pour limiter l'usage mémoire
worker_class = "sync"
worker_connections = 10  # Très limité
timeout = 240  # 4 minutes pour permettre le chargement du modèle
keepalive = 2
max_requests = 50  # Recyclage fréquent des workers
max_requests_jitter = 10
preload_app = False  # Pas de préchargement
worker_tmp_dir = "/dev/shm"  # Utiliser la RAM pour les fichiers temporaires

# Limites strictes
limit_request_line = 2048
limit_request_fields = 50
limit_request_field_size = 4096

# Configuration pour la production
loglevel = "error"
access_log_format = '%(h)s %(l)s %(t)s "%(r)s" %(s)s %(b)s'
