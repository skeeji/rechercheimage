import os

# Configuration pour Render.com gratuit
bind = f"0.0.0.0:{os.environ.get('PORT', 8000)}"
workers = 1
worker_class = "sync"
worker_connections = 10
timeout = 300  # 5 minutes pour permettre le chargement complet
keepalive = 2
max_requests = 30
max_requests_jitter = 5
preload_app = False
worker_tmp_dir = "/dev/shm"

# Limites strictes
limit_request_line = 2048
limit_request_fields = 50
limit_request_field_size = 4096

# Configuration des logs
loglevel = "info"
access_log_format = '%(h)s %(l)s %(t)s "%(r)s" %(s)s %(b)s'

# Hook pour initialiser l'app après le fork du worker
def when_ready(server):
    server.log.info("Serveur prêt!")

def worker_int(worker):
    worker.log.info("Worker interrompu")
