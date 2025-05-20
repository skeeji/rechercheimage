# Gunicorn config file
# Augmenter le timeout pour charger le modèle
timeout = 300
workers = 1
threads = 2
bind = "0.0.0.0:$PORT"
