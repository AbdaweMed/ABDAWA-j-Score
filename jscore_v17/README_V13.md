# ABDAWA J-Score v13.0

Nouveautés:
- Consentement **versionné** (ref 012-2025, version v1.0-2025-07-07) + impression dans le PDF.
- **Rate limiting** (10/min, 30/h par IP) pour éviter les abus.
- **Admin delete** : suppression d’une soumission avec suppression de la photo.
- **Traçabilité** : IP + User-Agent stockés.
- **Dark mode** + capture mobile de la caméra (input capture="environment").
- Page **/about** (infos éthiques).

## Démarrage rapide (macOS)
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export ADMIN_USER=admin ADMIN_PASS=admin SECRET_KEY=change-me-in-prod
python app.py
open http://localhost:5000
