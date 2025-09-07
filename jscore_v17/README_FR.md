# J-Score — Version corrigée (upload image + commandes d’ouverture)

Cette archive contient votre projet **corrigé** (champ fichier sans double `id`), prêt à être lancé.

## 1) Prérequis
- macOS avec **Python 3.8+** (vous avez 3.9, c’est bon)
- Accès Internet pour installer les dépendances

---

## 2) Lancer sur macOS (Terminal)
Ouvrez **Terminal**, placez-vous dans le dossier extrait, puis copiez-collez :

```bash
# 1) créer l'environnement virtuel
python3 -m venv .venv

# 2) l'activer
source .venv/bin/activate

# 3) mettre pip à jour et installer les dépendances
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) (optionnel) variables d'admin si vous utilisez la page /admin
export ADMIN_USER=admin
export ADMIN_PASS=admin
export SECRET_KEY=change-me-in-prod

# 5) lancer l'application
python app.py
```

Puis ouvrez votre navigateur sur : **http://localhost:5000**  
(Admin si besoin : **http://localhost:5000/admin** avec les identifiants ci-dessus.)

Pour **arrêter** : retournez sur le Terminal et faites `Ctrl + C`.  
Pour **réactiver** plus tard : `source .venv/bin/activate` puis `python app.py`.

---

## 3) Lancer sur Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

$env:ADMIN_USER="admin"
$env:ADMIN_PASS="admin"
$env:SECRET_KEY="change-me-in-prod"

python app.py
```

---

## 4) Correction appliquée (upload d’image)
Dans `templates/form.html` :
- Problème initial : `<input ... id="eye_file" ... id="eye_photo" ...>` (deux `id` sur le même input).
- Correction : `id="eye_photo" name="eye_photo"` et **remplacement de toutes les références JS** `eye_file` → `eye_photo`.

Si vous comparez, vous ne devriez plus trouver `eye_file` dans les fichiers `.html/.js`.

---

## 5) Dépannage rapide
- **`flask: command not found`** : lancez avec `python app.py` (c’est déjà prévu).
- **`ModuleNotFoundError`** : vérifiez que l’environnement est bien activé (`source .venv/bin/activate`) et refaites `pip install -r requirements.txt`.
- **Chemins avec espaces/parenthèses** : gardez les guillemets quand vous tapez un chemin complet.

Bon lancement !
