# Projet : Détection de pneumonie sur radiographies thoraciques

Contexte : projet académique de Deep Learning (5IIR). Classification binaire NORMAL vs PNEUMONIA avec CNN + transfer learning (MobileNetV2, ResNet50, VGG19), TensorFlow/Keras, Grad-CAM, application Streamlit.

Règles de travail :
- Travaille UNE étape à la fois, puis arrête-toi et attends ma validation.
- Code simple, lisible, commenté en français. Pas de sur-ingénierie.
- Seed fixe = 42 partout. Constantes dans src/config.py.
- Ne jamais calculer les métriques finales ailleurs que sur le test set.
- Teste chaque script avant de dire qu'il fonctionne.
- Après chaque étape, crée docs/etape_N.md contenant :
    1. Résumé (5 lignes) de ce qui a été fait et pourquoi
    2. Résultat obtenu (chiffres, fichiers produits, figures)
    3. Notions théoriques utilisées, expliquées simplement
    4. Tableau des fonctions/classes créées ou utilisées (nom, rôle, paramètres)
    5. Le code important, découpé en morceaux, avec explication ligne par ligne
    6. 3 questions qu'un jury pourrait poser sur cette étape, avec réponses
- Puis propose le message de commit Git.
