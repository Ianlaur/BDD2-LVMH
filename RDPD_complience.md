📋 CONSIGNES - Projet LVMH Voice-to-Tag : Phase 2 avec Contrainte RGPD
🎯 Contexte : Évolution du projet
Vous avez déjà travaillé sur la Phase 1 du projet : analyser les notes vocales des
Client Advisors (CA), extraire des tags structurés, et enrichir la taxonomie CRM
LVMH.
Une nouvelle réalité s'impose maintenant : la conformité RGPD.
Le constat
Lors d'audits internes, l'équipe juridique a identifié un risque majeur : certaines
notes CA contiennent des données personnelles sensibles que les Client
Advisors notent parfois involontairement lors de conversations avec les clients :
• Mentions de santé (problèmes médicaux, troubles psychologiques)
• Orientations sexuelles ou identités de genre
• Opinions politiques ou affiliations partisanes
• Croyances religieuses
• Situations familiales conflictuelles (divorces, litiges)
• Difficultés financières personnelles
• Commentaires sur l'apparence physique
Ces informations peuvent apparaître naturellement dans le contexte d'une
conversation (un client mentionne un burnout passé, ses convictions personnelles,
un divorce récent...) et le CA les note machinalement sans réaliser l'enjeu RGPD.
L'enjeu légal
⚖ Selon le RGPD (Articles 9 et 15) :
1. Les clients ont le droit d'accès à toutes leurs données personnelles
stockées par LVMH
2. Les données sensibles (santé, orientation, religion, politique...)
sont strictement interdites dans les bases CRM sans consentement
explicite
3. En cas de contrôle CNIL ou demande client, ces données ne doivent pas
être présentes
🚨 Risques :
• Sanctions CNIL (amendes jusqu'à 4% du CA mondial)
• Perte de confiance client
• Atteinte à la réputation
• Responsabilité légale

🛠 Votre mission : Adapter votre solution
Objectif
Vous devez adapter votre travail de la Phase 1 pour intégrer une nouvelle étape
obligatoire :
🔒 Détection et suppression automatique des données RGPD-sensibles
AVANT l'extraction des tags.
Pipeline mis à jour
📝 Note CA brute (CA_101-400)
↓
🔍 [NOUVEAU] Détection RGPD → Suppression des passages
sensibles
↓
🏷 Extraction des tags (votre travail Phase 1)
↓
📊 Enrichissement taxonomie CRM (votre travail Phase 1)
↓
✅ Tags propres et conformes RGPD stockés en base
📂 Nouveau dataset : CA_101-400
Vous recevez un fichier : lvmh_realistic_merged_ca101_400.csv
Caractéristiques
• 300 notes réelles de Client Advisors (CA_101 à CA_400)
• 5 langues : FR, EN, ES, IT, DE
• Notes prises dans des contextes professionnels variés (boutique,
événements, follow-ups)
• Format : id, date, duration, language, length,
transcription
⚠ Avertissement
Ce dataset reflète la réalité terrain : certaines notes contiennent des informations
que les CA ont notées sans se rendre compte du problème RGPD. Votre
système doit être capable de les identifier et de les traiter.