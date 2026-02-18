import csv
import random
from datetime import datetime, timedelta

# Function to generate random date within 2026
def random_date():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 12, 31)
    random_date = start + timedelta(days=random.randint(0, (end - start).days))
    return random_date.strftime('%Y-%m-%d')

# Function to generate random duration
def random_duration():
    durations = ['15 min', '20 min', '25 min', '30 min', '35 min', '40 min', '45 min', '50 min', '55 min', '60 min']
    return random.choice(durations)

# Function to determine length based on duration
def determine_length(duration):
    mins = int(duration.split()[0])
    if mins <= 20:
        return 'short'
    elif mins <= 40:
        return 'medium'
    else:
        return 'long'

# Languages
languages = ['FR', 'EN', 'IT', 'ES', 'DE']

# Expanded templates for more variety
fr_templates = [
    "Rendez-vous avec {name}, {age} ans, {profession}. Il cherche {item} pour {occasion}. Budget {budget}. Préfère {color} cuir. {details}.",
    "Mme {name}, {age} ans, {profession}, est venue pour un cadeau {occasion}. Elle aime {item} en {color}. Budget {budget}. {details}.",
    "M. {name}, {age} ans, {profession}, souhaite offrir {item} à {recipient}. Couleur préférée {color}, budget {budget}. {details}.",
    "Nouvelle cliente Mme {name}, {age} ans, {profession}. Recherche {item} pour {occasion}. Budget {budget}, cuir {color}. {details}.",
    "Client régulier M. {name}, {age} ans, {profession}. Intéressé par {item} en {color} pour {recipient}. Budget {budget}. {details}.",
]

en_templates = [
    "Meeting with {name}, {age} years old, {profession}. Looking for {item} for {occasion}. Budget around {budget}. Prefers {color} leather. {details}.",
    "Mrs. {name}, {age} years old, {profession}, came in for a {occasion} gift. She likes {item} in {color}. Budget {budget}. {details}.",
    "Mr. {name}, {age} years old, {profession}, wants to gift {item} to {recipient}. Favorite color {color}, budget {budget}. {details}.",
    "New client Mrs. {name}, {age} years old, {profession}. Seeking {item} for {occasion}. Budget {budget}, {color} leather. {details}.",
    "Regular customer Mr. {name}, {age} years old, {profession}. Interested in {item} in {color} for {recipient}. Budget {budget}. {details}.",
]

it_templates = [
    "Incontro con {name}, {age} anni, {profession}. Cerca {item} per {occasion}. Budget circa {budget}. Preferisce pelle {color}. {details}.",
    "Signora {name}, {age} anni, {profession}, è venuta per un regalo {occasion}. Ama {item} in {color}. Budget {budget}. {details}.",
    "Signor {name}, {age} anni, {profession}, vuole regalare {item} a {recipient}. Colore preferito {color}, budget {budget}. {details}.",
    "Nuova cliente Signora {name}, {age} anni, {profession}. Ricerca {item} per {occasion}. Budget {budget}, pelle {color}. {details}.",
    "Cliente abituale Signor {name}, {age} anni, {profession}. Interessato a {item} in {color} per {recipient}. Budget {budget}. {details}.",
]

es_templates = [
    "Reunión con {name}, {age} años, {profession}. Busca {item} para {occasion}. Presupuesto alrededor de {budget}. Prefiere cuero {color}. {details}.",
    "Sra. {name}, {age} años, {profession}, vino por un regalo {occasion}. Le gusta {item} en {color}. Presupuesto {budget}. {details}.",
    "Sr. {name}, {age} años, {profession}, quiere regalar {item} a {recipient}. Color favorito {color}, presupuesto {budget}. {details}.",
    "Nueva cliente Sra. {name}, {age} años, {profession}. Busca {item} para {occasion}. Presupuesto {budget}, cuero {color}. {details}.",
    "Cliente habitual Sr. {name}, {age} años, {profession}. Interesado en {item} en {color} para {recipient}. Presupuesto {budget}. {details}.",
]

de_templates = [
    "Treffen mit {name}, {age} Jahre, {profession}. Sucht {item} für {occasion}. Budget um {budget}. Bevorzugt {color} Leder. {details}.",
    "Frau {name}, {age} Jahre, {profession}, kam für ein {occasion} Geschenk. Sie mag {item} in {color}. Budget {budget}. {details}.",
    "Herr {name}, {age} Jahre, {profession}, möchte {item} an {recipient} schenken. Lieblingsfarbe {color}, Budget {budget}. {details}.",
    "Neue Kundin Frau {name}, {age} Jahre, {profession}. Sucht {item} für {occasion}. Budget {budget}, {color} Leder. {details}.",
    "Stammkunde Herr {name}, {age} Jahre, {profession}. Interessiert an {item} in {color} für {recipient}. Budget {budget}. {details}.",
]

templates = {'FR': fr_templates, 'EN': en_templates, 'IT': it_templates, 'ES': es_templates, 'DE': de_templates}

# Expanded data pools for variation
names = ['Dupont', 'Smith', 'Rossi', 'Garcia', 'Schmidt', 'Johnson', 'Bianchi', 'Martinez', 'Müller', 'Williams', 'Ferrari', 'Lopez', 'Koch', 'Brown', 'Conti', 'Rodriguez', 'Bauer', 'Davis', 'Romano', 'Hernandez', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson', 'Clark', 'Rodriguez', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 'Young', 'Hernandez', 'King', 'Wright', 'Lopez', 'Hill', 'Scott', 'Green', 'Adams', 'Baker', 'Gonzalez', 'Nelson', 'Carter']
professions = ['avocat', 'médecin', 'ingénieur', 'professeur', 'directeur', 'architecte', 'artiste', 'chef d\'entreprise', 'consultant', 'designer', 'banquier', 'journaliste', 'photographe', 'musicien', 'écrivain']
items = ['sac à main', 'portefeuille', 'montre', 'ceinture', 'valise', 'bijoux', 'accessoires', 'chaussures', 'écharpe', 'chapeau']
occasions = ['anniversaire', 'Noël', 'cadeau professionnel', 'voyage', 'mariage', 'naissance']
recipients = ['mari', 'femme', 'enfant', 'ami', 'collègue', 'parent', 'frère', 'sœur']
colors = ['noir', 'marron', 'rouge', 'bleu', 'vert', 'blanc', 'gris', 'beige', 'rose', 'violet']
budgets = ['1-2K', '2-3K', '3-5K', '5-10K', '10-20K', '20K+']
details = ['Voyage souvent en Europe.', 'Allergique au nickel.', 'Végétarien.', 'Collectionne l\'art.', 'Pratique le golf.', 'Aime la mode.', 'Travaille dans la finance.', 'A des enfants à l\'école.', 'Fan de cinéma.', 'Joue au tennis.', 'Cuisine italienne.', 'Lit beaucoup.', 'Aime la musique classique.']

# Generate CSV
with open('/Users/ian/Desktop/BDD2-LVMH/data/raw/LVMH_Stress_Test_Unique.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Date', 'Duration', 'Language', 'Length', 'Transcription'])
    
    for i in range(1, 3001):  # 3000 entries
        id_str = f'CA_{i:04d}'
        date = random_date()
        duration = random_duration()
        language = random.choice(languages)
        length = determine_length(duration)
        
        # Generate unique transcription
        template = random.choice(templates[language])
        name = random.choice(names)
        age = random.randint(25, 80)
        profession = random.choice(professions)
        item = random.choice(items)
        occasion = random.choice(occasions)
        recipient = random.choice(recipients)
        budget = random.choice(budgets)
        color = random.choice(colors)
        detail = random.choice(details)
        
        transcription = template.format(
            name=name,
            age=age,
            profession=profession,
            item=item,
            occasion=occasion,
            recipient=recipient,
            budget=budget,
            color=color,
            details=detail
        )
        
        writer.writerow([id_str, date, duration, language, length, f'"{transcription}"'])