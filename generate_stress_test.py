import csv
import random
from datetime import datetime, timedelta
import io

# Languages to cycle through
languages = ['FR', 'EN', 'IT', 'ES', 'DE', 'PT', 'RU', 'ZH', 'JA', 'AR']

# Sample transcription templates for each language
transcription_templates = {
    'FR': [
        "Rendez-vous avec M. {name}, {age} ans, {profession}. Cherche {item} pour {occasion}. Budget {budget}€. Préfère {color} {material}. {hobby}. {allergy}. Potentiel {potential}. Rappeler {when}.",
        "Mme {name}, cliente fidèle depuis {year}. Recherche {item} saison {season}. Budget flexible {budget}K. Style {style}. Voyage souvent {destination}. Allergie {allergy}. Excellent potentiel, référera {number} amis.",
        "Couple {name1} et {name2}, {ages} ans. Cherchent cadeaux {occasion}. Budget total {budget}K. Goûts {style}. Pratiques {hobby}. Allergies {allergy}. Potentiel multi-générationnel.",
    ],
    'EN': [
        "Meeting with {name}, {age} years old, {profession}. Looking for {item} for {occasion}. Budget around {budget}K. Prefers {color} {material}. {hobby}. {allergy}. Good potential, follow up {when}.",
        "Mrs. {name}, loyal client since {year}. Seeking {item} for spring season. Flexible budget {budget}K. Style {style}. Travels to {destination} often. Allergy to {allergy}. Excellent potential, will refer {number} friends.",
        "Mr. and Mrs. {name}, {ages} years old. Looking for gifts for {occasion}. Combined budget {budget}K. Contemporary tastes {style}. Practice {hobby}. Allergies {allergy}. High lifetime value.",
    ],
    'IT': [
        "Appuntamento con {name}, {age} anni, {profession}. Cerca {item} per {occasion}. Budget {budget}K. Preferisce {color} {material}. {hobby}. {allergy}. Buon potenziale, richiamare {when}.",
        "Signora {name}, cliente fedele dal {year}. Ricerca {item} stagione {season}. Budget flessibile {budget}K. Stile {style}. Viaggia spesso {destination}. Allergia {allergy}. Ottimo potenziale, riferirà {number} amici.",
        "Coppia {name1} e {name2}, {ages} anni. Cercano regali per {occasion}. Budget totale {budget}K. Gusti {style}. Praticano {hobby}. Allergie {allergy}. Valore vita eccellente.",
    ],
    'ES': [
        "Cita con {name}, {age} años, {profession}. Busca {item} para {occasion}. Presupuesto {budget}K. Prefiere {color} {material}. {hobby}. {allergy}. Buen potencial, seguimiento {when}.",
        "Sra. {name}, cliente fiel desde {year}. Busca {item} temporada {season}. Presupuesto flexible {budget}K. Estilo {style}. Viaja a {destination} frecuentemente. Alergia {allergy}. Excelente potencial, referirá {number} amigos.",
        "Pareja {name1} y {name2}, {ages} años. Buscan regalos para {occasion}. Presupuesto combinado {budget}K. Gustos {style}. Practican {hobby}. Alergias {allergy}. Alto valor vida.",
    ],
    'DE': [
        "Termin mit {name}, {age} Jahre, {profession}. Sucht {item} für {occasion}. Budget {budget}K. Bevorzugt {color} {material}. {hobby}. {allergy}. Gutes Potenzial, Nachfassen {when}.",
        "Frau {name}, treue Kundin seit {year}. Sucht {item} Saison {season}. Flexibles Budget {budget}K. Stil {style}. Reist oft nach {destination}. Allergie {allergy}. Exzellentes Potenzial, wird {number} Freunde empfehlen.",
        "Paar {name1} und {name2}, {ages} Jahre. Suchen Geschenke für {occasion}. Kombiniertes Budget {budget}K. Moderne Geschmäcker {style}. Treiben {hobby}. Allergien {allergy}. Hoher Lebenswert.",
    ],
    'PT': [
        "Encontro com {name}, {age} anos, {profession}. Procura {item} para {occasion}. Orçamento {budget}K. Prefere {color} {material}. {hobby}. {allergy}. Bom potencial, follow up {when}.",
        "Sra. {name}, cliente fiel desde {year}. Procura {item} temporada {season}. Orçamento flexível {budget}K. Estilo {style}. Viaja para {destination} frequentemente. Alergia {allergy}. Excelente potencial, referirá {number} amigos.",
        "Casal {name1} e {name2}, {ages} anos. Procuram presentes para {occasion}. Orçamento combinado {budget}K. Gostos {style}. Praticam {hobby}. Alergias {allergy}. Alto valor vida.",
    ],
    'RU': [
        "Встреча с {name}, {age} лет, {profession}. Ищет {item} для {occasion}. Бюджет {budget}K. Предпочитает {color} {material}. {hobby}. {allergy}. Хороший потенциал, follow up {when}.",
        "Г-жа {name}, постоянный клиент с {year}. Ищет {item} сезон {season}. Гибкий бюджет {budget}K. Стиль {style}. Часто путешествует в {destination}. Аллергия {allergy}. Отличный потенциал, порекомендует {number} друзей.",
        "Пара {name1} и {name2}, {ages} лет. Ищут подарки для {occasion}. Совместный бюджет {budget}K. Современные вкусы {style}. Занимаются {hobby}. Аллергии {allergy}. Высокая ценность.",
    ],
    'ZH': [
        "与{name}会面，{age}岁，{profession}。寻找{item}用于{occasion}。预算{budget}K。偏好{color}{material}。{hobby}。{allergy}。良好潜力，跟进{when}。",
        "{name}女士，自{year}年起忠实客户。寻找{item}春季系列。灵活预算{budget}K。风格{style}。经常旅行到{destination}。对{allergy}过敏。优秀潜力，会推荐{number}朋友。",
        "{name1}和{name2}夫妇，{ages}岁。为{occasion}寻找礼物。总预算{budget}K。当代品味{style}。练习{hobby}。过敏{allergy}。高终身价值。",
    ],
    'JA': [
        "{name}さんとのミーティング、{age}歳、{profession}。{occasion}用の{item}を探しています。予算{budget}K。{color}{material}を好みます。{hobby}。{allergy}。良いポテンシャル、{when}にフォローアップ。",
        "{name}夫人、{year}年から忠実なクライアント。春の{item}を探しています。柔軟な予算{budget}K。スタイル{style}。{destination}へ頻繁に旅行。{allergy}アレルギー。優れたポテンシャル、{number}人の友人を紹介します。",
        "{name1}と{name2}夫妻、{ages}歳。{occasion}のギフトを探しています。合計予算{budget}K。現代的な味{style}。{hobby}を練習。{allergy}アレルギー。高い生涯価値。",
    ],
    'AR': [
        "لقاء مع {name}، {age} عامًا، {profession}。يبحث عن {item} لـ {occasion}। ميزانية {budget}K. يفضل {color} {material}. {hobby}. {allergy}. إمكانية جيدة، متابعة {when}.",
        "السيدة {name}، عميلة مخلصة منذ {year}. تبحث عن {item} موسم {season}. ميزانية مرنة {budget}K. أسلوب {style}. تسافر غالبًا إلى {destination}. حساسية {allergy}. إمكانية ممتازة، ستشير {number} أصدقاء.",
        "زوجان {name1} و{name2}، {ages} عامًا. يبحثان عن هدايا لـ {occasion}. ميزانية مشتركة {budget}K. أذواق {style}. يمارسون {hobby}. حساسيات {allergy}. قيمة عالية مدى الحياة.",
    ],
}

# Helper data
names = ['Dupont', 'Smith', 'Rossi', 'Garcia', 'Schmidt', 'Silva', 'Ivanov', 'Wang', 'Tanaka', 'Ahmed', 'Laurent', 'Anderson', 'Petit', 'Kim', 'Dubois', 'Thompson', 'Schneider', 'García', 'Rossi', 'Dubois']
professions = ['avocat', 'banquier', 'médecin', 'ingénieur', 'artiste', 'chef d\'entreprise', 'professeur', 'architecte', 'designer', 'photographe']
items = ['portefeuille', 'sac à main', 'valise', 'ceinture', 'montre', 'bijoux', 'accessoires', 'bagagerie']
occasions = ['anniversaire', 'Noël', 'cadeau professionnel', 'mariage', 'naissance', 'promotion']
colors = ['noir', 'marron', 'cognac', 'bleu marine', 'rouge', 'beige', 'blanc']
materials = ['cuir', 'toile', 'nappa', 'veau velours']
hobbies = ['golf', 'tennis', 'voyage', 'art', 'photographie', 'yoga', 'course', 'vin']
allergies = ['noix', 'lactose', 'nickel', 'gluten', 'fruits de mer', 'produits chimiques']
potentials = ['bon', 'excellent', 'haut', 'VIP']
whens = ['février', 'mars', 'avril', 'mai', 'juin']
seasons = ['printemps', 'été', 'automne', 'hiver']
styles = ['classique', 'moderne', 'contemporain', 'épuré', 'raffiné']
destinations = ['Paris', 'Londres', 'Milan', 'New York', 'Tokyo', 'Dubai', 'Rome', 'Berlin']
numbers = ['2', '3', '4', '5']

def generate_transcription(lang):
    template = random.choice(transcription_templates[lang])
    return template.format(
        name=random.choice(names),
        age=random.randint(25, 70),
        profession=random.choice(professions),
        item=random.choice(items),
        occasion=random.choice(occasions),
        budget=random.randint(1, 50),
        color=random.choice(colors),
        material=random.choice(materials),
        hobby=random.choice(hobbies),
        allergy=random.choice(allergies),
        potential=random.choice(potentials),
        when=random.choice(whens),
        year=random.randint(2015, 2025),
        season=random.choice(seasons),
        style=random.choice(styles),
        destination=random.choice(destinations),
        number=random.choice(numbers),
        name1=random.choice(names),
        name2=random.choice(names),
        ages=f"{random.randint(30, 50)}-{random.randint(30, 50)}"
    )

# Generate CSV content
output = io.StringIO()
writer = csv.writer(output, quoting=csv.QUOTE_ALL)

# Write header
writer.writerow(['ID', 'Date', 'Duration', 'Language', 'Length', 'Transcription'])

# Start date
current_date = datetime(2026, 1, 1)

for i in range(1, 3001):
    id_str = f"CA_{i:04d}"
    date_str = current_date.strftime('%Y-%m-%d')
    duration = random.randint(20, 60)
    duration_str = f"{duration} min"
    lang = languages[(i-1) % len(languages)]
    if duration < 30:
        length = 'short'
    elif duration <= 45:
        length = 'medium'
    else:
        length = 'long'
    transcription = generate_transcription(lang)
    
    writer.writerow([id_str, date_str, duration_str, lang, length, transcription])
    
    # Increment date by 1 day
    current_date += timedelta(days=1)

# Get the CSV content
csv_content = output.getvalue()
output.close()

# Write to file
with open('/Users/ian/Desktop/BDD2-LVMH/LVMH_Stress_Test_Generated.csv', 'w') as f:
    f.write(csv_content)