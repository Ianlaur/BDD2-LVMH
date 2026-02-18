#!/usr/bin/env python
"""Quick analysis of the LVMH data patterns."""
import pandas as pd
import re

df = pd.read_csv('data/input/LVMH_Sales_Database.csv')

print("=== CLIENT TYPES (from parentheses) ===")
client_types = set()
for t in df['Transcription']:
    matches = re.findall(r'\(([^)]+)\)', t)
    client_types.update(matches)
for ct in sorted(client_types):
    print(f"  - {ct}")

print("\n=== PRODUCT CATEGORIES ===")
# The data has very specific product categories
patterns = {
    'Limited Edition Watch': ['Limited Edition Watch', 'Montre édition limitée', 'Orologio in edizione limitata', 
                              'Reloj de edición limitada', 'Limitierte Uhr', 'Gelimiteerd horloge', 
                              'Relógio de edição limitada', '한정판 시계', '限量版手表', '限定版ウォッチ', 
                              'ساعة إصدار محدود', 'Часы ограниченной серии'],
    'Fine Jewelry': ['Fine Jewelry', 'Haute Joaillerie', 'Alta Gioielleria', 'Alta Joyería', 
                     'Feiner Schmuck', 'Fijne sieraden', 'Alta Joalharia', '파인 주얼리', 
                     '高级珠宝', '高級ジュエリー', 'مجوهرات فاخرة', 'Ювелирные изделия'],
    'Exotic Skin Handbag': ['Exotic Skin Handbag', 'Sac en peau exotique', 'Borsa in pelle esotica', 
                            'Bolso de piel exótica', 'Exotische Handtasche', 'Exotische handtas', 
                            'Bolsa de pele exótica', '엑조틱 스킨 핸드백', '珍稀皮手袋', 
                            'エキゾチックレザーバッグ', 'حقيبة جلد غريب', 'Сумка из экзотической кожи'],
    'Rare Vintage Champagne': ['Rare Vintage Champagne', 'Champagne millésimé rare', "Champagne d'annata raro", 
                               'Champaña vintage rara', 'Seltener Jahrgangschampagner', 'Zeldzame vintage champagne', 
                               'Champanhe vintage raro', '희귀 빈티지 샴페인', '稀有年份香槟', 
                               '希少なヴィンテージシャンパン', 'شامبانيا معتقة نادرة', 'Редкое винтажное шампанское'],
    'Bespoke Trunk': ['Bespoke Trunk', 'Malle sur mesure', 'Baule su misura', 'Baúl a medida', 
                      'Maßgefertigter Koffer', 'Op maat gemaakte koffer', 'Mala sob medida', 
                      '맞춤형 트렁크', '定制行李箱', '特注のトランク', 'صندوق مخصص', 'Багаж на заказ'],
    'Haute Couture': ['Haute Couture', 'Alta Moda', 'Alta Costura', '오트쿠튀르', 
                      '高级定制', 'オートクチュール', 'هوت كوتور', 'От кутюр'],
}

for cat, variants in patterns.items():
    count = 0
    for v in variants:
        count += df['Transcription'].str.contains(v, case=False, na=False, regex=False).sum()
    print(f"  {cat}: {count} notes")

print("\n=== STATUS/ACTIONS ===")
statuses = {
    'Ready to buy': ['Ready to buy', 'Prêt à acheter', "Pronto all'acquisto", 'Listo para comprar', 
                     'Kaufbereit', 'Klaar om te kopen', 'Pronto para comprar', '구매 준비 완료', 
                     '准备购买', '購入の準備ができています', 'جاهز للشراء', 'Готов к покупке'],
    'Wire transfer pending': ['Wire transfer pending', 'Virement en attente', 'Bonifico in sospeso', 
                              'Transferencia pendiente', 'Überweisung ausstehend', 'Overschrijving in behandeling',
                              'Transferência pendente', '송금 대기 중', '电汇待定', '銀行振込待ち', 
                              'الحوالة معلقة', 'Перевод в ожидании'],
    'Private viewing requested': ['private viewing', 'visite privée', 'visione privata', 'vista privada',
                                  'private Besichtigung', 'privébezichtiging', 'visualização privada',
                                  '프라이빗 뷰잉', '私下查看', 'プライベートビューイング', 'عرض خاص', 'частный просмотр'],
    'Corporate gifting': ['Corporate gifting', "Cadeau d'entreprise", 'Regalo aziendale', 'Regalo corporativo',
                          'Firmengeschenk', 'Zakelijk cadeau', 'Presente corporativo', '기업용 선물',
                          '公司送礼', '法人向けギフト', 'هدية مؤسسية', 'Корпоративный подарок'],
    'Custom monogram needed': ['custom monogram', 'monogramme personnalisé', 'monogramma personalizzato',
                               'monograma personalizado', 'individuelles Monogramm', 'aangepast monogram',
                               '맞춤형 이니셜', '定制缩写', 'カスタムモノグラム', 'مونوغرام مخصص', 'индивидуальная монограмма'],
}

for status, variants in statuses.items():
    count = 0
    for v in variants:
        count += df['Transcription'].str.contains(v, case=False, na=False, regex=False).sum()
    print(f"  {status}: {count} notes")
