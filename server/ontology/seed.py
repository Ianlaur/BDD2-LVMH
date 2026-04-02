"""Seed the 14-dimension hierarchical ontology with initial concepts.

Generates taxonomy/ontology_v2.json with 200+ concepts across 14 dimensions,
each with multilingual aliases (at minimum FR + EN).

Usage:
    python -m server.ontology.seed
"""
from __future__ import annotations
import json
from pathlib import Path

from server.ontology.schema import (
    Concept, Category, Dimension, Ontology,
    Relationship, RelationshipType,
)
from server.ontology.loader import save_ontology
from server.shared.config import TAXONOMY_DIR


def _r(rtype: str, target: str, weight: float = 1.0) -> dict:
    """Shorthand for relationship dict."""
    return {"type": rtype, "target": target, "weight": weight}


def _build_ontology() -> Ontology:
    """Build the full 14-dimension ontology."""
    dimensions = []

    # ================================================================
    # 1. PRODUCT AFFINITY (30+ concepts)
    # ================================================================
    dim_product = Dimension(id="product_affinity", label="Product Affinity", categories=[
        Category(id="product_affinity.leather_goods", label="Leather Goods", concepts=[
            Concept(id="product_affinity.leather_goods.handbag", label="Handbag Interest",
                    aliases={"fr": ["sac à main", "sac", "sacoche"], "en": ["handbag", "bag", "purse"],
                             "it": ["borsa"], "es": ["bolso"], "de": ["handtasche"]}, weight=0.5),
            Concept(id="product_affinity.leather_goods.travel_bag", label="Travel Bag",
                    aliases={"fr": ["sac de voyage", "bagage"], "en": ["travel bag", "luggage", "carry-on"],
                             "it": ["borsa da viaggio"], "es": ["bolsa de viaje"], "de": ["reisetasche"]}, weight=0.7),
            Concept(id="product_affinity.leather_goods.small_leather_goods", label="Small Leather Goods",
                    aliases={"fr": ["petit maroquinerie", "portefeuille", "porte-cartes"],
                             "en": ["wallet", "card holder", "small leather goods", "key pouch"],
                             "it": ["portafoglio", "porta carte"], "es": ["cartera", "billetera"]}, weight=0.6),
            Concept(id="product_affinity.leather_goods.briefcase", label="Briefcase",
                    aliases={"fr": ["attaché-case", "serviette", "porte-documents"],
                             "en": ["briefcase", "business bag", "document holder"],
                             "it": ["valigetta", "cartella"], "es": ["maletín"]}, weight=0.7),
            Concept(id="product_affinity.leather_goods.clutch", label="Clutch / Evening Bag",
                    aliases={"fr": ["pochette", "sac de soirée", "minaudière"],
                             "en": ["clutch", "evening bag", "minaudière"],
                             "it": ["pochette", "borsa da sera"]}, weight=0.7),
            Concept(id="product_affinity.leather_goods.backpack", label="Backpack",
                    aliases={"fr": ["sac à dos"], "en": ["backpack", "rucksack"],
                             "it": ["zaino"], "es": ["mochila"], "de": ["rucksack"]}, weight=0.6),
        ]),
        Category(id="product_affinity.watches", label="Watches", concepts=[
            Concept(id="product_affinity.watches.watch_interest", label="Watch Interest",
                    aliases={"fr": ["montre", "horlogerie"], "en": ["watch", "timepiece", "horology"],
                             "it": ["orologio"], "es": ["reloj"], "de": ["uhr"]}, weight=0.6),
            Concept(id="product_affinity.watches.tourbillon_interest", label="Tourbillon Interest",
                    aliases={"fr": ["tourbillon", "mouvement tourbillon"], "en": ["tourbillon"],
                             "it": ["tourbillon"]}, weight=0.9,
                    relationships=[Relationship(type=RelationshipType.IMPLIES,
                                                target_id="lifestyle_signals.hobbies.watch_collector", weight=0.7)]),
            Concept(id="product_affinity.watches.chronograph", label="Chronograph",
                    aliases={"fr": ["chronographe"], "en": ["chronograph"],
                             "it": ["cronografo"], "es": ["cronógrafo"]}, weight=0.8),
            Concept(id="product_affinity.watches.vintage_watch", label="Vintage Watch",
                    aliases={"fr": ["montre vintage", "montre ancienne"], "en": ["vintage watch", "vintage timepiece"],
                             "it": ["orologio vintage"]}, weight=0.8),
        ]),
        Category(id="product_affinity.jewelry", label="Jewelry", concepts=[
            Concept(id="product_affinity.jewelry.fine_jewelry", label="Fine Jewelry",
                    aliases={"fr": ["joaillerie", "bijoux", "haute joaillerie"],
                             "en": ["fine jewelry", "jewelry", "jewellery", "high jewelry"],
                             "it": ["gioielleria", "gioielli"], "es": ["joyería", "joyas"]}, weight=0.7),
            Concept(id="product_affinity.jewelry.ring", label="Ring",
                    aliases={"fr": ["bague", "alliance", "anneau"], "en": ["ring", "band"],
                             "it": ["anello"], "es": ["anillo"]}, weight=0.6),
            Concept(id="product_affinity.jewelry.necklace", label="Necklace",
                    aliases={"fr": ["collier", "pendentif", "sautoir"], "en": ["necklace", "pendant", "chain"],
                             "it": ["collana", "ciondolo"], "es": ["collar", "colgante"]}, weight=0.6),
            Concept(id="product_affinity.jewelry.bracelet", label="Bracelet",
                    aliases={"fr": ["bracelet", "jonc"], "en": ["bracelet", "bangle", "cuff"],
                             "it": ["bracciale"], "es": ["pulsera"]}, weight=0.6),
            Concept(id="product_affinity.jewelry.earrings", label="Earrings",
                    aliases={"fr": ["boucles d'oreilles", "boucles"], "en": ["earrings", "studs"],
                             "it": ["orecchini"], "es": ["pendientes", "aretes"]}, weight=0.6),
        ]),
        Category(id="product_affinity.fragrance", label="Fragrance", concepts=[
            Concept(id="product_affinity.fragrance.perfume", label="Perfume",
                    aliases={"fr": ["parfum", "eau de parfum", "fragrance"], "en": ["perfume", "fragrance", "scent", "eau de parfum"],
                             "it": ["profumo"], "es": ["perfume", "fragancia"]}, weight=0.5),
            Concept(id="product_affinity.fragrance.candle", label="Candle / Home Fragrance",
                    aliases={"fr": ["bougie", "bougie parfumée"], "en": ["candle", "home fragrance", "scented candle"],
                             "it": ["candela"]}, weight=0.6),
        ]),
        Category(id="product_affinity.fashion", label="Ready-to-Wear & Shoes", concepts=[
            Concept(id="product_affinity.fashion.ready_to_wear", label="Ready-to-Wear",
                    aliases={"fr": ["prêt-à-porter", "vêtements"], "en": ["ready-to-wear", "clothing", "rtw"],
                             "it": ["prêt-à-porter", "abbigliamento"]}, weight=0.5),
            Concept(id="product_affinity.fashion.shoes", label="Shoes",
                    aliases={"fr": ["chaussures", "souliers", "sneakers", "baskets"],
                             "en": ["shoes", "sneakers", "loafers", "boots"],
                             "it": ["scarpe", "sneakers"], "es": ["zapatos"]}, weight=0.5),
            Concept(id="product_affinity.fashion.silk_scarf", label="Silk Scarf",
                    aliases={"fr": ["foulard", "carré de soie", "écharpe en soie"],
                             "en": ["silk scarf", "scarf"], "it": ["foulard"]}, weight=0.7),
            Concept(id="product_affinity.fashion.belt", label="Belt",
                    aliases={"fr": ["ceinture"], "en": ["belt"], "it": ["cintura"], "es": ["cinturón"]}, weight=0.5),
            Concept(id="product_affinity.fashion.sunglasses", label="Sunglasses",
                    aliases={"fr": ["lunettes de soleil", "lunettes"], "en": ["sunglasses", "eyewear"],
                             "it": ["occhiali da sole"]}, weight=0.5),
        ]),
        Category(id="product_affinity.home", label="Home & Art of Living", concepts=[
            Concept(id="product_affinity.home.decorative_objects", label="Decorative Objects",
                    aliases={"fr": ["objets décoratifs", "décoration", "art de vivre"],
                             "en": ["decorative objects", "home decor", "art of living"]}, weight=0.7),
            Concept(id="product_affinity.home.tableware", label="Tableware",
                    aliases={"fr": ["arts de la table", "vaisselle"], "en": ["tableware", "dinnerware"]}, weight=0.7),
            Concept(id="product_affinity.home.writing", label="Writing Instruments",
                    aliases={"fr": ["stylo", "écriture"], "en": ["pen", "writing instrument"]}, weight=0.7),
        ]),
    ])
    dimensions.append(dim_product)

    # ================================================================
    # 2. MAISON AFFINITY (20+ concepts)
    # ================================================================
    dim_maison = Dimension(id="maison_affinity", label="Maison Affinity", categories=[
        Category(id="maison_affinity.lvmh_fashion", label="LVMH Fashion & Leather", concepts=[
            Concept(id="maison_affinity.lvmh_fashion.louis_vuitton", label="Louis Vuitton",
                    aliases={"fr": ["louis vuitton", "vuitton"], "en": ["louis vuitton", "lv"]}, weight=0.5),
            Concept(id="maison_affinity.lvmh_fashion.dior", label="Dior",
                    aliases={"fr": ["dior", "christian dior"], "en": ["dior", "christian dior"]}, weight=0.6),
            Concept(id="maison_affinity.lvmh_fashion.fendi", label="Fendi",
                    aliases={"fr": ["fendi"], "en": ["fendi"]}, weight=0.7),
            Concept(id="maison_affinity.lvmh_fashion.celine", label="Celine",
                    aliases={"fr": ["céline", "celine"], "en": ["celine"]}, weight=0.7),
            Concept(id="maison_affinity.lvmh_fashion.loewe", label="Loewe",
                    aliases={"fr": ["loewe"], "en": ["loewe"]}, weight=0.7),
            Concept(id="maison_affinity.lvmh_fashion.givenchy", label="Givenchy",
                    aliases={"fr": ["givenchy"], "en": ["givenchy"]}, weight=0.7),
            Concept(id="maison_affinity.lvmh_fashion.berluti", label="Berluti",
                    aliases={"fr": ["berluti"], "en": ["berluti"]}, weight=0.8),
            Concept(id="maison_affinity.lvmh_fashion.rimowa", label="Rimowa",
                    aliases={"fr": ["rimowa"], "en": ["rimowa"]}, weight=0.7),
            Concept(id="maison_affinity.lvmh_fashion.kenzo", label="Kenzo",
                    aliases={"fr": ["kenzo"], "en": ["kenzo"]}, weight=0.7),
            Concept(id="maison_affinity.lvmh_fashion.loro_piana", label="Loro Piana",
                    aliases={"fr": ["loro piana"], "en": ["loro piana"]}, weight=0.8),
        ]),
        Category(id="maison_affinity.lvmh_watches_jewelry", label="LVMH Watches & Jewelry", concepts=[
            Concept(id="maison_affinity.lvmh_watches_jewelry.tiffany", label="Tiffany & Co",
                    aliases={"fr": ["tiffany"], "en": ["tiffany", "tiffany & co"]}, weight=0.6),
            Concept(id="maison_affinity.lvmh_watches_jewelry.bulgari", label="Bulgari",
                    aliases={"fr": ["bulgari", "bvlgari"], "en": ["bulgari", "bvlgari"]}, weight=0.7),
            Concept(id="maison_affinity.lvmh_watches_jewelry.tag_heuer", label="TAG Heuer",
                    aliases={"fr": ["tag heuer"], "en": ["tag heuer"]}, weight=0.7),
            Concept(id="maison_affinity.lvmh_watches_jewelry.hublot", label="Hublot",
                    aliases={"fr": ["hublot"], "en": ["hublot"]}, weight=0.8),
            Concept(id="maison_affinity.lvmh_watches_jewelry.zenith", label="Zenith",
                    aliases={"fr": ["zenith", "zénith"], "en": ["zenith"]}, weight=0.8),
            Concept(id="maison_affinity.lvmh_watches_jewelry.chaumet", label="Chaumet",
                    aliases={"fr": ["chaumet"], "en": ["chaumet"]}, weight=0.8),
            Concept(id="maison_affinity.lvmh_watches_jewelry.fred", label="Fred",
                    aliases={"fr": ["fred"], "en": ["fred jewelry"]}, weight=0.8),
        ]),
        Category(id="maison_affinity.lvmh_wines_spirits", label="LVMH Wines & Spirits", concepts=[
            Concept(id="maison_affinity.lvmh_wines_spirits.hennessy", label="Hennessy",
                    aliases={"fr": ["hennessy"], "en": ["hennessy"]}, weight=0.7),
            Concept(id="maison_affinity.lvmh_wines_spirits.dom_perignon", label="Dom Pérignon",
                    aliases={"fr": ["dom pérignon", "dom perignon"], "en": ["dom perignon", "dom pérignon"]}, weight=0.8),
            Concept(id="maison_affinity.lvmh_wines_spirits.moet_chandon", label="Moët & Chandon",
                    aliases={"fr": ["moët", "moët et chandon", "moet"], "en": ["moet", "moet & chandon"]}, weight=0.7),
        ]),
    ])
    dimensions.append(dim_maison)

    # ================================================================
    # 3. MATERIAL & CRAFTSMANSHIP (20+ concepts)
    # ================================================================
    dim_material = Dimension(id="material_craftsmanship", label="Material & Craftsmanship", categories=[
        Category(id="material_craftsmanship.leathers", label="Leathers", concepts=[
            Concept(id="material_craftsmanship.leathers.exotic_leather", label="Exotic Leather",
                    aliases={"fr": ["cuir exotique", "crocodile", "python", "autruche", "alligator"],
                             "en": ["exotic leather", "crocodile", "python", "ostrich", "alligator"]}, weight=0.9),
            Concept(id="material_craftsmanship.leathers.taurillon", label="Taurillon Leather",
                    aliases={"fr": ["taurillon", "cuir taurillon"], "en": ["taurillon leather", "taurillon"]}, weight=0.7),
            Concept(id="material_craftsmanship.leathers.epi_leather", label="Epi Leather",
                    aliases={"fr": ["cuir épi", "épi"], "en": ["epi leather", "epi"]}, weight=0.7),
            Concept(id="material_craftsmanship.leathers.monogram_canvas", label="Monogram Canvas",
                    aliases={"fr": ["monogramme", "toile monogram", "monogram"],
                             "en": ["monogram", "monogram canvas", "monogramme"]}, weight=0.4),
            Concept(id="material_craftsmanship.leathers.damier", label="Damier Canvas",
                    aliases={"fr": ["damier", "toile damier", "damier ébène", "damier azur"],
                             "en": ["damier", "damier canvas", "damier ebene", "damier azur"]}, weight=0.5),
        ]),
        Category(id="material_craftsmanship.metals", label="Metals & Stones", concepts=[
            Concept(id="material_craftsmanship.metals.gold", label="Gold",
                    aliases={"fr": ["or", "or jaune", "or blanc", "or rose"],
                             "en": ["gold", "yellow gold", "white gold", "rose gold"],
                             "it": ["oro"], "es": ["oro"]}, weight=0.5),
            Concept(id="material_craftsmanship.metals.platinum", label="Platinum",
                    aliases={"fr": ["platine"], "en": ["platinum"], "it": ["platino"]}, weight=0.8),
            Concept(id="material_craftsmanship.metals.diamond", label="Diamond",
                    aliases={"fr": ["diamant", "diamants"], "en": ["diamond", "diamonds"],
                             "it": ["diamante", "diamanti"], "es": ["diamante"]}, weight=0.7),
        ]),
        Category(id="material_craftsmanship.techniques", label="Techniques & Finishes", concepts=[
            Concept(id="material_craftsmanship.techniques.hand_stitched", label="Hand-Stitched",
                    aliases={"fr": ["cousu main", "fait main", "couture main"],
                             "en": ["hand-stitched", "handmade", "hand-crafted"]}, weight=0.8),
            Concept(id="material_craftsmanship.techniques.hot_stamping", label="Hot Stamping / Personalization",
                    aliases={"fr": ["marquage à chaud", "personnalisation", "initiales"],
                             "en": ["hot stamping", "personalization", "monogramming", "initials"]}, weight=0.6),
            Concept(id="material_craftsmanship.techniques.patina", label="Patina",
                    aliases={"fr": ["patine", "patiné"], "en": ["patina", "patinated"]}, weight=0.8),
            Concept(id="material_craftsmanship.techniques.engraving", label="Engraving",
                    aliases={"fr": ["gravure", "gravé"], "en": ["engraving", "engraved"],
                             "it": ["incisione"]}, weight=0.7),
        ]),
        Category(id="material_craftsmanship.sustainability", label="Sustainability", concepts=[
            Concept(id="material_craftsmanship.sustainability.eco_interest", label="Sustainability Interest",
                    aliases={"fr": ["durable", "éco-responsable", "recyclé", "développement durable"],
                             "en": ["sustainable", "eco-friendly", "recycled", "ethical"],
                             "it": ["sostenibile"], "es": ["sostenible"]}, weight=0.7),
            Concept(id="material_craftsmanship.sustainability.vegan_leather", label="Vegan Leather Interest",
                    aliases={"fr": ["cuir vegan", "sans cuir animal", "cuir végétal"],
                             "en": ["vegan leather", "faux leather", "plant-based leather"]}, weight=0.7),
        ]),
        Category(id="material_craftsmanship.fabrics", label="Fabrics", concepts=[
            Concept(id="material_craftsmanship.fabrics.silk", label="Silk",
                    aliases={"fr": ["soie", "en soie"], "en": ["silk"],
                             "it": ["seta"], "es": ["seda"]}, weight=0.6),
            Concept(id="material_craftsmanship.fabrics.cashmere", label="Cashmere",
                    aliases={"fr": ["cachemire"], "en": ["cashmere"],
                             "it": ["cashmere", "cachemire"]}, weight=0.7),
            Concept(id="material_craftsmanship.fabrics.denim", label="Denim",
                    aliases={"fr": ["denim", "jean"], "en": ["denim", "jeans"]}, weight=0.5),
            Concept(id="material_craftsmanship.fabrics.wool", label="Wool",
                    aliases={"fr": ["laine"], "en": ["wool"], "it": ["lana"], "es": ["lana"]}, weight=0.5),
        ]),
    ])
    dimensions.append(dim_material)

    # ================================================================
    # 4. PURCHASE CONTEXT (15+ concepts)
    # ================================================================
    dim_purchase = Dimension(id="purchase_context", label="Purchase Context", categories=[
        Category(id="purchase_context.type", label="Purchase Type", concepts=[
            Concept(id="purchase_context.type.self_purchase", label="Self Purchase",
                    aliases={"fr": ["pour moi", "pour lui-même", "pour elle-même", "achat personnel"],
                             "en": ["for myself", "self-purchase", "for himself", "for herself", "personal purchase"]}, weight=0.5),
            Concept(id="purchase_context.type.gift_purchase", label="Gift Purchase",
                    aliases={"fr": ["cadeau", "offrir", "pour offrir", "cadeau pour"],
                             "en": ["gift", "present", "gifting", "gift for"],
                             "it": ["regalo"], "es": ["regalo"], "de": ["geschenk"]}, weight=0.6),
            Concept(id="purchase_context.type.investment_piece", label="Investment Piece",
                    aliases={"fr": ["investissement", "pièce d'investissement", "placement"],
                             "en": ["investment piece", "investment", "collector's piece"]}, weight=0.8),
            Concept(id="purchase_context.type.replacement", label="Replacement",
                    aliases={"fr": ["remplacement", "remplacer"], "en": ["replacement", "replacing"]}, weight=0.6),
            Concept(id="purchase_context.type.impulse", label="Impulse Purchase",
                    aliases={"fr": ["coup de coeur", "impulsif"], "en": ["impulse", "love at first sight"]}, weight=0.7),
            Concept(id="purchase_context.type.corporate_gift", label="Corporate Gift",
                    aliases={"fr": ["cadeau d'entreprise", "cadeau corporate", "cadeau d'affaires"],
                             "en": ["corporate gift", "business gift", "company gift"]}, weight=0.8),
        ]),
        Category(id="purchase_context.decision", label="Decision Stage", concepts=[
            Concept(id="purchase_context.decision.browsing", label="Just Browsing",
                    aliases={"fr": ["juste regarder", "je regarde", "flâner"],
                             "en": ["just browsing", "just looking", "window shopping"]}, weight=0.4),
            Concept(id="purchase_context.decision.active_considering", label="Actively Considering",
                    aliases={"fr": ["hésite entre", "compare", "réfléchit"],
                             "en": ["considering", "comparing", "deciding between", "thinking about"]}, weight=0.6),
            Concept(id="purchase_context.decision.ready_to_buy", label="Ready to Buy",
                    aliases={"fr": ["prêt à acheter", "décidé", "va prendre"],
                             "en": ["ready to buy", "decided", "going to purchase"]}, weight=0.8),
            Concept(id="purchase_context.decision.collection_building", label="Collection Building",
                    aliases={"fr": ["collection", "collectionne", "agrandir sa collection"],
                             "en": ["collecting", "building collection", "collector"]}, weight=0.8),
        ]),
        Category(id="purchase_context.timeframe", label="Timeframe", concepts=[
            Concept(id="purchase_context.timeframe.urgent", label="Urgent / Immediate",
                    aliases={"fr": ["urgent", "aujourd'hui", "maintenant", "tout de suite"],
                             "en": ["urgent", "today", "right now", "immediately"],
                             "it": ["urgente"], "de": ["dringend"]}, weight=0.7),
            Concept(id="purchase_context.timeframe.weeks", label="Within Weeks",
                    aliases={"fr": ["dans les semaines", "prochaines semaines", "bientôt"],
                             "en": ["within weeks", "next few weeks", "soon"]}, weight=0.5),
            Concept(id="purchase_context.timeframe.no_rush", label="No Rush",
                    aliases={"fr": ["pas pressé", "prend son temps", "quand il trouvera"],
                             "en": ["no rush", "taking time", "no hurry"]}, weight=0.4),
            Concept(id="purchase_context.timeframe.event_deadline", label="Event Deadline",
                    aliases={"fr": ["avant l'événement", "pour la date", "avant le"],
                             "en": ["before the event", "by the date", "deadline"]}, weight=0.7),
        ]),
    ])
    dimensions.append(dim_purchase)

    # ================================================================
    # 5. OCCASION (15+ concepts)
    # ================================================================
    dim_occasion = Dimension(id="occasion", label="Occasion", categories=[
        Category(id="occasion.life_events", label="Life Events", concepts=[
            Concept(id="occasion.life_events.anniversary", label="Anniversary",
                    aliases={"fr": ["anniversaire de mariage", "anniversaire"], "en": ["anniversary", "wedding anniversary"],
                             "it": ["anniversario"], "es": ["aniversario"], "de": ["jubiläum"]}, weight=0.7),
            Concept(id="occasion.life_events.birthday", label="Birthday",
                    aliases={"fr": ["anniversaire", "fête"], "en": ["birthday"],
                             "it": ["compleanno"], "es": ["cumpleaños"], "de": ["geburtstag"]}, weight=0.6),
            Concept(id="occasion.life_events.wedding", label="Wedding",
                    aliases={"fr": ["mariage", "noces"], "en": ["wedding", "bridal"],
                             "it": ["matrimonio"], "es": ["boda"], "de": ["hochzeit"]}, weight=0.7),
            Concept(id="occasion.life_events.graduation", label="Graduation",
                    aliases={"fr": ["diplôme", "remise de diplôme"], "en": ["graduation"],
                             "it": ["laurea"], "es": ["graduación"]}, weight=0.7),
            Concept(id="occasion.life_events.engagement", label="Engagement",
                    aliases={"fr": ["fiançailles"], "en": ["engagement", "proposal"],
                             "it": ["fidanzamento"], "es": ["compromiso"]}, weight=0.8),
            Concept(id="occasion.life_events.baby", label="Baby / Birth",
                    aliases={"fr": ["naissance", "bébé"], "en": ["baby", "birth", "newborn"],
                             "it": ["nascita"], "es": ["nacimiento"]}, weight=0.7),
        ]),
        Category(id="occasion.calendar_events", label="Calendar Events", concepts=[
            Concept(id="occasion.calendar_events.christmas", label="Christmas",
                    aliases={"fr": ["noël", "fêtes de fin d'année"], "en": ["christmas", "holiday season", "xmas"],
                             "it": ["natale"], "es": ["navidad"], "de": ["weihnachten"]}, weight=0.6),
            Concept(id="occasion.calendar_events.valentines", label="Valentine's Day",
                    aliases={"fr": ["saint-valentin", "saint valentin"], "en": ["valentine's day", "valentines"],
                             "it": ["san valentino"]}, weight=0.7),
            Concept(id="occasion.calendar_events.mothers_day", label="Mother's Day",
                    aliases={"fr": ["fête des mères"], "en": ["mother's day", "mothers day"],
                             "it": ["festa della mamma"]}, weight=0.6),
            Concept(id="occasion.calendar_events.fathers_day", label="Father's Day",
                    aliases={"fr": ["fête des pères"], "en": ["father's day", "fathers day"]}, weight=0.6),
        ]),
        Category(id="occasion.cultural_events", label="Cultural Events", concepts=[
            Concept(id="occasion.cultural_events.chinese_new_year", label="Chinese New Year",
                    aliases={"fr": ["nouvel an chinois"], "en": ["chinese new year", "lunar new year"]}, weight=0.8,
                    relationships=[Relationship(type=RelationshipType.SEASONAL, target_id="occasion.calendar_events.christmas", weight=0.5)]),
            Concept(id="occasion.cultural_events.ramadan", label="Ramadan / Eid",
                    aliases={"fr": ["ramadan", "aïd"], "en": ["ramadan", "eid"]}, weight=0.8),
            Concept(id="occasion.cultural_events.diwali", label="Diwali",
                    aliases={"fr": ["diwali"], "en": ["diwali"]}, weight=0.8),
        ]),
        Category(id="occasion.travel", label="Travel", concepts=[
            Concept(id="occasion.travel.vacation", label="Vacation",
                    aliases={"fr": ["vacances", "voyage"], "en": ["vacation", "holiday", "trip"],
                             "it": ["vacanza", "viaggio"], "es": ["vacaciones", "viaje"], "de": ["urlaub", "reise"]}, weight=0.5),
            Concept(id="occasion.travel.business_trip", label="Business Trip",
                    aliases={"fr": ["voyage d'affaires", "déplacement professionnel"],
                             "en": ["business trip", "work travel"]}, weight=0.6),
        ]),
    ])
    dimensions.append(dim_occasion)

    # ================================================================
    # 6. BUDGET INTELLIGENCE (10+ concepts)
    # ================================================================
    dim_budget = Dimension(id="budget_intelligence", label="Budget Intelligence", categories=[
        Category(id="budget_intelligence.spending_tier", label="Spending Tier", concepts=[
            Concept(id="budget_intelligence.spending_tier.ultra_high", label="Ultra High Net Worth",
                    aliases={"fr": ["pas de limite", "peu importe le prix", "sans budget"],
                             "en": ["no budget limit", "price is not an issue", "money is no object"]}, weight=0.9),
            Concept(id="budget_intelligence.spending_tier.premium", label="Premium Spender",
                    aliases={"fr": ["budget confortable", "prêt à investir"],
                             "en": ["comfortable budget", "willing to invest"]}, weight=0.7),
            Concept(id="budget_intelligence.spending_tier.aspirational", label="Aspirational",
                    aliases={"fr": ["premier achat luxe", "se faire plaisir", "économisé pour"],
                             "en": ["first luxury purchase", "treat myself", "saved up for"]}, weight=0.6),
            Concept(id="budget_intelligence.spending_tier.budget_constrained", label="Budget Constrained",
                    aliases={"fr": ["budget limité", "budget serré", "pas trop cher"],
                             "en": ["limited budget", "tight budget", "not too expensive", "affordable"]}, weight=0.5,
                    relationships=[Relationship(type=RelationshipType.CONFLICTS,
                                                target_id="budget_intelligence.spending_tier.ultra_high")]),
        ]),
        Category(id="budget_intelligence.behavior", label="Budget Behavior", concepts=[
            Concept(id="budget_intelligence.behavior.price_sensitive", label="Price Sensitive",
                    aliases={"fr": ["sensible au prix", "compare les prix", "cherche le meilleur prix"],
                             "en": ["price sensitive", "comparing prices", "price conscious"]}, weight=0.5),
            Concept(id="budget_intelligence.behavior.value_seeker", label="Value Seeker",
                    aliases={"fr": ["rapport qualité-prix", "bon investissement"],
                             "en": ["value for money", "good investment", "worth it"]}, weight=0.6),
            Concept(id="budget_intelligence.behavior.stretch_willing", label="Willing to Stretch",
                    aliases={"fr": ["prêt à dépasser le budget", "peut monter un peu"],
                             "en": ["willing to stretch", "could go a bit higher"]}, weight=0.7),
            Concept(id="budget_intelligence.behavior.discount_seeking", label="Discount Seeking",
                    aliases={"fr": ["réduction", "soldes", "promotion", "remise"],
                             "en": ["discount", "sale", "promotion", "deal"]}, weight=0.5),
        ]),
        Category(id="budget_intelligence.amount", label="Amount Detection", concepts=[
            Concept(id="budget_intelligence.amount.detected", label="Budget Amount Detected",
                    aliases={"fr": ["budget", "budget de", "budget autour de"],
                             "en": ["budget", "budget of", "budget around"]}, weight=0.6),
            Concept(id="budget_intelligence.amount.range_stated", label="Price Range Stated",
                    aliases={"fr": ["entre", "de ... à"], "en": ["between", "range"]}, weight=0.6),
        ]),
    ])
    dimensions.append(dim_budget)

    # ================================================================
    # 7. LIFESTYLE SIGNALS (25+ concepts)
    # ================================================================
    dim_lifestyle = Dimension(id="lifestyle_signals", label="Lifestyle Signals", categories=[
        Category(id="lifestyle_signals.profession", label="Profession", concepts=[
            Concept(id="lifestyle_signals.profession.finance", label="Finance Professional",
                    aliases={"fr": ["banquier", "finance", "trader", "investisseur"],
                             "en": ["banker", "finance", "trader", "investor"],
                             "it": ["banchiere"], "es": ["banquero"]}, weight=0.7),
            Concept(id="lifestyle_signals.profession.lawyer", label="Lawyer",
                    aliases={"fr": ["avocat", "avocate", "juriste"], "en": ["lawyer", "attorney", "legal"],
                             "it": ["avvocato"], "es": ["abogado"], "de": ["anwalt"]}, weight=0.7),
            Concept(id="lifestyle_signals.profession.doctor", label="Doctor / Medical",
                    aliases={"fr": ["médecin", "docteur", "chirurgien"], "en": ["doctor", "surgeon", "physician"],
                             "it": ["medico", "chirurgo"], "es": ["médico", "cirujano"]}, weight=0.7),
            Concept(id="lifestyle_signals.profession.architect", label="Architect / Designer",
                    aliases={"fr": ["architecte", "designer", "architecte d'intérieur"],
                             "en": ["architect", "designer", "interior designer"],
                             "it": ["architetto"], "es": ["arquitecto"]}, weight=0.7),
            Concept(id="lifestyle_signals.profession.entrepreneur", label="Entrepreneur",
                    aliases={"fr": ["entrepreneur", "chef d'entreprise", "patron"],
                             "en": ["entrepreneur", "business owner", "ceo", "founder"]}, weight=0.7),
            Concept(id="lifestyle_signals.profession.artist", label="Artist",
                    aliases={"fr": ["artiste", "peintre", "sculpteur"], "en": ["artist", "painter", "sculptor"]}, weight=0.7),
            Concept(id="lifestyle_signals.profession.celebrity", label="Celebrity / Public Figure",
                    aliases={"fr": ["célébrité", "personnalité", "influenceur"],
                             "en": ["celebrity", "public figure", "influencer"]}, weight=0.9),
        ]),
        Category(id="lifestyle_signals.hobbies", label="Hobbies & Interests", concepts=[
            Concept(id="lifestyle_signals.hobbies.art_collector", label="Art Collector",
                    aliases={"fr": ["collectionneur d'art", "amateur d'art", "galerie"],
                             "en": ["art collector", "art enthusiast", "gallery"]}, weight=0.8),
            Concept(id="lifestyle_signals.hobbies.watch_collector", label="Watch Collector",
                    aliases={"fr": ["collectionneur de montres", "passionné d'horlogerie"],
                             "en": ["watch collector", "horology enthusiast"]}, weight=0.8),
            Concept(id="lifestyle_signals.hobbies.wine_connoisseur", label="Wine / Spirits Connoisseur",
                    aliases={"fr": ["amateur de vin", "oenophile", "connaisseur vins"],
                             "en": ["wine connoisseur", "wine lover", "oenophile"]}, weight=0.7),
            Concept(id="lifestyle_signals.hobbies.golf", label="Golf",
                    aliases={"fr": ["golf", "golfeur"], "en": ["golf", "golfer"]}, weight=0.6),
            Concept(id="lifestyle_signals.hobbies.equestrian", label="Equestrian",
                    aliases={"fr": ["équitation", "cheval", "dressage", "polo"],
                             "en": ["equestrian", "horse riding", "polo", "dressage"]}, weight=0.8),
            Concept(id="lifestyle_signals.hobbies.sailing", label="Sailing / Yachting",
                    aliases={"fr": ["voile", "nautisme", "yacht", "bateau"],
                             "en": ["sailing", "yachting", "yacht", "boating"]}, weight=0.8,
                    relationships=[Relationship(type=RelationshipType.IMPLIES,
                                                target_id="budget_intelligence.spending_tier.ultra_high", weight=0.6)]),
            Concept(id="lifestyle_signals.hobbies.skiing", label="Skiing",
                    aliases={"fr": ["ski", "station de ski"], "en": ["skiing", "ski resort"]}, weight=0.6),
            Concept(id="lifestyle_signals.hobbies.tennis", label="Tennis",
                    aliases={"fr": ["tennis"], "en": ["tennis"]}, weight=0.5),
            Concept(id="lifestyle_signals.hobbies.photography", label="Photography",
                    aliases={"fr": ["photographie", "photographe"], "en": ["photography", "photographer"]}, weight=0.6),
        ]),
        Category(id="lifestyle_signals.travel", label="Travel Patterns", concepts=[
            Concept(id="lifestyle_signals.travel.frequent_traveler", label="Frequent Traveler",
                    aliases={"fr": ["voyage souvent", "grand voyageur", "beaucoup de voyages"],
                             "en": ["frequent traveler", "travels often", "jet setter"]}, weight=0.6),
            Concept(id="lifestyle_signals.travel.asia_affinity", label="Asia Affinity",
                    aliases={"fr": ["japon", "tokyo", "asie", "singapour", "hong kong"],
                             "en": ["japan", "tokyo", "asia", "singapore", "hong kong"]}, weight=0.6),
            Concept(id="lifestyle_signals.travel.middle_east", label="Middle East",
                    aliases={"fr": ["dubaï", "dubai", "moyen-orient", "abu dhabi", "qatar"],
                             "en": ["dubai", "middle east", "abu dhabi", "qatar"]}, weight=0.6),
        ]),
        Category(id="lifestyle_signals.family", label="Family Context", concepts=[
            Concept(id="lifestyle_signals.family.married", label="Married",
                    aliases={"fr": ["mari", "femme", "époux", "épouse", "marié"],
                             "en": ["husband", "wife", "spouse", "married"]}, weight=0.4),
            Concept(id="lifestyle_signals.family.children", label="Has Children",
                    aliases={"fr": ["fils", "fille", "enfants", "enfant"],
                             "en": ["son", "daughter", "children", "kids"],
                             "it": ["figlio", "figlia"], "es": ["hijo", "hija"]}, weight=0.4),
        ]),
    ])
    dimensions.append(dim_lifestyle)

    # ================================================================
    # 8. CLIENT RELATIONSHIP (10+ concepts)
    # ================================================================
    dim_relationship = Dimension(id="client_relationship", label="Client Relationship", categories=[
        Category(id="client_relationship.tenure", label="Tenure", concepts=[
            Concept(id="client_relationship.tenure.first_visit", label="First Visit",
                    aliases={"fr": ["première visite", "première fois", "nouveau client", "nouvelle cliente"],
                             "en": ["first visit", "first time", "new client", "new customer"]}, weight=0.6),
            Concept(id="client_relationship.tenure.regular", label="Regular Client",
                    aliases={"fr": ["client fidèle", "habituée", "habitué", "revient souvent"],
                             "en": ["regular client", "loyal customer", "returning client", "comes back often"]}, weight=0.6),
            Concept(id="client_relationship.tenure.vip", label="VIP Client",
                    aliases={"fr": ["vip", "client vip", "très bon client", "gros client"],
                             "en": ["vip", "vip client", "top client", "high value client"]}, weight=0.8),
            Concept(id="client_relationship.tenure.lapsed", label="Lapsed Client",
                    aliases={"fr": ["pas revenu", "longtemps", "perdu de vue"],
                             "en": ["lapsed", "haven't seen", "been a while"]}, weight=0.7),
        ]),
        Category(id="client_relationship.engagement", label="Engagement Level", concepts=[
            Concept(id="client_relationship.engagement.highly_engaged", label="Highly Engaged",
                    aliases={"fr": ["très intéressé", "enthousiaste", "passionné"],
                             "en": ["very interested", "enthusiastic", "passionate"]}, weight=0.5),
            Concept(id="client_relationship.engagement.passive", label="Passive / Disengaged",
                    aliases={"fr": ["peu intéressé", "indifférent", "distant"],
                             "en": ["not interested", "indifferent", "disengaged"]}, weight=0.5),
            Concept(id="client_relationship.engagement.referred_by", label="Referred By",
                    aliases={"fr": ["recommandé par", "envoyé par", "suggestion de"],
                             "en": ["referred by", "recommended by", "sent by"]}, weight=0.6),
            Concept(id="client_relationship.engagement.ambassador", label="Brand Ambassador",
                    aliases={"fr": ["ambassadeur", "recommande la marque", "en parle autour"],
                             "en": ["ambassador", "recommends the brand", "word of mouth"]}, weight=0.8),
        ]),
        Category(id="client_relationship.acquisition", label="Acquisition Channel", concepts=[
            Concept(id="client_relationship.acquisition.walk_in", label="Walk-In",
                    aliases={"fr": ["passage", "passant", "entré dans la boutique"],
                             "en": ["walk-in", "walked in", "passing by"]}, weight=0.4),
            Concept(id="client_relationship.acquisition.online_to_store", label="Online to Store",
                    aliases={"fr": ["vu en ligne", "vu sur le site", "repéré sur internet"],
                             "en": ["saw online", "found on website", "online to store"]}, weight=0.5),
            Concept(id="client_relationship.acquisition.event_attendee", label="Event Attendee",
                    aliases={"fr": ["venu pour l'événement", "invité", "événement"],
                             "en": ["came for the event", "invited", "event attendee"]}, weight=0.6),
        ]),
    ])
    dimensions.append(dim_relationship)

    # ================================================================
    # 9. BEHAVIORAL MARKERS (10+ concepts)
    # ================================================================
    dim_behavioral = Dimension(id="behavioral_markers", label="Behavioral Markers", categories=[
        Category(id="behavioral_markers.decision_style", label="Decision Style", concepts=[
            Concept(id="behavioral_markers.decision_style.deliberate", label="Deliberate / Researched",
                    aliases={"fr": ["réfléchi", "prend le temps", "bien renseigné", "fait des recherches"],
                             "en": ["deliberate", "researched", "takes time", "well-informed"]}, weight=0.5),
            Concept(id="behavioral_markers.decision_style.impulsive", label="Impulsive / Emotional",
                    aliases={"fr": ["impulsif", "coup de coeur", "émotionnel"],
                             "en": ["impulsive", "emotional", "spontaneous"]}, weight=0.6),
            Concept(id="behavioral_markers.decision_style.indecisive", label="Indecisive",
                    aliases={"fr": ["indécis", "hésite", "ne sait pas"],
                             "en": ["indecisive", "hesitant", "can't decide", "unsure"]}, weight=0.5),
        ]),
        Category(id="behavioral_markers.communication", label="Communication Style", concepts=[
            Concept(id="behavioral_markers.communication.direct", label="Direct Communicator",
                    aliases={"fr": ["direct", "sait ce qu'il veut", "précis"],
                             "en": ["direct", "knows what they want", "specific"]}, weight=0.5),
            Concept(id="behavioral_markers.communication.needs_guidance", label="Needs Guidance",
                    aliases={"fr": ["a besoin de conseils", "demande des suggestions", "guide-moi"],
                             "en": ["needs guidance", "wants suggestions", "help me choose"]}, weight=0.5),
        ]),
        Category(id="behavioral_markers.brand_knowledge", label="Brand Knowledge", concepts=[
            Concept(id="behavioral_markers.brand_knowledge.expert", label="Brand Expert",
                    aliases={"fr": ["connaît bien la marque", "expert", "connaisseur"],
                             "en": ["knows the brand", "expert", "connoisseur", "knowledgeable"]}, weight=0.6),
            Concept(id="behavioral_markers.brand_knowledge.newcomer", label="Brand Newcomer",
                    aliases={"fr": ["découvre la marque", "ne connaît pas bien", "première expérience"],
                             "en": ["new to the brand", "discovering", "first experience"]}, weight=0.5),
        ]),
        Category(id="behavioral_markers.aesthetic", label="Aesthetic Markers", concepts=[
            Concept(id="behavioral_markers.aesthetic.minimalist", label="Minimalist",
                    aliases={"fr": ["minimaliste", "épuré", "sobre", "discret"],
                             "en": ["minimalist", "clean", "understated", "discreet"]}, weight=0.6),
            Concept(id="behavioral_markers.aesthetic.bold", label="Bold / Statement",
                    aliases={"fr": ["ostentatoire", "voyant", "statement", "bold"],
                             "en": ["bold", "statement", "flashy", "eye-catching"]}, weight=0.6,
                    relationships=[Relationship(type=RelationshipType.CONFLICTS,
                                                target_id="behavioral_markers.aesthetic.minimalist")]),
            Concept(id="behavioral_markers.aesthetic.classic", label="Classic / Timeless",
                    aliases={"fr": ["classique", "intemporel", "indémodable"],
                             "en": ["classic", "timeless", "traditional"]}, weight=0.5),
            Concept(id="behavioral_markers.aesthetic.trendy", label="Trendy / Fashion-Forward",
                    aliases={"fr": ["tendance", "à la mode", "fashion"],
                             "en": ["trendy", "fashion-forward", "on-trend"]}, weight=0.6),
        ]),
    ])
    dimensions.append(dim_behavioral)

    # ================================================================
    # 10. CULTURAL CONTEXT (10+ concepts)
    # ================================================================
    dim_cultural = Dimension(id="cultural_context", label="Cultural Context", categories=[
        Category(id="cultural_context.regional", label="Regional Preferences", concepts=[
            Concept(id="cultural_context.regional.japanese_aesthetics", label="Japanese Aesthetics",
                    aliases={"fr": ["esthétique japonaise", "wabi-sabi", "zen"],
                             "en": ["japanese aesthetics", "wabi-sabi", "zen", "japanese minimalism"]}, weight=0.8),
            Concept(id="cultural_context.regional.middle_east_modesty", label="Middle East Modesty",
                    aliases={"fr": ["mode pudique", "modeste", "couvrant"],
                             "en": ["modest fashion", "covered", "conservative dress"]}, weight=0.7),
            Concept(id="cultural_context.regional.european_understated", label="European Understated",
                    aliases={"fr": ["élégance discrète", "chic parisien", "dolce vita"],
                             "en": ["understated elegance", "parisian chic", "european style"]}, weight=0.6),
            Concept(id="cultural_context.regional.american_luxury", label="American Luxury",
                    aliases={"fr": ["luxe américain", "style new-yorkais"],
                             "en": ["american luxury", "new york style", "hollywood glamour"]}, weight=0.6),
        ]),
        Category(id="cultural_context.language", label="Language & Background", concepts=[
            Concept(id="cultural_context.language.multilingual", label="Multilingual Client",
                    aliases={"fr": ["parle plusieurs langues", "multilingue"],
                             "en": ["multilingual", "speaks multiple languages"]}, weight=0.5),
            Concept(id="cultural_context.language.tourist", label="Tourist / Visitor",
                    aliases={"fr": ["touriste", "de passage", "en visite"],
                             "en": ["tourist", "visiting", "traveler"]}, weight=0.5),
            Concept(id="cultural_context.language.expat", label="Expatriate",
                    aliases={"fr": ["expatrié", "expat", "vit à l'étranger"],
                             "en": ["expat", "expatriate", "lives abroad"]}, weight=0.6),
        ]),
        Category(id="cultural_context.generation", label="Generational Signals", concepts=[
            Concept(id="cultural_context.generation.gen_z", label="Gen Z",
                    aliases={"fr": ["gen z", "jeune", "génération z"], "en": ["gen z", "young", "generation z"]}, weight=0.6),
            Concept(id="cultural_context.generation.millennial", label="Millennial",
                    aliases={"fr": ["millennial", "millennials"], "en": ["millennial", "millennials"]}, weight=0.5),
            Concept(id="cultural_context.generation.mature", label="Mature Client",
                    aliases={"fr": ["mature", "senior", "expérimenté"], "en": ["mature", "senior", "experienced"]}, weight=0.5),
        ]),
    ])
    dimensions.append(dim_cultural)

    # ================================================================
    # 11. SENTIMENT EXTRACTION (10+ concepts)
    # ================================================================
    dim_sentiment = Dimension(id="sentiment_extraction", label="Sentiment Extraction", categories=[
        Category(id="sentiment_extraction.positive", label="Positive Signals", concepts=[
            Concept(id="sentiment_extraction.positive.delight", label="Delight / Excitement",
                    aliases={"fr": ["ravi", "enchanté", "adore", "magnifique", "superbe", "coup de coeur"],
                             "en": ["delighted", "love it", "amazing", "gorgeous", "wonderful", "stunning"]}, weight=0.5),
            Concept(id="sentiment_extraction.positive.satisfaction", label="Satisfaction",
                    aliases={"fr": ["satisfait", "content", "très bien", "parfait", "excellent"],
                             "en": ["satisfied", "happy", "perfect", "excellent", "great"]}, weight=0.4),
            Concept(id="sentiment_extraction.positive.trust", label="Trust / Confidence",
                    aliases={"fr": ["confiance", "fait confiance", "fidèle"],
                             "en": ["trust", "confidence", "reliable", "trustworthy"]}, weight=0.5),
        ]),
        Category(id="sentiment_extraction.negative", label="Negative Signals", concepts=[
            Concept(id="sentiment_extraction.negative.frustration", label="Frustration",
                    aliases={"fr": ["frustré", "déçu", "mécontent", "agacé", "pas content"],
                             "en": ["frustrated", "disappointed", "unhappy", "annoyed"]}, weight=0.6),
            Concept(id="sentiment_extraction.negative.complaint", label="Complaint / Issue",
                    aliases={"fr": ["problème", "plainte", "défaut", "cassé", "abîmé"],
                             "en": ["problem", "complaint", "defect", "broken", "damaged"]}, weight=0.7),
            Concept(id="sentiment_extraction.negative.price_objection", label="Price Objection",
                    aliases={"fr": ["trop cher", "cher", "hors budget", "au-dessus du budget"],
                             "en": ["too expensive", "overpriced", "out of budget", "above budget"]}, weight=0.6),
        ]),
        Category(id="sentiment_extraction.signals", label="Behavioral Signals", concepts=[
            Concept(id="sentiment_extraction.signals.urgency", label="Urgency",
                    aliases={"fr": ["pressé", "urgence", "il faut que", "absolument"],
                             "en": ["urgent", "rush", "need it by", "must have"]}, weight=0.6),
            Concept(id="sentiment_extraction.signals.hesitation", label="Hesitation",
                    aliases={"fr": ["hésite", "pas sûr", "je ne sais pas", "peut-être"],
                             "en": ["hesitant", "not sure", "maybe", "uncertain"]}, weight=0.5),
            Concept(id="sentiment_extraction.signals.brand_switching", label="Brand Switching Signal",
                    aliases={"fr": ["essayer autre chose", "changer de marque", "voir ailleurs"],
                             "en": ["try something else", "switch brands", "look elsewhere"]}, weight=0.8),
            Concept(id="sentiment_extraction.signals.loyalty_signal", label="Loyalty Signal",
                    aliases={"fr": ["toujours acheté ici", "ma marque préférée", "je reviens toujours"],
                             "en": ["always bought here", "my favorite brand", "always come back"]}, weight=0.6),
            Concept(id="sentiment_extraction.signals.recommendation_intent", label="Recommendation Intent",
                    aliases={"fr": ["je recommande", "vais en parler", "dire à mes amis"],
                             "en": ["recommend", "will tell friends", "word of mouth"]}, weight=0.6),
        ]),
    ])
    dimensions.append(dim_sentiment)

    # ================================================================
    # 12. SERVICE MAPPING (10+ concepts)
    # ================================================================
    dim_service = Dimension(id="service_mapping", label="Service Mapping", categories=[
        Category(id="service_mapping.actions", label="Advisor Actions", concepts=[
            Concept(id="service_mapping.actions.product_shown", label="Product Demonstrated",
                    aliases={"fr": ["montré", "présenté", "essayé", "a essayé"],
                             "en": ["showed", "demonstrated", "tried on", "presented"]}, weight=0.4),
            Concept(id="service_mapping.actions.sent_photos", label="Sent Photos / Catalog",
                    aliases={"fr": ["envoyé des photos", "envoyé le catalogue", "envoyé par whatsapp"],
                             "en": ["sent photos", "sent catalog", "sent via whatsapp"]}, weight=0.5),
            Concept(id="service_mapping.actions.appointment_made", label="Appointment Made",
                    aliases={"fr": ["rendez-vous", "rdv pris", "a pris rendez-vous"],
                             "en": ["appointment", "booked appointment", "scheduled visit"],
                             "it": ["appuntamento"], "es": ["cita"]}, weight=0.6),
        ]),
        Category(id="service_mapping.follow_up", label="Follow-up Needed", concepts=[
            Concept(id="service_mapping.follow_up.callback", label="Callback Needed",
                    aliases={"fr": ["rappeler", "à recontacter", "recontacter"],
                             "en": ["call back", "follow up", "contact again"]}, weight=0.6),
            Concept(id="service_mapping.follow_up.reserve_item", label="Item to Reserve",
                    aliases={"fr": ["réserver", "mettre de côté", "garder"],
                             "en": ["reserve", "hold", "set aside", "put aside"]}, weight=0.7),
            Concept(id="service_mapping.follow_up.special_order", label="Special Order",
                    aliases={"fr": ["commande spéciale", "sur commande", "commander"],
                             "en": ["special order", "custom order", "order"]}, weight=0.7),
            Concept(id="service_mapping.follow_up.waitlist", label="Waitlist",
                    aliases={"fr": ["liste d'attente", "en attente"],
                             "en": ["waitlist", "waiting list"]}, weight=0.7),
        ]),
        Category(id="service_mapping.channel", label="Channel Preference", concepts=[
            Concept(id="service_mapping.channel.whatsapp", label="WhatsApp",
                    aliases={"fr": ["whatsapp"], "en": ["whatsapp"]}, weight=0.5),
            Concept(id="service_mapping.channel.email", label="Email",
                    aliases={"fr": ["email", "mail", "e-mail", "courriel"],
                             "en": ["email", "e-mail"]}, weight=0.4),
            Concept(id="service_mapping.channel.phone", label="Phone",
                    aliases={"fr": ["téléphone", "appeler"], "en": ["phone", "call"]}, weight=0.4),
        ]),
        Category(id="service_mapping.personalization", label="Personalization Requests", concepts=[
            Concept(id="service_mapping.personalization.engraving", label="Engraving Request",
                    aliases={"fr": ["gravure", "faire graver"], "en": ["engraving", "engrave"]}, weight=0.7),
            Concept(id="service_mapping.personalization.monogramming", label="Monogramming Request",
                    aliases={"fr": ["marquage", "initiales", "monogramme personnalisé"],
                             "en": ["monogramming", "initials", "personalized monogram"]}, weight=0.6),
        ]),
    ])
    dimensions.append(dim_service)

    # ================================================================
    # 13. COMPETITIVE INTELLIGENCE (10+ concepts)
    # ================================================================
    dim_competitive = Dimension(id="competitive_intelligence", label="Competitive Intelligence", categories=[
        Category(id="competitive_intelligence.competitor_brands", label="Competitor Brands", concepts=[
            Concept(id="competitive_intelligence.competitor_brands.hermes", label="Hermès",
                    aliases={"fr": ["hermès", "hermes", "birkin", "kelly"], "en": ["hermes", "hermès", "birkin", "kelly"]}, weight=0.8),
            Concept(id="competitive_intelligence.competitor_brands.chanel", label="Chanel",
                    aliases={"fr": ["chanel", "coco chanel"], "en": ["chanel"]}, weight=0.7),
            Concept(id="competitive_intelligence.competitor_brands.gucci", label="Gucci",
                    aliases={"fr": ["gucci"], "en": ["gucci"]}, weight=0.6),
            Concept(id="competitive_intelligence.competitor_brands.prada", label="Prada",
                    aliases={"fr": ["prada"], "en": ["prada"]}, weight=0.7),
            Concept(id="competitive_intelligence.competitor_brands.cartier", label="Cartier",
                    aliases={"fr": ["cartier"], "en": ["cartier"]}, weight=0.7),
            Concept(id="competitive_intelligence.competitor_brands.rolex", label="Rolex",
                    aliases={"fr": ["rolex"], "en": ["rolex"]}, weight=0.7),
            Concept(id="competitive_intelligence.competitor_brands.bottega_veneta", label="Bottega Veneta",
                    aliases={"fr": ["bottega veneta", "bottega"], "en": ["bottega veneta", "bottega"]}, weight=0.7),
            Concept(id="competitive_intelligence.competitor_brands.saint_laurent", label="Saint Laurent",
                    aliases={"fr": ["saint laurent", "ysl", "yves saint laurent"],
                             "en": ["saint laurent", "ysl", "yves saint laurent"]}, weight=0.7),
            Concept(id="competitive_intelligence.competitor_brands.balenciaga", label="Balenciaga",
                    aliases={"fr": ["balenciaga"], "en": ["balenciaga"]}, weight=0.6),
        ]),
        Category(id="competitive_intelligence.signals", label="Competitive Signals", concepts=[
            Concept(id="competitive_intelligence.signals.brand_comparison", label="Brand Comparison",
                    aliases={"fr": ["compare avec", "par rapport à", "mieux que", "versus"],
                             "en": ["compared to", "versus", "better than", "vs"]}, weight=0.7),
            Concept(id="competitive_intelligence.signals.defector", label="Brand Defector",
                    aliases={"fr": ["quitte la marque", "déçu de", "vient de chez"],
                             "en": ["leaving brand", "disappointed with", "coming from", "switching from"]}, weight=0.8),
            Concept(id="competitive_intelligence.signals.loyal_elsewhere", label="Loyal to Competitor",
                    aliases={"fr": ["fidèle à", "client de", "achète chez"],
                             "en": ["loyal to", "customer of", "buys from"]}, weight=0.6),
        ]),
    ])
    dimensions.append(dim_competitive)

    # ================================================================
    # 14. COMMERCIAL SCORING (10+ concepts)
    # ================================================================
    dim_commercial = Dimension(id="commercial_scoring", label="Commercial Scoring", categories=[
        Category(id="commercial_scoring.cross_sell", label="Cross-Sell Opportunities", concepts=[
            Concept(id="commercial_scoring.cross_sell.fragrance_to_fashion", label="Fragrance → Fashion",
                    aliases={"fr": ["aussi intéressé par les vêtements", "a regardé le prêt-à-porter"],
                             "en": ["also interested in clothing", "looked at ready-to-wear"]}, weight=0.7),
            Concept(id="commercial_scoring.cross_sell.leather_to_accessories", label="Leather → Accessories",
                    aliases={"fr": ["aussi les accessoires", "a regardé les ceintures"],
                             "en": ["also accessories", "looked at belts", "matching accessories"]}, weight=0.6),
            Concept(id="commercial_scoring.cross_sell.multiple_categories", label="Multi-Category Interest",
                    aliases={"fr": ["intéressé par plusieurs catégories", "total look"],
                             "en": ["multiple categories", "total look", "full outfit"]}, weight=0.7),
        ]),
        Category(id="commercial_scoring.upsell", label="Upsell Signals", concepts=[
            Concept(id="commercial_scoring.upsell.trade_up", label="Trade-Up Potential",
                    aliases={"fr": ["monter en gamme", "plus haut de gamme", "version premium"],
                             "en": ["trade up", "higher end", "premium version", "upgrade"]}, weight=0.7),
            Concept(id="commercial_scoring.upsell.limited_edition", label="Limited Edition Interest",
                    aliases={"fr": ["édition limitée", "série limitée", "exclusif", "collector"],
                             "en": ["limited edition", "exclusive", "collector's", "special edition"]}, weight=0.8),
        ]),
        Category(id="commercial_scoring.timing", label="Timing Signals", concepts=[
            Concept(id="commercial_scoring.timing.seasonal_peak", label="Seasonal Peak Timing",
                    aliases={"fr": ["pour noël", "pour les fêtes", "pour la rentrée"],
                             "en": ["for christmas", "for the holidays", "for back to school"]}, weight=0.5),
            Concept(id="commercial_scoring.timing.pre_travel", label="Pre-Travel Purchase",
                    aliases={"fr": ["avant le voyage", "pour le voyage", "avant de partir"],
                             "en": ["before the trip", "for the trip", "before traveling"]}, weight=0.6),
        ]),
        Category(id="commercial_scoring.intent", label="Purchase Intent Signals", concepts=[
            Concept(id="commercial_scoring.intent.high_intent", label="High Purchase Intent",
                    aliases={"fr": ["va acheter", "veut acheter", "prêt à payer", "décidé"],
                             "en": ["will buy", "wants to buy", "ready to pay", "decided"]}, weight=0.7),
            Concept(id="commercial_scoring.intent.low_intent", label="Low Purchase Intent / Just Looking",
                    aliases={"fr": ["juste voir", "pas pour acheter", "regarde seulement"],
                             "en": ["just looking", "not buying today", "browsing only"]}, weight=0.4),
            Concept(id="commercial_scoring.intent.return_visit", label="Will Return",
                    aliases={"fr": ["va revenir", "revient bientôt", "reviendra"],
                             "en": ["will come back", "returning soon", "coming back"]}, weight=0.6),
        ]),
    ])
    dimensions.append(dim_commercial)

    return Ontology(dimensions=dimensions)


def main():
    """Generate and save the seed ontology."""
    ontology = _build_ontology()

    # Summary
    total_concepts = len(ontology.all_concepts())
    print(f"\n{'='*60}")
    print(f"  Ontology v2 Seed Generation")
    print(f"{'='*60}")
    print(f"  Dimensions: {len(ontology.dimensions)}")
    print(f"  Total concepts: {total_concepts}")
    print()

    for dim in ontology.dimensions:
        dim_concepts = sum(len(cat.concepts) for cat in dim.categories)
        print(f"  {dim.id:<30} {dim_concepts:>3} concepts  ({len(dim.categories)} categories)")

    # Validate
    ids = [c.id for c in ontology.all_concepts()]
    dupes = len(ids) - len(set(ids))
    if dupes:
        print(f"\n  WARNING: {dupes} duplicate concept IDs found!")
    else:
        print(f"\n  No duplicate IDs.")

    empty_aliases = [c.id for c in ontology.all_concepts() if not c.all_aliases()]
    if empty_aliases:
        print(f"  WARNING: {len(empty_aliases)} concepts with no aliases!")
    else:
        print(f"  All concepts have aliases.")

    # Save
    output_path = TAXONOMY_DIR / "ontology_v2.json"
    save_ontology(ontology, output_path)
    print(f"\n  Saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
