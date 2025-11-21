# Systèeme de recommandation - filtration basée sur le contenu

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Dataset (exp simple)
data = {
        "product_id": list(range(1, 21)),
        "title": [
                  "smartphone Sanmsung Galaxy A50",
                  "Coque silicone pour smartphone",
                  "Ordinateur portable HP Pavilion",
                  "Disque dur externe 1To",
                  "Chargeur rapide USB-C 25W",
                  "Casque bluetooth sans fil",
                  "Écouteurs intra-auriculaires JBL",
                  "Smartwatch Samsung Galaxy Watch",
                  "Souris sans fil Logitech",
                  "Clavier mécanique RGB Corsair",
                  "Télévision 4K LG 55 pouces",
                  "Batte électrique multi-fonction",
                  "Aspirateur robot 3 en 1",
                  "Réfrigérateur double porte Samsung",
                  "Machine à café Nespresso",
                  "Haltere ajustable 20kg",
                  "Tapis de sport anti-dérapant",
                  "Chaussures de running Nike",
                  "Montre analogieque Rolex",
                  "Sac à dos PC Portable 15 pouces",
        ],
        "description": [
            "Téléphone Android performant avec appareil photo",
            "Protection flexible pour smartphone toutes marques",
            "Laptop performant pour travail et études",
            "Stockage externe compatible PC et Mac",
            "Chargeur rapide compatible Android et iPhone avec câble USB-C",
            "Casque audio Bluetooth avec réduction de bruit et longue autonomie",
            "Écouteurs confortables offrant un son puissant et des basses profondes",
            "Montre connectée avec suivi sportif, capteur cardiaque et GPS",
            "Souris sans fil ergonomique pour bureau, gaming et productivité",
            "Clavier mécanique RGB idéal pour gaming, frappe rapide et précision",
            "Télévision 4K ultra haute définition avec écran LED 55 pouces",
            "Batteur électrique de cuisine pour mélanger, fouetter et pétrir facilement",
            "Aspirateur robot intelligent avec navigation et nettoyage automatique",
            "Réfrigérateur double porte grande capacité avec distributeur de glace",
            "Machine à café automatique compatible capsules Nespresso",
            "Haltère ajustable de 2 à 20kg pour musculation à domicile",
            "Tapis de sport antidérapant idéal pour yoga, fitness et exercices",
            "Chaussures de running légères avec amorti pour course à pied",
            "Montre analogique élégante en acier pour homme",
            "Sac à dos résistant avec compartiment rembourré pour PC portable"
        ]
}

# création dataframe pour facilité la manipulation des données
df = pd.DataFrame(data)

"""
2. vectorisation tf - idf
stop_words : retirer les mots inutiles : "the", "for", "and", "", etc
"""
vectorizer = TfidfVectorizer(stop_words='english')

print(vectorizer)
#df['text'] = df['title'] + ' ' + df['description']
tfidf_matrix = vectorizer.fit_transform(df['description'])
print(tfidf_matrix.shape) # (25 produits, nombre de mots uniques)

"""
3. similarité cosinus
similarity_matrix[i][j] : score de similarité entre les produits i et j
et plus le score est proche de 1, plus ils sont similaires
"""
similarity_matrix = cosine_similarity(tfidf_matrix)

# 4. fonction de recommandation de n produits similaires à product_id
def recommandation(product_id, n=3):
    index = df[df.product_id == product_id].index[0] # index du produit dans le dataframe
    scores = list(enumerate((similarity_matrix[index]))) # créer une liste de tuples (index produit, score de similarité)
    scores = sorted(scores, key=lambda x: x[1], reverse=True) # trier la liste par score de similarité du plus haut au plus bas
    top = scores[1:n+1] # prendre les n produits les plus similaires en supprimant le produit lui-meme (scores[0])
    
    print(f"\nRecommandations pour : {df.loc[index, 'title']}\n")
    for i, score in top:
        # afficher le titre et la similarité avec un score arrondi sur 2 chiffres
        print(f'- {df.loc[i, 'title']} (similarité : {round(score, 2)})')

recommandation(2) # teste le système avec le produit d'id 2