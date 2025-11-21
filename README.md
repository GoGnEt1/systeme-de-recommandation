# Système de recommandation - filtrage basé sur le contenu en utilisant Scikit-Learn

### Description :

Mini-projet de machine learning utilisant TF-IDF et la Similarité Cosinus pour recommander automatiquement des articles/produits similaires en se basant uniquement sur leur contenu textuel.

### Objectifs :

• Prétraitement du texte

• Vectorisation TF-IDF

• Calcul de similarité cosinus

• Recommandation automatisée

• implémentation Scikit-Learn

• Comprehension du Content-Based Filtering

### Bibliothèques utilisées :

**Python** : Environnent général du projet.

**Pandas** : permet de manipuler (charger, créer un DataFrame, filtrer, nettoyage, ..) des données tabulaires.

**Scikit-Learn** :

_• TfidfVectorizer_ : transforme du texte en vecteurs numériques basés sur TF-IDF où chaque dimension représente un mot important.

```bat
  TF : Term Frequency (fréquence du mot) = (nbre totat du terme présent dans le doc) / (somme des mots du doc)

  IDF : Iverse Document Frequency (fréquence inverse du doc) = log((nbre totat de docs) / (nbre de docs contenant le terme))
```

_• cosine_similarity_ : calcule la similarité entre vecteurs (tous les couples de produits ) à l'aide du cosinus. C'est la mésure la plus utilisée en NPL pour les moteur de recommandation.

#### Formule mathématiques :

```bat
 cosine_similarity(A, B) = (produit sacalaire de A et B) / (le produit de la norme ecludienne de A et de B)

 Interprétation :
 • si angle = 0° → score = 1 → textes identiques
 • si angle = 60° → score = 0.5 → textes moyennement proches
 • si angle = 90° → score = 0 → pas de relation
```

### Fonctionnement :

1. Charger un petit dataset (20 produits)
2. Convertir le texte en vecteurs TF-IDF
3. Nettoyer le texte (stopwords, minuscules,...)
4. Vectoriser TF-IDF
5. Calculer la similarité cosinus
6. Recommander les k produits les plus similaires

### Apperçu du code

```bat
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description'])
similarity_matrix = cosine_similarity(tfidf_matrix)
```

# Mon Notebook : Généralités_ML.ipynb

> Ce projet de recommadation s'appuie sur ce Notebook qui est un résumé structuré de mon apprentissage en Machine Learning accessible au [Généralités_ML.ipynb](https://colab.research.google.com/drive/1uvB2eNU445_3NYbnA8oCnItwhvm-aifU#scrollTo=nzUmBxLIZpwC)
