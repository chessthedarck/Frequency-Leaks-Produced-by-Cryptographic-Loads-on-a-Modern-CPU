I/ Problematiques de recherche

Ce projet vise a etudier si l'execution d'algorithmes cryptographiques peut produire des signatures mesurables au niveau du systeme, en particulier via:

- les variations de frequence CPU
- les emissions acoustiques
- les caracteristiques temporelles du processeur

L'objectif final est d'explorer si ces signatures peuvent etre exploitees dans un contexte de cryptanalyse par canal auxiliaire.

En particulier, on s'inspire des travaux de Genkin, Shamir et Tromer (2013) sur les attaques acoustiques contre RSA.

II/ Construction de l'environnement experimental

Pour mener les tests, nous construisons un pipeline experimental experimental simple et reproductible.

# OS utilise
Rasberry OS LITE

# Objectif

Creer un environnement permettant de :

    - Executer un algorithme cryptographique
    - Enregistrer les frequences CPU
    - Analyser les variations temporelles

III/ Mise en place du logger CPU

Un script de monitoring CPU a ete mis en place pour enregistrer:

    - Numero de CPU
    - Frequence actuelle
    - Frequence Min/Max
    - Gouverneur CPU

Les donnees sont enregistrees sous forme de CSV.

Cela permet de produire une trace temporelle du comportement du CPU.

IV/ Definition d'un scenario baseline (S0)

Afin d'avoir un point de comparaison, un scenario baseline a ete defini.

S0: systeme sans charge cryptographique
Le systeme est laisse quasi idle, sans execution d'algorithmes lourds.

Les frequences CPU enregistrees representent alors:

    - Le comportements normal du systeme
    - Le bruit de fond du scheduler
    - Les interruptions systeme

V/ Definition d'un scenario cryptographique

Un test simple a ete mis en place pour generer une charge cryptographique.

# Algorithme utilise

RSA (cryptographique asymetrique).

# Implementation

Utilisation d'OpenSSL:

# Experience realisee

Une boucle executant 1000 signatures RSA.

Un script enregistre pour chaque iteration:

    - Tiemstamp de debut
    - Timestamp de fin
    - Duree de l'operation

Cela produit un fichier ".csv"

VI/ Acquisition simultanee des frequences CPU

Pendant l'execution des 1000 signatures RSA, le logger CPU tourne en parallele.

Ainsi deux types de donnees sont collectees :

    - Trace CPU
    - Trace Cryptographique

Cela permet de correler activite cryptographique et comportement CPU.

VII/ Pipeline d'analyse Python

Un pipeline Python a ete developpe pour:

    - Importer les CSV
    - Reconstruire le temps relatif
    - calculer la frequence moyenne CPU
    - Produire des visualisations

Les graphiques montrent :

    - Evolution temporelle de la frequence de la frequence CPU
    - Enveloppe min/max
    - Comparaison baseline vs crypto
    - Difference entre les deux signaux

VIII/ Resultats observes

Les graphiques montrent plusieurs phenomenes interessants :

    1. RSA augmente la variabilite CPU
        La frequence CPU devient plus instable pendant l'execution des signatures.

    2. Apparition de bursts
        On observe des pics de frequence intermittents, correspondant probablement a des phases de calcul intensif.
    
    3. Difference statistique avec la baseline
        Meme si la moyenne reste proche, la structure temporelle change:

            - plus de dispersion
            - variations plus rapides
            - dynamique differente

IX/ Interpretation scientifique

Les resultats suggerent que:
    - l'execution d'operations RSA influence le comportement du processeur
    - cette l'influence est observable via des metriques simples
    - ces variations pourraient constituer une signature indirecte de l'indirecte  de l'algorithme

Cela correspond au principe des Side-Channel attacks, ou l'on exploite :
    - Le temps
    - l'energie
    - les emissions electromagnetiques
    - les vibrations acoustiques

X/ Position dans la demarche de recherche

Le travail realise correspond a une premiere etape exploratoire.

Objectif en cours:

    * Mise en place d'un place d'un environnement reproductible
    * Acquisition de donnees
    * Visualisation des effets d'un algorithme cryptographique


----------------------------

Premiere etape : il y a de quoi publie 

Purement Data Maining
rang A et ptr rang B

utilise une premiere cle 

qu'elle cle a utilise pour qu'elle attaque

attaque par template 
probleme de classifieur

on peux pas se permettre d'avoir 

que des morceaux de la cle
collecter des templates
1 octets de la cle 

deja fait en utilisant des rayonnements pas en acoustique

reduire les labels jusque 150 labels (labels parties de la cle)
a 2**8 (fesable en ML)

pas 2**256 (pas fesable en ML)



metrique d'explicabilite 


poids de hamming ne suffit pas pourquoi?

le poids de hamming ne nous suffiras pas 


1000 traces pour chaque possibilite 
dizaines de traces par cas possible


## Methodes ML

Attention discretisation possible
Qualite d'apprentissage depend des donnes precises

Guaussien et centree 
mesures toutes les traces 
analyse en 
garder que ce qui est significatif

en quoi le bruit me gene ?

# labellisation