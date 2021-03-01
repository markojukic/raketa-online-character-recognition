# Raketa: Online character recognition

## Instaliranje potrebnih paketa:

Potrebno je imati instaliran Python 3.9. Svi potrebni paketi instaliraju se pokretanjem naredbe:
```
pip install -r requirements.txt
```
## Opisi datoteka:

`requirements.txt` - korišteni Python paketi

`drawing.py` - osnovna klasa za crteže s kojom radimo

`ujipenchars2.py` - parser za UJIpenchars2 skup podatka

`preprocessing.py` - transformacije crteža (skaliranje, resampling, crtanje slika...)

`hosvdclassifier.py` - implementacija HOSVD klasifikatora za generalnu upotrebu

`dtw.py` - funkcije za efikasno računanje DTW i RSIDTW

`classifier_pickle.py` - kod za spremanje modela na disk

`data/` - korišteni podaci

### Interaktivna aplikacija

`app.py` - interaktivna aplikacija za prepoznavanje znamenaka

`models/` - spremljeni modeli (za `app.py`)

### Jupyter bilježnice:

`HOSVD.ipynb` - Klasifikacija znamenki pomoću HOSVD: analiza modela

`HOSVD-Video.ipynb` - Klasifikacija znamenki pomoću slika i videa

`HOSVD-Gradient.ipynb` - Klasifikacija znamenki pomoću gradijenta

`k-NN.ipynb` - Klasifikacija znamenki pomoću k najbližih susjeda koristeći DTW i RSIDTW

`SVM.ipynb` - Klasifikacija znamenki pomoću metode potpornih vektora koristeći DTW i RSIDTW
