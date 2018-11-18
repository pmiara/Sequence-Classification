# Python module for comparing sequence classification algorithms.

# Getting Started
## Installing
### Docker
```
requirements.txt - zależności do Pythona

docker-compose up - uruchamianie aplikacji, w razie błędów dodaj sudo

docker image rm -f sequences - po zmianach w requirements.txt usuwanie istniejącego obrazu dockerowego,
potem docker compose up żeby zbudować nowy, ustawić properties i uruchomić
```
### Google Colab
1. Go to: https://colab.research.google.com/github/piotermiarer/Sequence-Classification/blob/master/sequence-classification/jupiter_shared/usage_example.ipynb
2. Click 'COPY TO DRIVE'.
3. Click 'OPEN IN NEW TAB'.
4. Create new cell above and run this code in it:
```bash
!git clone https://github.com/piotermiarer/Sequence-Classification
!pip install -q mlxtend
%cd Sequence-Classification/sequence-classification/jupiter_shared
```
5. Run remaining cells. Don't run again the cell you've just added, because it would cause some errors.

# Authors
* Jakub Berezowski
* Magda Lipowska
* Piotr Miara
* Grzegorz Szczepaniak
* Maciej Piernik - supervisor

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
