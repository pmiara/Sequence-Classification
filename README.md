# Python module for comparing sequence classification algorithms.

# Getting Started
## Installing
### Docker
```
Plik requirements.txt zawiera niezbędne zależności potrzebne do poprawnego działania systemu.
Po instalacji Docker (https://www.docker.com) oraz docker-compose (https://docs.docker.com/compose) należu przejść do katalogu
głównego aplikacji. Następnie należy użyć polecenia docker-compose up. 

Po dodaniu innych bibliotek do pliku requirenments.txt należy przebudować obraz systemu. Można to osiągnąć usuwając go
docker image rm -f sequences, a następnie docker-compose up żeby zbudować nowy obraz, ustawić properties i uruchomić system.
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
