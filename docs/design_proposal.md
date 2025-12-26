# Design Proposal

> [!IMPORTANT]  
> Temat: Sprawdzenie jakości działania metod atrybucji danych na danych spoza dystrybucji

<!-- TOC -->
* [Design Proposal](#design-proposal)
  * [Harmonogram prac](#harmonogram-prac)
  * [Zakres eksperymentów](#zakres-eksperymentów)
  * [Rezultaty / Planowana funckjonalność programu](#rezultaty--planowana-funckjonalność-programu)
  * [Stos technologiczny](#stos-technologiczny)
  * [Bibliografia](#bibliografia)
<!-- TOC -->


## Harmonogram prac

| Nr tygodnia | Data                                  | Cel do wykonania                                                                                                                                                |
|-------------|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1           | 30.10 - 05.11                         | Wstępna analiza literatury, doprecyzowanie celu projektu, przygotowanie design proposal, wstępny plan eksperymentów                                             |
| 2           | 06.11 - 12.11                         | Przygotowanie prototypu - przygotowanie danych, zestawienie potoku treningowego, test na wybranej metodzie atrybucji, konfiguracja środowiska eksperymentalnego |
| 3           | 13.11 - 19.11                         | Prezentacja prototypu, konsultacja planu eksprymentów                                                                                                           |
| 4           | 20.11 - 26.11 (kolokwium nr.1)        | Przerwa na naukę do kolokwium                                                                                                                                   |
| 5           | 27.11 - 03.12                         | Zatwierdzenie planu eksperymentów                                                                                                                               |
| 6           | 04.12 - 10.12                         | Przeprowadzenie eksperymentów 2 z 4 metod                                                                                                                       |
| 7           | 11.12 - 17.12                         | Sprawdzenie jakości atrybucji, weryfikacja wpływu rozbieżności w dystrybucji na zmiany w atrybucji                                                                                                                                |
| 8           | 18.12 - 24.12                         | Przeprowadzenie eksperymentów z pozostałymi metodami i sprawdzenie jakości ich wytłumaczeń                                                                      |
| 9           | 25.12 - 31.12                         | Przerwa świąteczna                                                                                                                                              |
| 10          | 01.01 - 07.01                         | Opisywanie przeprowadzonych eksperymentów i konsultacje                                                                                                         |
| 11          | 08.01 - 14.01                         | Ostatnie szlify: poprawianie jakości kodu, przeprowadzanie dodatkowych eksperymentów z użyciem transformera wizyjnego etc.                                                                        |
| 12          | 15.01 - 21.01 (kolokwium nr.2)        |                                                                                                                                                                 |
| 13          | 22.01 - 28.01 (prezentacja projektów) | Przygotowanie prezentacji                                                                                                                                       |                                                                                                                                                                                                                                                                     |

## Zakres eksperymentów
- **Główny cel**: Sprawdzenie działania różnych metod atrybucji danych i porównanie ich wyników na danych z i spoza dystrybucji


- **Zbiór danych**: https://metashift.readthedocs.io/en/latest/ - zbiór oparty o VisualGenom zawierający obrazki podzielone na 410 klas występujące w 1853 kontekstach. Na potrzeby ekperymentów planujemy wykorzystać podzbiór zaproponowany w artykule [Metashift](https://arxiv.org/pdf/2202.06523). Dla każdego zadania wybrane zostaną dwa różne zestawy danych treningowych - zbliżony do kontekstu testowego i bardzo od niego odległy (szczegóły opisano rozdziale):
  - **koty vs. psy** 

      Zbiór testowy - psy z półką

      Zbiór treningowy łatwy - koty z sofą, koty z łóżkiem, psy z szafą, psy z łóżkiem

      Zbiór treningowy trudny - koty z sofą, koty z łóżkiem, psy z łódką, psy z deską surfingową

  <br>
   
  - **ciężarówki vs. autobusy**
      
      Zbiór testowy - ciężarówki z samolotem

      Zbiór treningowy łatwy - autobusy z zegarem, autobusy ze światłami, ciężarówki z płotem, ciężarówki z pachołkiem

      Zbiór treningowy trudny - autobusy z zegarem, autobusy ze światłami, ciężarówki ze świałami ulicznymi, ciężarówki z psem

  <br>

  - **konie vs. słonie**

    Zbiór testowy - konie ze stodołą

    Zbiór treningowy łatwy - słonie z płotem, słonie ze skałą, konie z drzewem, konie z ziemią

    Zbiór treningowy trudny - słonie z płotem, słonie ze skałą, konie z pomnikiem, konie z wózkiem

  <br>
  
  - **miski vs. kubki**
 
    Zbiór testowy - kubki z kawą

    Zbiór treningowy łatwy - miska z owocami, miska z tacką, kubki z nożem, kubki z tacką

    Zbiór treningowy trudny - miska z owocami, miska z tacką, kubki z toaletą, kubki z pudełkiem
    
    <br>

    Należy zaznaczyć, że zbiory te **nie muszą być rozłączne**. Przy określaniu "bliskości" bazujemy na metodzie i przykładach zasugerowanych w ww. artykule. W każdym zadaniu autorzy używali jako zbioru testowego kontekstu niewystępującego w zbiorze treningowym co pozwala sprawdzić zachowanie modelu na danych spoza dystrybucji. 
    
    Poza powyższymi podziałami rozważamy również skorzystanie z podziału "wyższego poziomu" (general context) dostępnego w ramach tego samego zbioru danych (np. indoor/outdoor, sunny/rainy). Każdy z takich *general context* składa się z kilkunastu lub kiludziesięciu bardziej kontekstów bardziej szczegółowych.
    
    <br>

- **Porównywane metody**: 
  - Główne:
    - TracIn
    - TRAK
    - DualDA
    
  - Dodatkowo:
    - Influence Functions (w ramach możliwości sprzętowych)
    - Wartości Shapelya (na danych zgrupowanych, w ramach możliwości sprzętowych)

- **Eksperymenty**: 
  
  1. Wstępem do eksperymentów będzie wytrenowanie modelu **ResNet18** na różnych kombinacjach zbiorów treningowych jak opisano w rozdziale [Zbiór danych](#zakres-eksperymentów) a następnie wykonanie predykcji na zbiorze testowym dla danego zadania. Umożliwi to porównanie uzyskanych wyników modelu (accuracy) dla danych spoza dystrybucji z wartościami opublikowanymi w artykule.

  2. Kolejnym krokiem będzie wykorzystanie każdej z porównywanych metod w celu sprawdzenia atrybucji poszczególnych przykładów treningowych w końcowej predykcji. Dla każdego zestawu sprawdzone zostaną co najmniej 3 podstawowe metody - daje to 6 eksperymentów na zadanie.
  
  3. Uzyskane wyniki zostaną następnie zwizualizowane i zweryfikowane (zapewnie ręcznie, być może z wykorzystaniem LDS - do weryfikacji) i porównane.

  4. Ciekawym rozszerzeniem eksperymentów może być porówywanie atrybucji nie poszczególnych danych, a kontekstów (np. pies w salonie, pies przy kanapie) użytych w treningu w predykcji w nowym kontekście testowym. W szczególności ciekawe może być sprawdzenie korelacji między odległościami kontekstów wyliczonymi metodą autorów artykułu a wartościami uzyskanymi z metod atrybucji. W tym celu poza przytoczonymi wcześniej małymi przykładami z maksymalnie 4 konteksami przydatne mogą okazać się opisane wcześniej *general context* dostepne w zbiorze danych.

- **Proponowane modele**: ResNet18, Transformer wizyjny (opcjonalnie, jeżeli zostanie czas)

## Rezultaty / Planowana funckjonalność programu
- weryfikacja, czy metody atrybucji danych wskazują na spodziewane przykłady ze zbioru trenignowego podczas klasyfikacji danych spoza dystrybucji sklasyfikowanych właściwie przez model oraz czy występują znaczące różnice pomiędzy poszczególnymi metodami
- sprawdzenie, czy są obserwowalne różnice w działaniu metod atrybucji w zależności od wielkości różnic pomiędzy zbiorem treningowym i testowym (intuicja - w przypadku zbiorów zbliżonych model może koncentrować się na kilku podobnych przykładach, podczas gdy w zbiorach mocno różnych atrybucja rozłoży się bardziej równomiernie) 
- odpowiedź na pytanie czy metody atrybucji danych mogą stanowić wskazówkę umożliwiającą udoskonalenie procesu uczenia przy wykorzystaniu danych spoza dystrybucji sklasyfikowanych błędnie
- sprawdzenie, czy wyniki metod atrybucji danych korespondują z "odległością" pomiędzy danymi spoza dytsrybucji i danymi z dystrybucji wyliczoną innym metodami
- przygotowanie potoku przetwarzania danych/treningu i środowiska umożliwiającego powtórzenie i ewentualne rozwinięcie eksperymentów w przyszłości
- zaliczenie drugiego kolokwium :)

## Stos technologiczny
- uv
- ruff
- Pytorch
- Tensorboard / wandb.ai


## Bibliografia

| Artykuł                                                                                 | Autorski komentarz                                                                                                                                                                                                                                             | Kod                                        | Metryki                                                                                          |
|-----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|--------------------------------------------------------------------------------------------------|
| Metashift                                                                               | zawarty w ramach opisu zbioru danych                                                                                                                                                                                                                           | https://github.com/Weixin-Liang/MetaShift/ | M                                                                                                |
| Influence Functions                                                                     | paper zasugerowany na wykładzie, szczegółowy opis metody IF, kod do replikacji eksperymentów                                                                                                                                                                   | https://arxiv.org/abs/1703.04730           | Replikacja eksperymentów w artykule: http://bit.ly/gt-influence oraz  http://bit.ly/cl-influence |
| TracIn                                                                                  | paper zasugerowany na wykładzie, szczegółowy opis metody TracIn, kod do replikacji eksperymentów                                                                                                                                                               | https://arxiv.org/abs/2002.08484           | https://github.com/frederick0329/TracIn                                                          |
| TRAK                                                                                    | paper zasugerowany na wykładzie, szczegółowy opis metody TRAK, kod do replikacji eksperymentów                                                                                                                                                                 | https://arxiv.org/abs/2303.14186           | https://github.com/MadryLab/trak                                                                 |
| DualDA                                                                                  | paper zasugerowany na wykładzie, szczegółowy opis metody DualDA, kod do replikacji eksperymentów                                                                                                                                                               | https://arxiv.org/abs/2402.12118v2         | https://github.com/gumityolcu/DualXDA                                                            |
| Out-of-Distribution Generalization Analysis via Influence Function                      | Analiza wykorzystania IF do wykrywania danych spoza dystrybucji - do doczytania                                                                                                                                                                                | https://arxiv.org/abs/2101.08521           |                                                                                                  |
| "Why did the Model Fail?": Attributing Model Performance Changes to Distribution Shifts | Wykorzystanie metod Shapelya do wyjaśnienia spadku performancu modelu na danych spoza dystrybucji - do doczytania, ścisle związane z tematem                                                                                                                   | https://arxiv.org/abs/2210.10769           |                                                                                                  |
| LossVal                                                                                 | Wykorzystanie ważonej funkcji straty do identyfikacji przykładów, które można zaklasyfikować jako szum. Przykłady zawierają zbiór CIFAR10 więc jest potencjał do wykorzystania w naszym projekcie - do doczytania i potwierdzenia zgodności z tematem projektu | https://arxiv.org/pdf/2412.04158           | https://github.com/twibiral/LossVal                                                              | 

<!-- LIME, SHAP Bartek; GradCAM Marcin; Saliency Kajetan -->
<!-- | [LIME][]      | LIME                                 | https://github.com/marcotcr/lime                        |         |        |
| [SHAP][]      | SHAP                                 | https://github.com/shap/shap                            |         |        |
| [GradCAM][]   | GradCAM                              | https://github.com/jacobgil/pytorch-grad-cam            |         |        |
| [Saliency][]  | Saliency                             | https://github.com/sunnynevarekar/pytorch-saliency-maps |         |        | -->


<!-- [Metashift]: https://arxiv.org/abs/2202.06523
[LIME]: https://arxiv.org/abs/1602.04938
[SHAP]: https://dl.acm.org/doi/pdf/10.5555/3295222.3295230
[GradCAM]: https://arxiv.org/abs/1610.02391
[Saliency]: https://arxiv.org/abs/1312.6034 -->

