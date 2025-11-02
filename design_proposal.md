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
| 7           | 11.12 - 17.12                         | Sprawdzenie jakości -wytłumaczeń, weryfikacja wpływu rozbieżności w dystrybucji na jakość wytłumaczeń                                                                                                                                |
| 8           | 18.12 - 24.12                         | Przeprowadzenie eksperymentów z pozostałymi metodami i sprawdzenie jakości ich wytłumaczeń                                                                      |
| 9           | 25.12 - 31.12                         | Przerwa świąteczna                                                                                                                                              |
| 10          | 01.01 - 07.01                         | Opisywanie przeprowadzonych eksperymentów i konsultacje                                                                                                         |
| 11          | 08.01 - 14.01                         | Ostatnie szlify: poprawianie jakości kodu, przeprowadzanie dodatkowych eksperymentów z użyciem transformera wizyjnego etc.                                                                        |
| 12          | 15.01 - 21.01 (kolokwium nr.2)        |                                                                                                                                                                 |
| 13          | 22.01 - 28.01 (prezentacja projektów) | Przygotowanie prezentacji                                                                                                                                       |                                                                                                                                                                                                                                                                     |

## Zakres eksperymentów
- **Główny cel**: Sprawdzenie wyjaśnień generowanych przez metody atrybucji danych modelu i jakości ich działania na danych spoza dystrybucji

- **Zbiór danych**: https://metashift.readthedocs.io/en/latest/ - zbiór oparty o VisualGenom zawierający obrazki podzielone na 410 klas występujące w 1853 kontekstach. Na potrzeby ekperymentów planujemy wykorzystać podzbiór zaproponowany w artykule [Metashift][] tj. koty vs. psy, ciężarówki vs. autobusy, konie vs. słonie i miska vs. kubek w wybranych kontesktach dla zbioru treningowego i testowego. W każdym zadaniu autorzy używali jako zbioru testowego kontekstu niewystępującego w zbiorze treningowym co pozwala sprawdzić zachowanie modelu na danych spoza dystrybucji.

- **Porównywane metody**: 
    - LIME
    - GradCAM
    - Saliency 
    - SHAP

- **Eksperymenty**: Polegają na douczeniu modelu ResNet18 na przygotowanym zbiorze danych a następnie wykonaniu predykcji. Umożliwi to porównanie uzyskanych wyników modelu (accuracy) dla danych spoza dystrybucji z wartościami opublikowanymi w artykule.

 Kolejnym krokiem będzie wykorzystanie każdej z porównywanych metod w celu wygenerowania wytłumaczenia predykcji modelu na danych spoza dystrybucji (z innym kontekstem) tj. na co model zwraca uwagę, czy na podmiot zadania (pies lub kot) czy inne cechy obrazów. Dla każdego zadania wybrane zostaną dwa różne zestawy danych treningowych - zbliżony do kontekstu testowego i bardzo od niego odległy <!-- TODO: wg jakiej metryki -->, a następnie dla każdego zestawu sprawdzone zostaną wszystkie 4 metody - daje to 8 eksperymentów na zadanie. Taki dobór eksperymntów pozwoli odpowiedzieć na pytanie, o użyteczność poszczególnych metod atrybucji w kontekście zróżnicowanego odstępstwa od dystrybucji treningowej.

- **Proponowane modele**: ResNet18, Transformer wizyjny (opcjonalnie, jeżeli zostanie czas)

## Rezultaty / Planowana funckjonalność programu
- weryfikacja, czy metody atrybucji danych wskazują na oczekiwane regiony na danych spoza dystrybucji sklasyfikowanych właściwie przez model 
- odpowiedź na pytanie jak metody atrybucji danych radzą sobię w wyjaśnianiu błędów modeli na danych spoza dystrybucji i czy mogą stanowić wskazówkę umożliwiającą udoskonalenie procesu uczenia
- przygotowanie potoku przetwarzania danych/treningu i środowiska umożliwiającego powtórzenie i ewentaulne rozwinięcie eksperymentów w przyszłości
- zaliczenie drugiego kolokwium :)

## Stos technologiczny
- uv
- ruff
- Pytorch
- LIME
- SHAP
- Tensorboard / wandb.ai


## Bibliografia

| Artykuł       | Autorski komentarz                   | Kod                                                     | Metryki | Modele |
|---------------|--------------------------------------|---------------------------------------------------------|---------|--------|
| [Metashift][] | zawarty w ramach opisu eksperymentów | https://github.com/Weixin-Liang/MetaShift/              | M       | X      |
| [LIME][]      | LIME                                 | https://github.com/marcotcr/lime                        |         |        |
| [SHAP][]      | SHAP                                 | https://github.com/shap/shap                            |         |        |
| [GradCAM][]   | GradCAM                              | https://github.com/jacobgil/pytorch-grad-cam            |         |        |
| [Saliency][]  | Saliency                             | https://github.com/sunnynevarekar/pytorch-saliency-maps |         |        |

<!-- LIME, SHAP Bartek; GradCAM Marcin; Saliency Kajetan -->

[Metashift]: https://arxiv.org/abs/2202.06523
[LIME]: https://arxiv.org/abs/1602.04938
[SHAP]: https://dl.acm.org/doi/pdf/10.5555/3295222.3295230
[GradCAM]: https://arxiv.org/abs/1610.02391
[Saliency]: https://arxiv.org/abs/1312.6034
