# **NLP-w2v**

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMTYuNjQ5OTk5OTk5OTk5OTgiIGhlaWdodD0iMzUiIHZpZXdCb3g9IjAgMCAyMTYuNjQ5OTk5OTk5OTk5OTggMzUiPjxyZWN0IGNsYXNzPSJzdmdfX3JlY3QiIHg9IjAiIHk9IjAiIHdpZHRoPSIxMjYuMjMiIGhlaWdodD0iMzUiIGZpbGw9IiMzMUM0RjMiLz48cmVjdCBjbGFzcz0ic3ZnX19yZWN0IiB4PSIxMjQuMjMiIHk9IjAiIHdpZHRoPSI5Mi40MTk5OTk5OTk5OTk5OSIgaGVpZ2h0PSIzNSIgZmlsbD0iIzM4OUFENSIvPjxwYXRoIGNsYXNzPSJzdmdfX3RleHQiIGQ9Ik0xNy4zMyAyMkwxNC4yMiAyMkwxNC4yMiAxMy40N0wxNy4xNCAxMy40N1ExOC41OSAxMy40NyAxOS4zNCAxNC4wNVEyMC4xMCAxNC42MyAyMC4xMCAxNS43OEwyMC4xMCAxNS43OFEyMC4xMCAxNi4zNiAxOS43OCAxNi44M1ExOS40NyAxNy4zMCAxOC44NiAxNy41NkwxOC44NiAxNy41NlExOS41NSAxNy43NSAxOS45MyAxOC4yNlEyMC4zMSAxOC43OCAyMC4zMSAxOS41MUwyMC4zMSAxOS41MVEyMC4zMSAyMC43MSAxOS41MyAyMS4zNlExOC43NiAyMiAxNy4zMyAyMkwxNy4zMyAyMlpNMTUuNzAgMTguMTVMMTUuNzAgMjAuODJMMTcuMzUgMjAuODJRMTguMDQgMjAuODIgMTguNDQgMjAuNDdRMTguODMgMjAuMTMgMTguODMgMTkuNTFMMTguODMgMTkuNTFRMTguODMgMTguMTggMTcuNDcgMTguMTVMMTcuNDcgMTguMTVMMTUuNzAgMTguMTVaTTE1LjcwIDE0LjY2TDE1LjcwIDE3LjA2TDE3LjE1IDE3LjA2UTE3Ljg0IDE3LjA2IDE4LjIzIDE2Ljc1UTE4LjYyIDE2LjQzIDE4LjYyIDE1Ljg2TDE4LjYyIDE1Ljg2UTE4LjYyIDE1LjIzIDE4LjI2IDE0Ljk1UTE3LjkwIDE0LjY2IDE3LjE0IDE0LjY2TDE3LjE0IDE0LjY2TDE1LjcwIDE0LjY2Wk0yNC42NCAxOS4xNkwyNC42NCAxOS4xNkwyNC42NCAxMy40N0wyNi4xMiAxMy40N0wyNi4xMiAxOS4xOFEyNi4xMiAyMC4wMyAyNi41NSAyMC40OFEyNi45OCAyMC45MyAyNy44MyAyMC45M0wyNy44MyAyMC45M1EyOS41NCAyMC45MyAyOS41NCAxOS4xM0wyOS41NCAxOS4xM0wyOS41NCAxMy40N0wzMS4wMiAxMy40N0wzMS4wMiAxOS4xN1EzMS4wMiAyMC41MyAzMC4xNSAyMS4zMlEyOS4yOCAyMi4xMiAyNy44MyAyMi4xMkwyNy44MyAyMi4xMlEyNi4zNiAyMi4xMiAyNS41MCAyMS4zM1EyNC42NCAyMC41NSAyNC42NCAxOS4xNlpNMzcuMTUgMjJMMzUuNjcgMjJMMzUuNjcgMTMuNDdMMzcuMTUgMTMuNDdMMzcuMTUgMjJaTTQ3LjMyIDIyTDQxLjk2IDIyTDQxLjk2IDEzLjQ3TDQzLjQ0IDEzLjQ3TDQzLjQ0IDIwLjgyTDQ3LjMyIDIwLjgyTDQ3LjMyIDIyWk01My4xMiAxNC42Nkw1MC40OSAxNC42Nkw1MC40OSAxMy40N0w1Ny4yNSAxMy40N0w1Ny4yNSAxNC42Nkw1NC41OSAxNC42Nkw1NC41OSAyMkw1My4xMiAyMkw1My4xMiAxNC42NlpNNjYuODUgMTkuMTZMNjYuODUgMTkuMTZMNjYuODUgMTMuNDdMNjguMzMgMTMuNDdMNjguMzMgMTkuMThRNjguMzMgMjAuMDMgNjguNzYgMjAuNDhRNjkuMjAgMjAuOTMgNzAuMDQgMjAuOTNMNzAuMDQgMjAuOTNRNzEuNzUgMjAuOTMgNzEuNzUgMTkuMTNMNzEuNzUgMTkuMTNMNzEuNzUgMTMuNDdMNzMuMjMgMTMuNDdMNzMuMjMgMTkuMTdRNzMuMjMgMjAuNTMgNzIuMzYgMjEuMzJRNzEuNDkgMjIuMTIgNzAuMDQgMjIuMTJMNzAuMDQgMjIuMTJRNjguNTcgMjIuMTIgNjcuNzEgMjEuMzNRNjYuODUgMjAuNTUgNjYuODUgMTkuMTZaTTc3LjM2IDE5LjQyTDc3LjM2IDE5LjQyTDc4Ljg1IDE5LjQyUTc4Ljg1IDIwLjE1IDc5LjMzIDIwLjU1UTc5LjgxIDIwLjk1IDgwLjcwIDIwLjk1TDgwLjcwIDIwLjk1UTgxLjQ4IDIwLjk1IDgxLjg3IDIwLjYzUTgyLjI2IDIwLjMyIDgyLjI2IDE5LjgwTDgyLjI2IDE5LjgwUTgyLjI2IDE5LjI0IDgxLjg2IDE4Ljk0UTgxLjQ3IDE4LjYzIDgwLjQzIDE4LjMyUTc5LjQwIDE4LjAxIDc4Ljc5IDE3LjYzTDc4Ljc5IDE3LjYzUTc3LjYzIDE2LjkwIDc3LjYzIDE1LjcyTDc3LjYzIDE1LjcyUTc3LjYzIDE0LjY5IDc4LjQ3IDE0LjAyUTc5LjMxIDEzLjM1IDgwLjY1IDEzLjM1TDgwLjY1IDEzLjM1UTgxLjU0IDEzLjM1IDgyLjI0IDEzLjY4UTgyLjk0IDE0LjAxIDgzLjMzIDE0LjYxUTgzLjczIDE1LjIyIDgzLjczIDE1Ljk2TDgzLjczIDE1Ljk2TDgyLjI2IDE1Ljk2UTgyLjI2IDE1LjI5IDgxLjg0IDE0LjkxUTgxLjQyIDE0LjU0IDgwLjY0IDE0LjU0TDgwLjY0IDE0LjU0UTc5LjkxIDE0LjU0IDc5LjUxIDE0Ljg1UTc5LjExIDE1LjE2IDc5LjExIDE1LjcxTDc5LjExIDE1LjcxUTc5LjExIDE2LjE4IDc5LjU0IDE2LjUwUTc5Ljk4IDE2LjgxIDgwLjk3IDE3LjEwUTgxLjk3IDE3LjQwIDgyLjU3IDE3Ljc4UTgzLjE4IDE4LjE2IDgzLjQ2IDE4LjY1UTgzLjc0IDE5LjEzIDgzLjc0IDE5Ljc5TDgzLjc0IDE5Ljc5UTgzLjc0IDIwLjg2IDgyLjkyIDIxLjQ5UTgyLjEwIDIyLjEyIDgwLjcwIDIyLjEyTDgwLjcwIDIyLjEyUTc5Ljc4IDIyLjEyIDc5LjAwIDIxLjc3UTc4LjIyIDIxLjQzIDc3Ljc5IDIwLjgzUTc3LjM2IDIwLjIyIDc3LjM2IDE5LjQyWk04OS41OSAyMkw4OC4xMiAyMkw4OC4xMiAxMy40N0w4OS41OSAxMy40N0w4OS41OSAyMlpNOTUuODkgMjJMOTQuNDAgMjJMOTQuNDAgMTMuNDdMOTUuODkgMTMuNDdMOTkuNzAgMTkuNTRMOTkuNzAgMTMuNDdMMTAxLjE3IDEzLjQ3TDEwMS4xNyAyMkw5OS42OSAyMkw5NS44OSAxNS45NUw5NS44OSAyMlpNMTA1LjY2IDE4LjEzTDEwNS42NiAxOC4xM0wxMDUuNjYgMTcuNDZRMTA1LjY2IDE1LjUzIDEwNi41OSAxNC40NFExMDcuNTIgMTMuMzUgMTA5LjE3IDEzLjM1TDEwOS4xNyAxMy4zNVExMTAuNjAgMTMuMzUgMTExLjQ0IDE0LjA1UTExMi4yNyAxNC43NiAxMTIuNDQgMTYuMDhMMTEyLjQ0IDE2LjA4TDExMC45OSAxNi4wOFExMTAuNzQgMTQuNTQgMTA5LjIwIDE0LjU0TDEwOS4yMCAxNC41NFExMDguMjEgMTQuNTQgMTA3LjY5IDE1LjI2UTEwNy4xNyAxNS45OCAxMDcuMTUgMTcuMzdMMTA3LjE1IDE3LjM3TDEwNy4xNSAxOC4wMlExMDcuMTUgMTkuNDAgMTA3Ljc0IDIwLjE3UTEwOC4zMiAyMC45MyAxMDkuMzYgMjAuOTNMMTA5LjM2IDIwLjkzUTExMC40OSAyMC45MyAxMTAuOTcgMjAuNDJMMTEwLjk3IDIwLjQyTDExMC45NyAxOC43NUwxMDkuMjIgMTguNzVMMTA5LjIyIDE3LjYyTDExMi40NSAxNy42MkwxMTIuNDUgMjAuODlRMTExLjk5IDIxLjUwIDExMS4xNyAyMS44MVExMTAuMzUgMjIuMTIgMTA5LjMwIDIyLjEyTDEwOS4zMCAyMi4xMlExMDguMjMgMjIuMTIgMTA3LjQwIDIxLjYzUTEwNi41OCAyMS4xNCAxMDYuMTMgMjAuMjRRMTA1LjY4IDE5LjMzIDEwNS42NiAxOC4xM1oiIGZpbGw9IiNGRkZGRkYiLz48cGF0aCBjbGFzcz0ic3ZnX190ZXh0IiBkPSJNMTM3Ljk5IDE3LjgwTDEzNy45OSAxNy44MFExMzcuOTkgMTYuNTQgMTM4LjU5IDE1LjU0UTEzOS4xOSAxNC41NSAxNDAuMjUgMTMuOTlRMTQxLjMyIDEzLjQzIDE0Mi42NyAxMy40M0wxNDIuNjcgMTMuNDNRMTQzLjg0IDEzLjQzIDE0NC43OCAxMy44M1ExNDUuNzIgMTQuMjIgMTQ2LjM0IDE0Ljk3TDE0Ni4zNCAxNC45N0wxNDQuODMgMTYuMzNRMTQzLjk4IDE1LjQwIDE0Mi44MSAxNS40MEwxNDIuODEgMTUuNDBRMTQyLjc5IDE1LjQwIDE0Mi43OSAxNS40MEwxNDIuNzkgMTUuNDBRMTQxLjcxIDE1LjQwIDE0MS4wNSAxNi4wNlExNDAuMzkgMTYuNzEgMTQwLjM5IDE3LjgwTDE0MC4zOSAxNy44MFExNDAuMzkgMTguNTAgMTQwLjY5IDE5LjA0UTE0MC45OSAxOS41OSAxNDEuNTMgMTkuODlRMTQyLjA3IDIwLjIwIDE0Mi43NyAyMC4yMEwxNDIuNzcgMjAuMjBRMTQzLjQ1IDIwLjIwIDE0NC4wNSAxOS45M0wxNDQuMDUgMTkuOTNMMTQ0LjA1IDE3LjYyTDE0Ni4xNSAxNy42MkwxNDYuMTUgMjEuMTBRMTQ1LjQzIDIxLjYxIDE0NC40OSAyMS44OVExNDMuNTYgMjIuMTcgMTQyLjYyIDIyLjE3TDE0Mi42MiAyMi4xN1ExNDEuMzAgMjIuMTcgMTQwLjI0IDIxLjYxUTEzOS4xOSAyMS4wNSAxMzguNTkgMjAuMDVRMTM3Ljk5IDE5LjA2IDEzNy45OSAxNy44MFpNMTU3Ljg4IDIyTDE1MS4xNCAyMkwxNTEuMTQgMTMuNjBMMTU3LjczIDEzLjYwTDE1Ny43MyAxNS40NEwxNTMuNDkgMTUuNDRMMTUzLjQ5IDE2Ljg1TDE1Ny4yMyAxNi44NUwxNTcuMjMgMTguNjNMMTUzLjQ5IDE4LjYzTDE1My40OSAyMC4xN0wxNTcuODggMjAuMTdMMTU3Ljg4IDIyWk0xNjUuMDIgMjJMMTYyLjY5IDIyTDE2Mi42OSAxMy42MEwxNjQuNjQgMTMuNjBMMTY4LjM1IDE4LjA3TDE2OC4zNSAxMy42MEwxNzAuNjggMTMuNjBMMTcwLjY4IDIyTDE2OC43MyAyMkwxNjUuMDIgMTcuNTJMMTY1LjAyIDIyWk0xNzUuMjYgMjEuMjRMMTc1LjI2IDIxLjI0TDE3Ni4wNCAxOS40OVExNzYuNjAgMTkuODYgMTc3LjM0IDIwLjA5UTE3OC4wOSAyMC4zMiAxNzguODEgMjAuMzJMMTc4LjgxIDIwLjMyUTE4MC4xNyAyMC4zMiAxODAuMTggMTkuNjRMMTgwLjE4IDE5LjY0UTE4MC4xOCAxOS4yOCAxNzkuNzkgMTkuMTFRMTc5LjQwIDE4LjkzIDE3OC41MyAxOC43NEwxNzguNTMgMTguNzRRMTc3LjU4IDE4LjUzIDE3Ni45NSAxOC4zMFExNzYuMzEgMTguMDYgMTc1Ljg2IDE3LjU1UTE3NS40MCAxNy4wMyAxNzUuNDAgMTYuMTZMMTc1LjQwIDE2LjE2UTE3NS40MCAxNS4zOSAxNzUuODIgMTQuNzdRMTc2LjI0IDE0LjE1IDE3Ny4wOCAxMy43OVExNzcuOTEgMTMuNDMgMTc5LjEyIDEzLjQzTDE3OS4xMiAxMy40M1ExNzkuOTQgMTMuNDMgMTgwLjc1IDEzLjYyUTE4MS41NSAxMy44MCAxODIuMTcgMTQuMTdMMTgyLjE3IDE0LjE3TDE4MS40NCAxNS45M1ExODAuMjQgMTUuMjggMTc5LjExIDE1LjI4TDE3OS4xMSAxNS4yOFExNzguNDAgMTUuMjggMTc4LjA3IDE1LjQ5UTE3Ny43NSAxNS43MCAxNzcuNzUgMTYuMDRMMTc3Ljc1IDE2LjA0UTE3Ny43NSAxNi4zNyAxNzguMTQgMTYuNTRRMTc4LjUyIDE2LjcxIDE3OS4zNyAxNi44OUwxNzkuMzcgMTYuODlRMTgwLjMzIDE3LjEwIDE4MC45NiAxNy4zM1ExODEuNTkgMTcuNTYgMTgyLjA1IDE4LjA3UTE4Mi41MiAxOC41OCAxODIuNTIgMTkuNDZMMTgyLjUyIDE5LjQ2UTE4Mi41MiAyMC4yMSAxODIuMTAgMjAuODNRMTgxLjY4IDIxLjQ0IDE4MC44NCAyMS44MFExODAuMDAgMjIuMTcgMTc4Ljc5IDIyLjE3TDE3OC43OSAyMi4xN1ExNzcuNzggMjIuMTcgMTc2LjgxIDIxLjkyUTE3NS44NSAyMS42NyAxNzUuMjYgMjEuMjRaTTE4OS40NyAyMkwxODcuMDkgMjJMMTg3LjA5IDEzLjYwTDE4OS40NyAxMy42MEwxODkuNDcgMjJaTTE5Ni44NCAyMkwxOTQuNjQgMjJMMTk0LjY0IDEzLjYwTDE5Ni42MCAxMy42MEwxOTkuNTUgMTguNDVMMjAyLjQ0IDEzLjYwTDIwNC4zOSAxMy42MEwyMDQuNDEgMjJMMjAyLjIzIDIyTDIwMi4yMSAxNy41NUwyMDAuMDUgMjEuMTdMMTk4Ljk5IDIxLjE3TDE5Ni44NCAxNy42N0wxOTYuODQgMjJaIiBmaWxsPSIjRkZGRkZGIiB4PSIxMzcuMjMwMDAwMDAwMDAwMDIiLz48L3N2Zz4=)](https://forthebadge.com)

-----
## ***Objectives:***
This repository intends to familiarise you with the above techniques by implementing one of the **frequency-based modelling approaches** and comparing it to embeddings acquired using one of the word2vec variations. You'll start by attempting to obtain embeddings using the **Singular Value Decomposition (SVD)** method with a corpora given. The **CBOW implementation** of word2vec with **Negative Sampling** would then be used. Following that, a brief analysis would be conducted, showing the differences in the quality of the obtained embeddings.

-----
## ***File Structure:***
1. `preprocess.py` contains the code used for preprocessing the text and separating them based on **spaces**.
2. `read_json.py` contains the code to read limited number of sentences from the large dataset(about 500MB) and store them after doing the necessities in `derived_data.txt`.
3. `svd.py` implements the co-occurence matrix and SVD. It generates `cooccurence_matrix.csv` and `U_reduced.csv` which are important components of the concept. 
4. `model1.py` Implements the first model given in the document. It displays the most similar words for the word 'camera' and generates t-sne graphs for 5 different grammatical words.
5. `model2.py` Implements the second model given in the document. It displays the most similar words for the word 'camera' and generates t-sne graphs for 5 different grammatical words.
6. `pretrained_model.py` runs the code on a pretrained word2vec model. It is used to compare with the 2 created models above. 
-----

## ***Execution:***

The code can be fully executed in the following manner:
```py
python3 <filename>.py
```
The order in which the files are run is:

1. `read_json.py`  
2. `svd.py`     
3. `model1.py`  
4. `model2.py`  
5. `pretrained_model.py`

-----






