# ExoDetect — Deteção automatizada de exoplanetas com Machine Learning

**Um classificador Random Forest treinado com dados reais do telescópio Kepler (NASA) que separa exoplanetas confirmados, candidatos e falsos positivos — e explica cada decisão com SHAP.**

![Python](https://img.shields.io/badge/Python-3-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange)
![SHAP](https://img.shields.io/badge/explicabilidade-SHAP-purple)

O Kepler observou milhares de estrelas à procura de quedas de brilho causadas por planetas em trânsito. A maioria dos sinais não é um planeta, e classificá-los à mão não escala. Este projeto automatiza essa triagem e mostra **porque** o modelo decide como decide.

## Resultados

9 564 observações · 50 colunas originais, 31 usadas no treino · divisão 80/20 estratificada · 500 árvores · `random_state=42`

| Classe | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| CANDIDATE | 0.83 | 0.80 | 0.81 | 449 |
| CONFIRMED | 0.83 | 0.85 | 0.84 | 459 |
| FALSE POSITIVE | 0.98 | 0.99 | 0.98 | 1 005 |
| **Accuracy** | | | **90.96%** | 1 913 |

### Nota metodológica — fuga de informação

O treino inclui `koi_score` e as quatro colunas `koi_fpflag_*`. Estas são saídas do próprio processo de validação do Kepler, ou seja, estão muito próximas da resposta que o modelo tenta prever. Ao repetir o treino sem elas (mesmo pipeline, mesma semente):

| | Accuracy | Recall CANDIDATE | Recall CONFIRMED | Recall FALSE POSITIVE |
|---|---|---|---|---|
| Como no notebook (31 colunas) | 90.96% | 0.80 | 0.85 | 0.99 |
| Sem `koi_score` e `koi_fpflag_*` (26 colunas) | 79.30% | 0.46 | 0.85 | 0.92 |

O número mais honesto sobre o que o modelo aprende só a partir das propriedades físicas do sinal é o segundo. Os candidatos continuam a ser a classe mais difícil, o que faz sentido: são, por definição, os casos ambíguos.

## O que o projeto faz, por fases

1. **Configuração e inspeção** — carregamento do dataset, tipos de dados, escolha da coluna alvo (`koi_disposition`).
2. **Análise exploratória** — desequilíbrio de classes, histogramas das características físicas, matriz de correlação.
3. **Pré-processamento** — remoção de identificadores e de colunas `_err2` (espelhos das `_err1`, correlação −1), imputação de nulos pela mediana, normalização MinMax.
4. **Treino e avaliação** — Random Forest, precision/recall/F1 por classe, matriz de confusão.
5. **Explicabilidade** — SHAP global (que características pesam mais) e local (porque uma estrela concreta foi classificada assim).

## Estrutura

```text
ExoDetect/
└── projeto_ExoDetect/
    ├── ExoDetect.ipynb          # Análise, treino e SHAP
    ├── cumulative.csv           # Kepler Exoplanet Search Results (NASA)
    └── relatorio_exodetect.tex  # Relatório completo (LaTeX)
```

## Como correr

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap jupyter
```

Abrir `projeto_ExoDetect/ExoDetect.ipynb` e executar todas as células por ordem. O notebook lê `cumulative.csv` do mesmo diretório.

## Dados

*Kepler Exoplanet Search Results* (NASA Exoplanet Archive), tabela cumulativa de Kepler Objects of Interest.

## Contexto

Projeto individual do 1.º ano da licenciatura em Inteligência Artificial e Ciência de Dados, Universidade da Beira Interior (abril de 2026).

**Autor:** Bernardo Nabais
