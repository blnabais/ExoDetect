# ExoDetect — classificar exoplanetas do Kepler

Projeto individual do 1.º ano da licenciatura em Inteligência Artificial e
Ciência de Dados, Universidade da Beira Interior.

O telescópio Kepler observou milhares de estrelas à procura de pequenas quedas de
brilho causadas por planetas a passar à frente delas. A maior parte destes sinais
acaba por não ser um planeta. O objetivo aqui é treinar um modelo que, dado um
sinal, decida entre três classes: **planeta confirmado**, **candidato** ou
**falso positivo** — e perceber em que é que esse modelo se baseia para decidir.

Dados: tabela cumulativa de *Kepler Objects of Interest* do NASA Exoplanet
Archive (9 564 observações, 50 colunas).

## Resultados

Random Forest com 500 árvores, 26 *features*, divisão 80/20 estratificada,
`random_state=42`.

| Classe | Precision | Recall | F1 | Observações |
|---|---|---|---|---|
| CANDIDATE | 0,66 | 0,46 | 0,54 | 449 |
| CONFIRMED | 0,81 | 0,85 | 0,83 | 459 |
| FALSE POSITIVE | 0,83 | 0,92 | 0,87 | 1 005 |
| **Accuracy** | | | **79,30%** | 1 913 |

Matriz de confusão (linhas = classe real, colunas = classe prevista):

```
                 CAND  CONF   FP
CANDIDATE         205    72  172
CONFIRMED          45   391   23
FALSE POSITIVE     62    22  921
```

Como linha de base, responder sempre "FALSE POSITIVE" daria 52,5%, porque é a
classe maioritária. Os 79,30% ficam 27 pontos acima disso.

Os candidatos são a classe mais difícil (recall 0,46) e o erro mais frequente do
modelo é classificá-los como falsos positivos. Isto não me parece um defeito do
modelo: um candidato é, por definição, uma observação cuja evidência não foi
conclusiva, e as suas características sobrepõem-se às das outras duas classes.

## A parte que mais me ensinou: fuga de informação

Na primeira versão deste trabalho o modelo tinha 90,96% de accuracy. Estava a
usar, entre as *features*, a coluna `koi_score` e as quatro `koi_fpflag_*`.

Essas colunas não são medições do telescópio. São o **resultado do processo de
validação da NASA**: a `koi_score` é a confiança que a própria NASA atribuiu à
classificação, e as `koi_fpflag_*` assinalam os motivos pelos quais uma
observação foi marcada como falso positivo. Dá-las ao modelo é dar-lhe parte da
resposta — ele não aprende a reconhecer um planeta, aprende a copiar uma decisão
que já estava tomada.

Retirei-as e voltei a treinar, com o mesmo código e a mesma semente:

| | Accuracy | Recall CANDIDATE | Recall CONFIRMED | Recall FALSE POSITIVE |
|---|---|---|---|---|
| Com `koi_score` e `koi_fpflag_*` (31 features) | 90,96% | 0,80 | 0,85 | 0,99 |
| Sem essas colunas (26 features) | 79,30% | 0,46 | 0,85 | 0,92 |

Onze pontos de accuracy e metade do recall dos candidatos vinham daquelas cinco
colunas. O número mais alto era o número errado, e os 79,30% são o que o modelo
consegue fazer a partir das propriedades do sinal. O que me levou a desconfiar foi
o gráfico do SHAP: a `koi_score` aparecia como a *feature* mais importante, muito
à frente de tudo o resto.

## O que o SHAP mostra

Depois da correção, as três *features* mais influentes para a classe CONFIRMED
são `koi_steff_err1`, `koi_duration_err1` e `koi_model_snr`.

Eu esperava o sinal-ruído (`koi_model_snr`) em primeiro lugar, porque foi a
*feature* que melhor separava as classes nos histogramas da análise exploratória.
Aparece, mas em terceiro. As duas primeiras são **incertezas de medição** — o erro
na temperatura da estrela e o erro na duração do trânsito. Parte do que o modelo
aprendeu é portanto a qualidade com que cada observação foi medida, e não a física
do planeta. É uma limitação do trabalho e fica registada como tal.

O passo seguinte, se voltar a este projeto, é treinar só com as colunas de medição
física, deixando de fora todas as `_err1`, para medir quanto é que o resultado
depende da qualidade da medição.

## Como está organizado

```
projeto_ExoDetect/
├── ExoDetect.ipynb          análise, limpeza, treino e SHAP, por fases
├── cumulative.csv           dados do NASA Exoplanet Archive
└── relatorio_exodetect.tex  relatório escrito (LaTeX)
```

O notebook está dividido em cinco fases, cada uma com o que foi feito e porquê:
carregar e inspecionar os dados, análise exploratória, preparação dos dados,
treino e avaliação, explicabilidade com SHAP.

## Como correr

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap jupyter
jupyter notebook projeto_ExoDetect/ExoDetect.ipynb
```

Executar as células por ordem. O notebook lê o `cumulative.csv` do mesmo
diretório. O cálculo dos valores SHAP é a parte mais lenta e leva alguns minutos.

## Decisões que vale a pena justificar

- **Random Forest** — lida com colunas em escalas muito diferentes, resiste aos
  *outliers* que este dataset tem em abundância, e permite extrair a importância
  das *features*, que era necessária para a última fase.
- **Mediana e não média** para preencher valores em falta, porque as distribuições
  são muito assimétricas e a média seria arrastada pelos valores extremos.
- **`stratify=y`** na divisão treino/teste, para as três classes manterem a sua
  proporção nos dois conjuntos; sem isso, as métricas da classe CANDIDATE
  deixariam de ser fiáveis.
- **Normalização MinMax** — não é necessária para árvores de decisão. Fi-la para
  poder comparar os valores SHAP entre colunas medidas em unidades diferentes.
- **Remoção das colunas `_err2`** — são a margem de erro inferior da mesma medição
  que as `_err1`, ou seja, informação duplicada. Para várias colunas a correlação
  entre as duas é exatamente −1.

---

**Autor:** Bernardo Nabais · Universidade da Beira Interior · 2026
