# Attributing Fair Decisions with Attention Interventions


**Requirements:**

tensorflow >= 1.12.0

numpy == 1.19.4

pandas == 1.1.5

scikit-learn == 0.23.2

scipy == 1.5.4

_____________________________________________

**To run the code for Adult:**

go to the corresponding adult folder and run:

```bash
python adult.py seed decade-rate
```

where seed is the random seed (int)


decade-rate is the rate for which you want to decrease the attention weights (float)

_____________________________________________

**To run the code for Health:**

go to the corresponding health folder and run:

```bash
python health.py seed decade-rate
```

where seed is the random seed (int)

decade-rate is the rate for which you want to decrease the attention weights (float)
______________________________________________
Seeds tried for the 5 runs: 2019, 2021, 0, 100, 5000

_____________________________________________

**Referenced paper link:**

https://aclanthology.org/2022.trustnlp-1.2.pdf

_____________________________________________

**Citation**

```
cite
@inproceedings{mehrabi-etal-2022-attributing,
    title = "Attributing Fair Decisions with Attention Interventions",
    author = "Mehrabi, Ninareh  and
      Gupta, Umang  and
      Morstatter, Fred  and
      Steeg, Greg Ver  and
      Galstyan, Aram",
    booktitle = "Proceedings of the 2nd Workshop on Trustworthy Natural Language Processing (TrustNLP 2022)",
    month = jul,
    year = "2022",
    address = "Seattle, U.S.A.",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.trustnlp-1.2",
    doi = "10.18653/v1/2022.trustnlp-1.2",
    pages = "12--25",
    abstract = "The widespread use of Artificial Intelligence (AI) in consequential domains, such as health-care and parole decision-making systems, has drawn intense scrutiny on the fairness of these methods. However, ensuring fairness is often insufficient as the rationale for a contentious decision needs to be audited, understood, and defended. We propose that the attention mechanism can be used to ensure fair outcomes while simultaneously providing feature attributions to account for how a decision was made. Toward this goal, we design an attention-based model that can be leveraged as an attribution framework. It can identify features responsible for both performance and fairness of the model through attention interventions and attention weight manipulation. Using this attribution framework, we then design a post-processing bias mitigation strategy and compare it with a suite of baselines. We demonstrate the versatility of our approach by conducting experiments on two distinct data types, tabular and textual.",
}
```
