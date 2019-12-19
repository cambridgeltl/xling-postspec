# xling-postspec
Cross-lingual Semantic Specialization via Lexical Relation Induction

Edoardo Maria Ponti, Ivan Vulić, Goran Glavaš, Roi Reichart, and Anna Korhonen. 2019. **Cross-lingual Semantic Specialization via Lexical Relation Induction**. In Proceedings of EMNLP 2019.
[[pdf]](https://www.aclweb.org/anthology/D19-1226.pdf)

If you use this software for academic research, please cite the paper in question:
```
@inproceedings{ponti2019cross,
  title={Cross-lingual Semantic Specialization via Lexical Relation Induction},
  author={Ponti, Edoardo Maria and Vuli{\'c}, Ivan and Glava{\v{s}}, Goran and Reichart, Roi and Korhonen, Anna},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2206--2217},
  year={2019}
}
```

## Nearest neighbours

Run the script `get_nn.py' to generate a word translation of the lexical constraints from English to a target language. The script requires source language (attract / repel) constraints and aligned word embeddings in the two languages.

## Full pipeline

Run the script `src/main.py' to train and evaluate the full pipeline consisting of 1) refinement of the raw constraints based on cross-lingual lexical relation classification; 2) Attract-Repel in the target language; 3) post-specialization in the target language.

## Embeddings and constraints

Coming soon.

## Acknowledgements

Part of the code has been borrowed from the GAN implementation in [MUSE](https://github.com/facebookresearch/MUSE), with some changes. The link contains a copy of the original license.
