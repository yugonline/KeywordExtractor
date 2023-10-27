## Kaggle Program Entry
Hello all,

This is my submission #1 to the Kaggle Mentorship Program. It is a keyword Extractor.

It is a direct implementation of the research paper : https://arxiv.org/pdf/2106.04939v1.pdf 

The idea is to get both Graph-based embeddings and Text based embeddings for a given text (various datasets were used, the code includes the best implementation of all the different combinations) 

## My Failures

Unfortunately, I did not get a good enough accuracy to be a full reference implementation of the paper. While in the paper the authors have gotten a best F1 score of 0.71; on the other hand my F1 score was around the 0.5 mark. 

It seems that there could be some internal step that is missing

## Possible Improvements

- I will be changing the BERT Token embedding to generate an embedding on the entire dataset instead right now I am doing it per abstract
- I will be realigning the Graph embeddings, it seems that Node2Vec even in the author's implementation gives lower accuracy than ExEm Graph embeddings
- My Neural Network skills are still improving. I am still completing the Coursera Certification on Neural Network and hope to get better at it as time goes by
