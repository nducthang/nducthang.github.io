---
title: "[LLM 02] Data cleaning and Tokenizations"
date: "2024-06-10"
tags: ["LLM"]
---
This article summarizes word knowledge from <cite>Large Language Models: A Survey [^1]</cite> . This is a series of articles continuing [LLM 01 - Large Language Model Families](../2024-05-01-llm-large-language-model-families/)

[^1]: Shervin Minaee et al. “Large Language Models: A Survey”. In: arXiv preprint arXiv:2402.06196 (2024).

# 1. Data Cleaning
Data quality is pivotal to the performance of language models. Effective data cleaning techniques, such as filtering and deduplication, can significantly enhance model performance. 
## 1.1 Data Filtering
Data filtering aims to enhance the quality of training data and improve the effectiveness of the trained language models. Common data filtering techniques include:

1. **Removing Noise**: This involves eliminating irrelevant or noisy data that might impair the model's ability to generalize. For instance, removing false information from the training data can reduce the likelihood of the model generating incorrect responses. Two mainstream approaches for quality filtering are classifier-based and heuristic-based frameworks.

2. **Handling Outliers**: Identifying and managing outliers or anomalies in the data to prevent them from disproportionately influencing the model. This ensures that the model learns from typical data patterns rather than skewed anomalies.

3. **Addressing Imbalances**: Balancing the distribution of classes or categories in the dataset to avoid biases and ensure fair representation. This is particularly important for responsible model training and evaluation.

4. **Text Preprocessing**: Cleaning and standardizing text data by removing stop words, punctuation, or other elements that may not contribute significantly to the model’s learning. This step ensures that the data is uniform and free of unnecessary noise.

5. **Dealing with Ambiguitie**s: Resolving or excluding ambiguous or contradictory data that might confuse the model during training. By clarifying these ambiguities, the model can provide more definite and reliable answers.

## 1.2 Deduplication
Deduplication refers to the process of removing duplicate instances or repeated occurrences of the same data in a dataset. Duplicate data points can introduce biases in the model training process and reduce data diversity, as the model may overfit on those particular instances.

## 1.3 Summary
In summary, data cleaning, which includes both filtering and deduplication, is essential for improving the performance of language models. By ensuring high-quality, diverse, and representative training data, we can train more effective and reliable models. As demonstrated by the Falcon40B example, rigorous data cleaning can lead to significant advancements in model performance, highlighting the importance of these techniques in the development of large language models.

# 2. Tokenizations
Tokenization refers to the process of converting a sequence of text into smaller parts known as tokens. While the simplest tokenization tool merely splits text based on white space, more advanced tokenization tools rely on a word dictionary. However, this approach faces the challenge of out-of-vocabulary (OOV) words, as the tokenizer can only recognize words within its dictionary. To mitigate this issue and improve dictionary coverage, popular tokenizers for large language models (LLMs) are based on sub-words. These sub-words can be combined to form a large number of words, including those not seen during training or those in different languages. Here, we will explore three widely used tokenization methods: Byte Pair Encoding, WordPiece Encoding, and SentencePiece Encoding.

## 2.1 Byte Pair Encoding (BPE)
Byte Pair Encoding is a algorithm that uses frequent patterns at byte level to compress data.
For example, Our corpus contain words:
```
low low low lower
```
So, we have base vocabulary is:
```
 ['l', 'o', 'w', 'e', 'r']
```
After getting this base vocabulary, we add new tokens until the desired vocabulary size is reached by learning merges, which are rules to merge two elements of the existing vocabulary together into a new one.
At any step during the tokenizer training, the BPE algorithm will search for the most frequent pair of existing tokens (by “pair,” here we mean two consecutive tokens in a word). That most frequent pair is the one that will be merged, and we rinse and repeat for the next step.
Iterations:
```
Vocabulary: ['l', 'o', 'w', 'e', 'r']
Corpus: [('l', 'o', 'w', 3), ('l', 'o', 'w', 'e', 'r', 1)]
```
Then we look at pairs. The pair ('l', 'o') is present in the word low and lower with total 4 times.
So, we have:
```
Vocabulary: ['l', 'o', 'w', 'e', 'r', 'lo']
Corpus: [('lo', 'w', 3), ('lo', 'w', 'e', 'r', 1)]
```
Then:
```
Vocabulary: ['l', 'o', 'w', 'e', 'r', 'lo', 'low']
Corpus: [('low', 3), ('low', 'e', 'r', 1)]
```
And we continue like this until we reach the desired vocabulary size.

Implement: [Detail](https://huggingface.co/learn/nlp-course/en/chapter6/5)

**Advantages**: BPE is effective in representing morphological variations of frequent words, as common prefixes and suffixes are well captured if they frequently appear in the training data.

**Application**: BPE is widely used in models where efficient vocabulary management and flexibility to handle new words are crucial.

## 2.2 Word Piece Encoding
WordPiece Encoding is primarily used in well-known models like BERT and Electra. It aims to address the issue of unknown tokens (UNK) by ensuring all characters in the training data are represented in the vocabulary.

WordPiece starts from a small vocabulary including the special tokens used by the model and the initial alphabet. Since it identifies subwords by adding a prefix (like ## for BERT), each word is initially split by adding that prefix to all the characters inside the word. So, for instance, "word" gets split like this:
```
w ##o ##r ##d
```

Thus, the initial alphabet contains all the characters present at the beginning of a word and the characters present inside a word preceded by the WordPiece prefix.
Then, again like BPE, WordPiece learns merge rules. The main difference is the way the pair to be merged is selected. Instead of selecting the most frequent pair, WordPiece computes a score for each pair, using the following formula:
$$score=(freq\_of\_pair)/(freq\_of\_first\_element \times freq\_of\_second\_element)$$

By dividing the frequency of the pair by the product of the frequencies of each of its parts, the algorithm prioritizes the merging of pairs where the individual parts are less frequent in the vocabulary.

Example: We have vocabulary and frequent of each words as follow:
```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```
The most frequent pair is ("##u", "##g") (present 20 times), but the individual frequency of "##u" is very high, so its score is not the highest (it’s 1 / 36). All pairs with a "##u" actually have that same score (1 / 36), so the best score goes to the pair ("##g", "##s") — the only one without a "##u" — at 1 / 20, and the first merge learned is ("##g", "##s") -> ("##gs").
```
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs"]
Corpus: ("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##gs", 5)
```
Iterative: ("h", "##u") -> "hu". 
```
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu"]
Corpus: ("hu" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("hu" "##gs", 5)
```
Then the next best score is shared by ("hu", "##g") and ("hu", "##gs") (with 1/15, compared to 1/21 for all the other pairs), so the first pair with the biggest score is merged:
```
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu", "hug"]
Corpus: ("hug", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("hu" "##gs", 5)
```
and we continue like this until we reach the desired vocabulary size.

Implement: [Detail](https://huggingface.co/learn/nlp-course/en/chapter6/6?fw=pt)

**Advantages**: This method reduces the occurrence of unknown tokens, enhancing the model’s ability to handle diverse and previously unseen inputs.

**Application**: WordPiece Encoding is essential for models that require high accuracy in token representation and robust handling of diverse inputs.
## 2.3 Unigram Encoding
The Unigram algorithm is often used in SentencePiece, which is the tokenization algorithm used by models like AlBERT, T5, mBART, Big Bird, and XLNet.

Unigram Encoding is a probabilistic model that treats each token as an independent unit, selecting tokens based on their probability of occurrence in the training data. Unlike other methods that build tokens by iteratively merging characters or sub-words, Unigram Encoding starts with a large vocabulary and iteratively prunes it to find the optimal set of tokens.

Example:
```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```
**Initial Vocabulary**: Unigram Encoding begins with a large vocabulary that includes all possible tokens derived from the training data. This vocabulary often consists of individual characters, sub-words, and entire words.
```
["h", "u", "g", "hu", "ug", "p", "pu", "n", "un", "b", "bu", "s", "hug", "gs", "ugs"]
```

**Probability Assignment**: Each token in the initial vocabulary is assigned a probability based on its frequency of occurrence in the training data. These probabilities help in evaluating the importance of each token.

For Example, Here are the frequencies of all the possible subwords in the vocabulary:
```
("h", 15) ("u", 36) ("g", 20) ("hu", 15) ("ug", 20) ("p", 17) ("pu", 17) ("n", 16)
("un", 16) ("b", 4) ("bu", 4) ("s", 5) ("hug", 15) ("gs", 5) ("ugs", 5)
```
So, the sum of all frequencies is 210, and the probability of the subword "ug" is thus 20/210.

**Iterative Pruning**: The algorithm iteratively removes tokens that contribute the least to the model's likelihood, recalculating probabilities at each step. This process continues until an optimal vocabulary size is achieved, balancing model complexity and performance.

For example, "pug" has the probability:
```
["p", "u", "g"] : 0.000389
["p", "ug"] : 0.0022676
["pu", "g"] : 0.0022676
```
So, "pug" would be tokenized as ["p", "ug"] or ["pu", "g"], depending on which of those segmentations is encountered first

Each word in the corpus has a score, and the loss is the negative log likelihood of those scores — that is, the sum for all the words in the corpus of all the -log(P(word)).

**Final Vocabulary**: The resulting vocabulary consists of tokens that maximize the likelihood of the training data. This vocabulary is used to tokenize new text inputs for the language model

Let’s go back to our example with the following corpus:
```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

The tokenization of each word with their respective scores is:
```
"hug": ["hug"] (score 0.071428)
"pug": ["pu", "g"] (score 0.007710)
"pun": ["pu", "n"] (score 0.006168)
"bun": ["bu", "n"] (score 0.001451)
"hugs": ["hug", "s"] (score 0.001701)
```
So the loss is:
```
10 * (-log(0.071428)) + 5 * (-log(0.007710)) + 12 * (-log(0.006168)) + 4 * (-log(0.001451)) + 5 * (-log(0.001701)) = 169.8
```

Now we need to compute how removing each token affects the loss. removing "hug" will make the loss worse, because the tokenization of "hug" and "hugs" will become:
```
"hug": ["hu", "g"] (score 0.006802)
"hugs": ["hu", "gs"] (score 0.001701)
```

These changes will cause the loss to rise by:
```
- 10 * (-log(0.071428)) + 10 * (-log(0.006802)) = 23.5
```

Therefore, the token "pu" will probably be removed from the vocabulary, but not "hug".

Implement: [Detail](https://huggingface.co/learn/nlp-course/en/chapter6/7?fw=pt)

Unigram Encoding is a powerful tokenization method that leverages probabilistic modeling to optimize the token vocabulary for language models. Its simplicity, efficiency, and flexibility make it a valuable tool in NLP, helping to create robust and versatile language models. By understanding and applying Unigram Encoding, we can enhance the performance and adaptability of our language processing systems.

## 2.4 SentencePiece Encoding
All tokenization algorithms described so far have the same problem: It is assumed that the input text uses spaces to separate words. However, not all languages use spaces to separate words. One possible solution is to use language specific pre-tokenizers, e.g. XLM uses a specific Chinese, Japanese, and Thai pre-tokenizer). To solve this problem more generally, SentencePiece treats the input as a raw input stream, thus including the space in the set of characters to use. It then uses the BPE or unigram algorithm to construct the appropriate vocabulary.

**Advantages**: This approach is highly versatile, making it suitable for languages with complex word boundaries and noisy text data.

**Application**: SentencePiece is particularly useful in multilingual models and scenarios where the input text includes noise, such as OCR outputs or social media text.
