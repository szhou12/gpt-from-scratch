# gpt-from-scratch
Self-teaching GPT and related topics from Andrej Karpathy's [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy).

## bigram_v2 -> bigram_v3
1. At the beginning in `BigramLanguageModel` class, we vectorize tokens into vectors by embedding. Later in `Head` class, we transform embedding vectors into key, query and value vectors by linear transformation. Can you explain why use embedding technique first but linear transformation later? Why not use same technique for both stages? Embedding is essentially a vectorization technique, then how does embedding differ from encoding (e.g. one-hot encoding)?
    1. `nn.Embedding` is `nn.Linear`. Embedding allows indexing, which enhances computing efficiency.
        - [Understanding the Difference Between Embedding Layers and Linear Layers](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/03_bonus_embedding-vs-matmul)
    2. Embedding vector: Dense. Contain correlation & similarities between vectors. One-hot Encoding vector: Sparse. No correlation & similarities between vectors.
        - [Understanding Differences Between Encoding and Embedding](https://www.linkedin.com/pulse/understanding-differences-between-encoding-embedding-mba-ms-phd/)
        - [自然语言处理中的'token', 'embedding'与'encoding'：基本概念与差异](https://cloud.baidu.com/article/1887285)
        - [position embedding和position encoding是什么有什么区别](https://docs.pingcode.com/ask/46926.html)
        - [Master Positional Encoding: Part I](https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3)
        - [Master Positional Encoding: Part II](https://medium.com/towards-data-science/master-positional-encoding-part-ii-1cfc4d3e7375)



## Resources
- [GitHub repo for the Andrej's video](https://github.com/karpathy/ng-video-lecture)
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy)
- [Andrej Karpathy《从零开始搭建GPT|Let's build GPT from scratch, in code, spelled out》中英字](https://www.bilibili.com/video/BV1v4421c7fr/?spm_id_from=333.337.search-card.all.click&vd_source=0c02ef6f6e7a2b0959d7dd28e9e49da4)
- [Illustrated Guide to Transformers Neural Network: A step by step explanation](https://www.youtube.com/watch?v=4Bdc55j80l8&list=LL&index=7&ab_channel=TheAIHacker): Good for a quick walkthrough of GPT Review