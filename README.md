# Robust Net 
### Introduction
A novel seq2seq model aim to be robust and lightweight, an independent resaerch and effort

### Architecture

The design is that of the original transformer model with slight changes to reduce parameter count and increase robustness

A the model has a shared embedding matrix that is decomposed to a smaller dimension followed by factorization and the ouput is added to sinusoidal encodings.
It uses the usual self attention but with same weights for query and key. 
`emb_dim = d_model / 2` and `fact_weights = d_model / 2  x  d_model`

The enocder part has a dynamic gating mechanism to refine the attention output.\
`A_refined = LayerNorm(sigmoid(WA + b) * A_i + A_i)`

where A_i is current layer self attention and A is previous layer attention output

The feed forward layer uses a GeLU activation and with `ff_units = 2 * d_model`

### Pretraining and setup

The model is pretrained on sequence completion task, with dynamic noise (5% random token deletion + 5% random token replacement).
Due to hardware constraints, I could only pretrain a small scale version of this model

`d_model = 256` `n_layers = 2` `n_heads = 8` `vocab_size = 15037` `maxlen = 128` with a total of 8.2M Params\
`batch_size = 64` `epochs = 15`

The model was pretrained on 2.6M samples (diverse data collected from kaggle and preprocessed) with 39000 steps per epoch

AdamW optimizer with `weight_decay = 0.01`\
`lr = 2e-4` `warmup_rate = 0.05` followed by cosine decay

### Tokenization

Byte Pair tokenizer, that was implemented from scratch.\
The tokenizer was trained on 500000 words 

tokenization example - 
`John's family is camping.` -> `john ' s family is camp ing .`

### Fine-tuning

The model is fine tuned for summarization and NMT task, as well as text classification (the encoder only model).\
The observed best lr scheduling turned out to be
5% linear warmup till `2e-4` and then cosine decay to `1e-7`

News summarization
`BLEU-1 : 0.60 | BLEU-2 : 0.51 | BLEU-3 : 0.43 | BLEU-4 : 0.34`
`ROUGE-1 : 0.57 | ROUGE-2 : 0.29 | ROUGE-L : 0.51`

French to english NMT
`BLEU-1 : 0.72 | BLEU-2 : 0.64 | BLEU-3 : 0.57 | BLEU-4 : 0.51`
`ROUGE-1 : 0.70 | ROUGE-2 : 0.51 | ROUGE-L : 0.67`

for text classification, the encoder only model is 3M params

IMDB Reviews
`F1 Score : 0.87`

Disaster tweet classification
`F1 Score : 0.77`

The output from the `test.csv` file was submitted to the kaggle competition and achieved `F1 Score = 0.80`













