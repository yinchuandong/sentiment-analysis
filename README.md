# ML Library
1. [Dependencies](#dependencies)
2. [Introduciton](#introduction)

## Dependencies
- Python >= 3.7


## Introduciton
- [x] rubixml (A pip library)
  - [x] TextCNN (PyTorch implementation from scratch)
  - [ ] RNN (LSTM/GRU cell)
  - [ ] spaCy (Residual TextCNN)
  - [ ] BERT (Transformer)

- [x] trainer (The main application to train models)
  - [ ] Analytics (LDA + Word Cloud)
  - [x] Train Model
  - [x] Docker

- [x] webapi (Back-end)
  - [x] Flask Server
  - [x] Docker

- [ ] webapp (Front-end)
  - [ ] React-Redux

- [ ] Pipeline
  - [ ] buildkite


## webapi

- Endpoint
``` HTTP
POST /api/textcnn/predict
```

- Description:

| name       | type            | description                    |
| ---------- | ---------       | ------------------------------ |
| text       | list of strings | the text to be predicted       |  

- Example:
![webapi1](./docs/imgs/webapi1.png)
