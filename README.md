# crypto-trade

Reinforcement learning crypto trading bot. 
* Custom OpenAI gym environment
* proximal policy optimizations algorithm
* 


## Dataframe dimensions

| coin | open | high | low | closing | volume |
|------|------|------|-----|---------|--------|
| btc  | .... | .... | ... | ....... | ...... |



| time | btc_open | btc_close | ... | btc_volume | btc_amount | eth_open | ... | eth_volume | eth_amount | ...

## Tensorboard

* https://towardsdatascience.com/a-quickstart-guide-to-tensorboard-fb1ade69bbcf


## Dockerfile tensorflow

docker build . --tag='crypto-trade-tf'

docker run -d --gpus all --name crypto-trade-tf crypto-trade-tf


## Papers

* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7384672/
* https://proceedings.neurips.cc/paper/1998/file/4e6cd95227cb0c280e99a195be5f6615-Paper.pdf
* https://helda.helsinki.fi/bitstream/handle/10138/316961/thesis-AndresHuertas.pdf
* https://arxiv.org/pdf/1901.08740.pdf

