# Logical Transformer

Here is a quick summary of where we are:
* We have an autoencoder that can do flat logical word embedding at the level of neural networks. https://arxiv.org/abs/2301.00709
* We know that TM produces human-like rules in NLP. https://onlinelibrary.wiley.com/doi/full/10.1111/exsy.12873
* Set of words is our preferred representation.
* Hypervectors with bundling of words gives similar accuracy, in fewer epochs. Each word is represented by a random binary vector, and ORed together to represent a text.
* Convolution over single word vectors works.
* We know how to combine different perspectives on the data in a Tsetlin Machine Composite. https://arxiv.org/abs/2309.04801

Questions to investigate:
* How can we make the random hypervectors less random and more semantic for boosted accuracy, e.g., in integration with logical embeddings?
* How can we create hierarchical logical embeddings that represent more complex sentence structures, above the word level representation? E.g., by building up a representation around each word separately, incorporating context into the hypervector.
* How can we integrate language and images in the autoencoder?
* Given the new Tsetlin machine composites, should the architecture be flat or deep?
* Our starting point is a single set-of-words representation encompassing the complete document. How can we create one set-of-words representation per word instead, incorporating multiple local views of the document?
* Should the transformer part be Bayesian pre-processing (see Bayesian Word Profiler), or should the Tsetlin machine perform the transformation?
* How to we incorporate relative word positions into the representation? 
* And more, ... :-)

## Promising Solution Ingredients

* Composites - goes from 66% to 80% accuracy on CIFAR10 with 2000 clauses per class per member classifier.
* Reasoning by elimination - obtains 90% accuracy on IMDB in a single epoch with 1000 clauses per class.
* Convolution - effective for images.
* Logical word modelling produces highly competitive word embeddings.
* Set of tokens - computationally efficient and effective for language modelling.
* Question: How to integrate them?
   * Model: Set of tokens (by Rupsa). Supports semantic modelling and reasoning by elimination. Convolution can be used to produce image tokens (Vojtech) or brute force by extracting all unique patches occuring e.g. more that five times in the data (Vojtech).
   * For each class, train a classifier per token, predicting the presence of that token (by Ahmed). Can be done self-supervised, scales well. Incorporates logical language modelling (Bimal). Each classifier is an "embedding" of its token.
   * Introduce relative position information (by Vojtech). Supports learning image and language constructs from the context, like a transformer.
   * Composite inference (All):
      1. For a given input, go through each token present (maybe also not present, but that requires more computation).
      2. Calculate the overall class sum for each class by summing up the vote sum of each token classifier for the tokens present.
      3. Output the class with the largest overal class sum.
   * **Summarized:** Each token gets its own Tsetlin machine as its embedding. Sentence/document/image embedding is a *composite* of the Tsetlin machines of the tokens appearing in the sentence/document/image.  
* Alternative strategy: Concatenate individual tokens that are related (e.g., one predicts the other) to create "joint" tokens. The joint tokens are to represent the text more precisely.

## Logical Transformer Architecture

<p align="center">
  <img width="90%" src="https://github.com/cair/LogicalTransformer/blob/main/Logical_Transformer_Architecture_NLP.png">
</p>
      
## Overleaf Paper

https://www.overleaf.com/5141817728jzzqkkspjwjc

## Tsetlin Machine Embedding

dockerhub public image:   https://hub.docker.com/layers/bobsbimal58/tsetlinembed/v2/images/sha256-6bc5aa7e4cc3c24f797aed64854883b4b0e9014dcbccf0bfa549085ada8f85b0?context=repo

playground:  https://labs.play-with-docker.com/p/cl5nvoggftqg00bohm10#cl5nvogg_cl5pfhufml8g00evf7mg

script to run in playground: docker run -dp 0.0.0.0:80:80 bobsbimal58/tsetlinembed:v2

## Example with Bayesian Word Profiling and Word Sense Refinement

### Two Meanings of Heart

```bash
python3 ./profile.py --target_one "heart" --target_two "love"

['bleeding' 'soul' 'valentine' 'crazy' 'hearts' 'romance' 'sacred' 'dies'
 'romantic' 'rhythm' 'dear' 'tale' 'purple' 'ace' 'lonely' 'pray' 'song'
 'sing' 'loving' 'joy' 'twilight' 'passionate' 'poetry' 'passion'
 'happiness' 'lyrics' 'diamonds' 'my' 'lover' 'darkness' 'emotion' 'sung'
 'songs' 'beloved' 'anthem' 'humor' 'memoir' 'humour' 'sang' 'tales'
 'singing' 'emotions' 'hbo' 'chocolate' 'beautiful' 'novel' 'emotionally'
 'loved' 'genre' 'childhood' 'lifestyle' 'publicist' 'ellen' 'myers'
 'princess' 'jesus' 'emotional' 'poet' 'gentle' 'loves' 'eat' 'finds'
 'forever' 'novels' 'grief' 'hannah' 'spirit' 'mighty' 'endless'
 'genuinely' 'shaped' 'god' 'shakespeare' 'caring' 'dying' 'grammy'
 'enduring' 'vibrant' 'diana' 'syndrome' 'your' 'sight' 'drama'
 'guitarist' 'hormone' 'widow' 'combines' 'diamond' 'chorus' 'singer'
 'pure' 'pains' 'intimate' 'lies' 'charm' 'lovely' 'story' 'actress'
 'truly' 'fat']
```

```bash
python3 ./profile.py --target_one "heart" --target_two "hospital"

['cardiac' 'transplant' 'underwent' 'surgeon' 'kidney' 'complications'
 'chest' 'surgery' 'cardiovascular' 'surgeons' 'lungs' 'patients' 'lung'
 'bypass' 'liver' 'bleeding' 'respiratory' 'stroke' 'pains' 'stab'
 'undergoing' 'acute' 'condition' 'patient' 'surgical' 'breathing'
 'paramedics' 'doctors' 'organs' 'died' 'diabetes' 'organ' 'mortality'
 'asthma' 'blood' 'procedure' 'suffering' 'recovering' 'oxygen' 'treated'
 'admissions' 'hospitalized' 'discharged' 'disease' 'treating' 'mortem'
 'symptoms' 'suffered' 'ambulance' 'chronic' 'medicine' 'medical'
 'infection' 'treatment' 'illnesses' 'clinical' 'premature' 'dr'
 'diagnosed' 'scan' 'physician' 'strokes' 'undergo' 'dying' 'clinic'
 'nurse' 'physicians' 'medication' 'stabbed' 'stomach' 'intensive'
 'cancer' 'diseases' 'nurses' 'outcomes' 'tissue' 'disorders' 'doctor'
 'infections' 'brain' 'hospitals' 'therapy' 'trauma' 'attack' 'babies'
 'unconscious' 'diagnosis' 'medications' 'dementia' 'publicist' 'wound'
 'johns' 'treat' 'imaging' 'infants' 'prescribed' 'chemotherapy' 'donor'
 'syndrome' 'treatments']
```

### Two Meanings of Bank

```bash
python3 ./profile.py --target_one "bank" --target_two "river"

['fargo' 'banks' 'flows' 'frozen' 'barrier' 'flowing' 'thames' 'stricken'
 'pumping' 'deposits' 'west' 'shore' 'pumped' 'plunged' 'branches' 'rogue'
 'toxic' 'central' 'collapsed' 'branch' 'rescue' 'rescued' 'plunging'
 'hike' 'rock' 'institution' 'fdic' 'wells' 'northern' 'surging' 'abbey'
 'steep' 'agricultural' 'construction' 'drain' 'headquartered' 'jordan'
 'pump' 'strip' 'sank' 'freezing' 'plunge' 'charlotte' 'meets' 'east'
 'largest' 'winding' 'falls' 'rising' 'lowered' 'below' 'millennium' 'ny'
 'boutique' 'rises' 'reopen' 'facility' 'forecasting' 'located'
 'settlement' 'operates' 'emergency' 'above' 'triggered' 'shanghai'
 'checkpoints' 'reinforced' 'bear' 'checkpoint' 'colonial' 'elevated'
 'jumped' 'restore' '6m' 'ga' '450' 'warning' 'massive' 'owned'
 'headquarters' 'capital' 'dried' 'flow' 'owns' 'darling' 'backdrop'
 'height' 'tel' 'plummeted' 'dip' 'territory' 'sands' 'overnight' 'spree'
 'manages' 'predicted' 'forecast' 'collapse' 'widening' 'richmond']
```

```bash
python3 ./profile.py --target_one "bank" --target_two "money"

['laundering' 'savers' 'tarp' 'lend' 'bailout' 'repaid' 'repay' 'deposit'
 'taxpayer' 'easing' 'icelandic' 'lending' 'deposits' 'borrow' 'inject'
 'borrowed' 'accounts' 'fdic' 'lent' 'taxpayers' 'banks' 'printing'
 'borrowing' 'bonuses' 'liquidity' 'pumped' 'collateral' 'owed' 'raise'
 'funds' 'ecb' 'sums' 'loans' 'account' 'transfers' 'mutual' 'billions'
 'fund' 'assets' 'invested' 'asset' 'treasury' 'lender' 'ubs' 'savings'
 'insured' 'pumping' 'bonds' 'monetary' 'banking' 'loan' 'lenders'
 'lehman' 'reserve' 'investors' 'goldman' 'injected' 'debts' 'creditors'
 'investment' 'guarantee' 'investments' 'raising' 'bailed' 'aig'
 'citigroup' 'institutions' 'transactions' 'bankers' 'hbos' 'purchases'
 'financed' 'bets' 'withdraw' 'pump' 'cash' 'stimulus' 'mortgages'
 'stearns' 'hedge' 'credit' 'imf' 'stimulate' 'invest' 'fraud' 'financing'
 'brokerage' 'troubled' 'guarantees' 'clients' 'lloyds' 'rates' 'interest'
 'mortgage' 'sachs' 'finance' 'flowing' 'jpmorgan' 'iceland' 'sum']
```

## Bayesian Transformer Prototype

### A. Without Position Encoding

1. Input: Document **d**, Bayesian embedding vector **v<sub>t</sub>** for each target word **w<sub>t</sub>** in the vocabulary, Relevance threshold **r**.
2. For each target word **w<sub>t</sub>** appearing in document **d**:
   1. Look up the value of each neighbouring word **w<sub>n</sub>** in document **d** using the embedding vector **v<sub>t</sub>** of the target word **w<sub>t</sub>**.
   2. Identify the neighbouring words **w<sub>n</sub>** whose value in **v<sub>t</sub>** exceeds the relevance threshold **r</sub>**, i.e., **v<sub>t</sub>[w<sub>n</sub>] > r**.
   3. Refine the embedding vector **v<sub>t</sub>** of the target word **w<sub>t</sub>** by multiplying in the embedding vectors **v<sub>n</sub>** of the identified neighbouring words **w<sub>n</sub>** (see above demo for an example of how to multiply the embedding vectors).
3. Add the refined embeddings together to form the final representation of the document.
   
### B. With Position Encoding

#### Create Bayesian Embedding With Position Information

1. Create a 3D matrix *WxWxL* where *W* is the size of the vocabulary and *L* is size of the relative position information.
2. Calculate position specific mutual information between each target word **w<sub>t</sub>** and each neighbour word **w<sub>n</sub>** at each position **i** relative to the target word **w<sub>t</sub>**.

**Example:** A vocabulary of size *W=10000* with relative position encoding *[<=-10, -9, -8, ...,-2, -1, +1, +2, ..., +8, +9, >=+10]*, i.e., *L=20*, gives a 3D matrix of size 7.5 GB. If we increase the vocabulary size to *W=50000*, the size of the 3D matrix grows to 186 GB.

#### Create Representation of Document With Position Information

1. Input: Document **d**, Bayesian embedding vector **v<sub>t</sub>** with positional information for each target word **w<sub>t</sub>** in the vocabulary, Relevance threshold **r**.
2. For each target word **w<sub>t</sub>** appearing in document **d**:
   1. Look up the value of each neighbouring word **w<sub>n</sub>** in document **d** using the embedding vector **v<sub>t</sub>** of the target word **w<sub>t</sub>**. Take into account the position **p** of the neighbouring word **w<sub>n</sub>** relative to the target word **w<sub>t</sub>**.
   2. Identify the neighbouring words **w<sub>n</sub>** whose value in **v<sub>t</sub>** at relative position **p** exceeds the relevance threshold **r</sub>**, i.e., **v<sub>t</sub>[p:w<sub>n</sub>] > r**.
   3. Refine the embedding vector **v<sub>t</sub>** of the target word **w<sub>t</sub>** by multiplying in the embedding vectors **v<sub>n</sub>** of the identified neighbouring words **w<sub>n</sub>**. Remember to align the relative position information of the embedding vectors.
3. Add the refined embeddings together to form the final representation of the document.

## Bayesian Embedding With Position Information for 2D Data (Images, Board Games)

Add 2D position grid to the Bayesian embedding approach.

## Datasets

* [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)?
* Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens)
* [One Billion Words Benchmark](https://www.kaggle.com/datasets/alexrenz/one-billion-words-benchmark)
* [Downloaded](https://www.dropbox.com/scl/fo/otf37qiwyy7p4ic6il7dl/h?rlkey=aa28muzrr5ng1qm4ksmwqtw56&dl=0)
* [Yahoo Answers Topic Classification](https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset)

## Possible Baseline for Embeddings

[Near-lossless Binarization of Word Embeddings](https://github.com/tca19/near-lossless-binarization)

## Binary Embeddings

[Collection of Binary Embeddings](https://www.dropbox.com/scl/fo/5uvac9ztiazyw90rtmp28/h?rlkey=9mb1tck9tv65llvezbayy6gc1&dl=0)

