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

## Overleaf Paper

https://www.overleaf.com/5141817728jzzqkkspjwjc

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

1. Input: Document, Bayesian embeddings, Distance threshold **d**
2. For each target word  **w<sub>t</sub>** in the document:
   1. Compare the embedding of the target word **w<sub>t</sub>** with the embedding of the neighbour words **w<sub>n</sub>** (cosinus distance)
   2. Refine the embedding values of the target word **w<sub>t</sub>** by multiplying in the embedding values of the neighbour words **w<sub>n</sub>** (see above demo). Only the neighbour words that are within cosinus distance **d** are used here.
3. Add the refined embeddings together to form the final representation of the document.
   
### B. With Position Encoding

#### Create Bayesian Embedding With Position Information

1. Create a 3D matrix *WxWx2L* where *W* is the size of the vocabulary and *L* is size of the relative position information.
2. Calculate position specific mutual information between each target word **w<sub>t</sub>** and each neighbour word **w<sub>n</sub>** at each position **i** relative to the target word **w<sub>t</sub>**.

**Example:** A vocabulary of size *W=10000* with relative position encoding *[<=-10, -9, -8, ...,-2, -1, +1, +2, ..., +8, +9, >=+10]* gives a 3D matrix of size 7.5 GB.

**Optional:** Group position information into ranges.

#### Create Representation of Document With Position Information

1. Input: Document, Bayesian embeddings with positional information, Distance threshold **d**
2. For each unique target word **w<sub>t</sub>** in the document:
   1. Compare the embedding of the target word **w<sub>t</sub>** with the embedding of the neighbour words **w<sub>n</sub>**.
      1. Align the relative position information of each embedding to correctly match the contexts (e.g., for the phrase "a car", "a" has position *-1* relative to "car" in the embedding of "car", while "car" has position *+1* relative to "a" in the embedding of "a"). 
   3. Refine the embedding values of the target word **w<sub>t</sub>** by multiplying in the embedding values of the neighbour words **w<sub>n</sub>** (after the position information has been aligned). Only the neighbour words that are within cosinus distance **d** are used here.
3. Add the refined embeddings together to form the final representation of the document.
