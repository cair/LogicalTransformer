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

## Two Meanings of Heart

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

## Two Meanings of Bank

```bash
python3 ./profile.py --target_one "bank" --target_two "river"

['fargo' 'banks' 'flows' 'frozen' 'barrier' 'flowing' 'thames' 'stricken'
 'pumping' 'deposits' 'pumped' 'west' 'shore' 'plunged' 'rogue' 'branches'
 'toxic' 'central' 'collapsed' 'branch' 'rescue' 'rescued' 'plunging'
 'hike' 'rock' 'fdic' 'institution' 'wells' 'northern' 'surging' 'abbey'
 'steep' 'drain' 'agricultural' 'construction' 'headquartered' 'jordan'
 'pump' 'freezing' 'sank' 'plunge' 'strip' 'charlotte' 'meets' 'winding'
 'east' 'largest' 'falls' 'millennium' 'lowered' 'rising' 'below' 'ny'
 'boutique' 'forecasting' 'reopen' 'rises' 'facility' 'located'
 'settlement' 'checkpoints' 'reinforced' 'operates' 'triggered' 'shanghai'
 'emergency' 'above' 'checkpoint' 'bear' 'elevated' 'colonial' 'jumped'
 '6m' 'restore' 'ga' '450' 'dried' 'warning' 'massive' 'tel'
 'headquarters' 'owned' 'flow' 'backdrop' 'capital' 'owns' 'plummeted'
 'darling' 'height' 'sands' 'dip' 'spree' 'territory' 'manages'
 'overnight' 'widening' 'predicted' 'richmond' 'forecast' 'collapse']
```

```bash
python3 ./profile.py --target_one "bank" --target_two "money"

['laundering' 'savers' 'tarp' 'lend' 'repaid' 'repay' 'bailout' 'deposit'
 'icelandic' 'taxpayer' 'easing' 'deposits' 'lending' 'borrow' 'inject'
 'borrowed' 'fdic' 'lent' 'accounts' 'taxpayers' 'printing' 'banks'
 'pumped' 'borrowing' 'liquidity' 'bonuses' 'collateral' 'owed' 'raise'
 'ecb' 'funds' 'sums' 'transfers' 'loans' 'mutual' 'account' 'billions'
 'fund' 'assets' 'invested' 'asset' 'lender' 'insured' 'treasury'
 'pumping' 'ubs' 'savings' 'bonds' 'monetary' 'lehman' 'lenders' 'banking'
 'loan' 'reserve' 'injected' 'investors' 'goldman' 'creditors' 'debts'
 'guarantee' 'investment' 'bailed' 'investments' 'raising' 'aig'
 'citigroup' 'transactions' 'institutions' 'hbos' 'bankers' 'purchases'
 'bets' 'financed' 'withdraw' 'pump' 'stimulate' 'stearns' 'mortgages'
 'cash' 'stimulus' 'imf' 'hedge' 'brokerage' 'credit' 'invest'
 'guarantees' 'financing' 'fraud' 'lloyds' 'troubled' 'clients' 'flowing'
 'rates' 'sachs' 'interest' 'mortgage' 'jpmorgan' 'finance' 'iceland'
 'sum']
```



