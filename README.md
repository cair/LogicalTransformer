# LogicalTransformer

Here is a quick summary of where we are:
* We have an autoencoder that can do flat logical word embedding at the level of neural networks. https://arxiv.org/abs/2301.00709
* We know that TM produces human-like rules in NLP. https://onlinelibrary.wiley.com/doi/full/10.1111/exsy.12873
* Set of words is our preferred representation.
* Hypervectors with bundling of words gives similar accuracy, in fewer epochs. Each word is represented by a random binary vector, and ORed together to represent a text.
* Convolution over single word vectors works.
* We know how to combine different perspectives on the data in a Tsetlin Machine Cpmposite. https://arxiv.org/abs/2309.04801

Questions to investigate:
* How can we make the random hypervectors less random and more semantic for boosted accuracy, e.g., in integration with logical embeddings?
* How can we create hierarchical logical embeddings that represent more complex sentence structures, above the word level representation? E.g., by building up a representation around each word separately, incorporating context into the hypervector.
* How can we integrate language and images in the autoencoder?
* Given the new Tsetlin machine composites, should the architecture be flat or deep?
* And more, ... :-)

## Example with Bayesian Word Profiling and Word Sense Refinement

'''
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
'''
