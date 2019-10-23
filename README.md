# PrototypicalTripletLoss


A Tensorflow implementation of the prototypical triplet loss.
 
This adaptation of the standard triplet loss minimizes the distance between an anchor 
and the prototype of its class (the centroid of all embeddings in this class), while 
maximizing with the prototype of all other classes.

Prototype computation is done online within each mini-batch, and followed by
semi-hard triplet mining. Here, each triplet contains an anchor, its positive
prototype (the centroid of all embeddings of the anchor's class) and a negative prototype
(the centroid of all embeddings of class which is not the anchor's class).


## References
See this <a href="https://arxiv.org/pdf/1910.09862.pdf">paper</a> for details. 

See original implementation <a href="https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss">here</a>.