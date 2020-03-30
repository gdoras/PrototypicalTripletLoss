# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Addons")
class PrototypicalTripletLoss(tf.keras.losses.Loss):
    """Computes the prototypical triplet loss with semi-hard negative mining.

    This is a direct adaptation of the original triplet loss implementation.
    See: https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss

    The loss encourages the positive distances (between an anchor embedding and
    the centroid of embeddings with same labels) to be smaller than the minimum negative
    distance (between the anchor embedding and the centroids of embedding clusters
    with a different label) among which are at least greater than the positive distance plus the
    margin constant (called semi-hard negative) in the mini-batch. If no such negative exists,
    uses the largest negative distance instead

    See:
        https://arxiv.org/abs/1503.03832 (for the original triplet loss)
        https://arxiv.org/pdf/1910.09862 (for the prototypical triplet loss)
    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      name: Optional name for the op.
    """

    def __init__(self, margin=1.0, name=None):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.margin = margin

    def call(self, y_true, y_pred):
        return prototypical_triplet_loss(y_true, y_pred, margin=self.margin)

    def get_config(self):
        config = {"margin": self.margin}
        base_config = super().get_config()
        return {**base_config, **config}


@tf.function
def prototypical_triplet_loss(y_true, y_pred, margin=1.0):
    """
    Computes the prototypical triplet loss with semi-hard negative mining.

    This is a direct adaptation of the original triplet loss implementation.
    See: https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss

    The loss encourages the positive distances (between an anchor embedding and
    the centroid of embeddings with same labels) to be smaller than the minimum negative
    distance (between the anchor embedding and the centroids of embedding clusters
    with a different label) among which are at least greater than the positive distance plus the
    margin constant (called semi-hard negative) in the mini-batch. If no such negative exists,
    uses the largest negative distance instead

    See:
        https://arxiv.org/abs/1503.03832 (for the original triplet loss)
        https://arxiv.org/pdf/1910.09862 (for the prototypical triplet loss)

    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.

    Returns:
        triplet_loss: tf.float32 scalar.
    """
    labels, embeddings = y_true, y_pred

    # get cluster labels
    lshape = tf.shape(labels)
    clusters_labels, _, num_embeddings_per_cluster = tf.unique_with_counts(
        tf.reshape(labels, [lshape[0]])
    )

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    labels = tf.reshape(labels, [lshape[0], 1])

    lshape = tf.shape(clusters_labels)
    # clusters_labels = tf.reshape(clusters_labels, [lshape[0], 1])
    num_clusters = tf.size(clusters_labels)

    # get the adjacency matrix (rows=embeddings, cols=clusters, content: embedding belong to cluster
    adjacency = tf.equal(
        labels, tf.transpose(clusters_labels)
    )  # [batch_size, num_clusters]

    # computes the centroids embedding
    centroids = tf.linalg.matmul(
        tf.cast(adjacency, dtype=tf.float32), embeddings, transpose_a=True
    )  # [num_clusters, embeddings_size]

    centroids = tf.divide(
        centroids,
        tf.expand_dims(tf.cast(num_embeddings_per_cluster, dtype=tf.float32), axis=1),
    )
    centroids = tf.nn.l2_normalize(centroids, axis=1)

    # Build pairwise squared distance matrix.
    pairwise_distances = _pairwise_distances(
        feature_A=embeddings, feature_B=centroids, squared=True
    )  # [batch_size, num_clusters]

    # Invert so we can select negatives only.
    adjacency_not = tf.logical_not(adjacency)

    batch_size = tf.size(labels)

    # Compute the mask [bxn, n] equal to:
    # - d_aj < d_ak for all k s.t. j and k have different labels
    # - False if k and j have same labels
    pairwise_distances_tile = tf.tile(pairwise_distances, [num_clusters, 1])

    mask = tf.logical_and(
        tf.tile(adjacency_not, [num_clusters, 1]),
        tf.greater(
            pairwise_distances_tile,
            tf.reshape(tf.transpose(pairwise_distances), [-1, 1]),
        ),
    )  # [bxn, n]
    mask_final = tf.reshape(
        tf.greater(
            tf.reduce_sum(tf.cast(mask, dtype=tf.float32), 1, keepdims=True), 0.0
        ),
        [num_clusters, batch_size],
    )
    mask_final = tf.transpose(mask_final)
    # comment Guillaume: [mask_final]_{ij} = [it exists at least another distance d_{mn} s.t. d_{ij} < d_{mn}]

    adjacency_not = tf.cast(adjacency_not, dtype=tf.float32)  # [b, n]
    mask = tf.cast(mask, dtype=tf.float32)  # [bxn, n]

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = tf.reshape(
        masked_minimum(pairwise_distances_tile, mask), [num_clusters, batch_size]
    )
    negatives_outside = tf.transpose(negatives_outside)
    # comment Guillaume: [negatives_outside]_{ij} = [if it exists, the smallest d_{mn} s.t. d_{ij} < d_{mn}, else 0]

    # negatives_inside: largest D_an.
    negatives_inside = tf.tile(
        masked_maximum(pairwise_distances, adjacency_not), [1, num_clusters]
    )
    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)
    # comment Guillaume: [semi_hard_negatives]_{ij} = [if it exists, the smallest d_{mn} s.t. d_{ij} < d_{mn}, else the largest d_{mn} < d_{ij}]

    loss_mat = tf.add(margin, pairwise_distances - semi_hard_negatives)

    mask_positives = tf.cast(
        adjacency, dtype=tf.float32
    )  # - tf.diag(tf.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = tf.reduce_sum(mask_positives)

    triplet_loss = tf.truediv(
        tf.reduce_sum(tf.maximum(tf.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name="prototypical_triplet_loss",
    )

    return triplet_loss


def _pairwise_distances(feature_A, feature_B=None, squared=False):
    """
    Directly from https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss

    Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      feature_A: 2-D Tensor of size [number of data A, feature dimension].
      feature_B: 2-D Tensor of size [number of data B, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data A, number of data B].
    """
    if feature_B is None:
        feature_B = feature_A

    pairwise_distances_squared = tf.add(
        tf.reduce_sum(tf.square(feature_A), axis=[1], keepdims=True),
        tf.reduce_sum(tf.square(tf.transpose(feature_B)), axis=[0], keepdims=True),
    ) - 2.0 * tf.linalg.matmul(feature_A, tf.transpose(feature_B))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.sqrt(
            pairwise_distances_squared + tf.cast(error_mask, tf.float32) * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = tf.multiply(
        pairwise_distances, tf.cast(tf.logical_not(error_mask), tf.float32)
    )

    if feature_B is None:
        num_data = tf.shape(feature_A)[0]
        # Explicitly set diagonals to zero.
        mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
            tf.ones([num_data])
        )
        pairwise_distances = tf.multiply(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Directly from https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.
    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = tf.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.reduce_max(tf.multiply(data - axis_minimums, mask), dim, keepdims=True)
        + axis_minimums
    )
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Directly from https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.
    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = tf.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.reduce_min(tf.multiply(data - axis_maximums, mask), dim, keepdims=True)
        + axis_maximums
    )
    return masked_minimums
