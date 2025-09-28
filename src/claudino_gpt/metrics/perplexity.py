import tensorflow as tf


class Perplexity(tf.keras.metrics.Metric): # type: ignore
    def __init__(self, name='perplexity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_loss = self.add_weight(name='total_loss', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # type: ignore
        current_loss = loss_fn(y_true, y_pred)
        self.total_loss.assign_add(tf.reduce_mean(current_loss))
        self.count.assign_add(1.0)

    def result(self):
        mean_loss = self.total_loss / self.count
        return tf.math.exp(mean_loss)

    def reset_state(self):
        self.total_loss.assign(0.)
        self.count.assign(0.)
