import tensorflow as tf


class MacroF1(tf.keras.metrics.Metric):
    """Streaming macro-averaged F1 (vector shape pre-allocated)"""
    def __init__(self, num_classes: int, name="macro_f1", **kw):
        super().__init__(name=name, **kw)
        self.num_classes = num_classes
        self.tp = self.add_weight(
            name="tp", shape=(num_classes,), initializer="zeros", dtype=tf.float32
        )
        self.fp = self.add_weight(
            name="fp", shape=(num_classes,), initializer="zeros", dtype=tf.float32
        )
        self.fn = self.add_weight(
            name="fn", shape=(num_classes,), initializer="zeros", dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        y_true_oh = tf.one_hot(y_true, self.num_classes)
        y_pred_oh = tf.one_hot(y_pred, self.num_classes)

        if sample_weight is not None:
            sample_weight = tf.reshape(sample_weight, [-1, 1])
            y_true_oh *= sample_weight
            y_pred_oh *= sample_weight

        self.tp.assign_add(tf.reduce_sum(y_true_oh * y_pred_oh, axis=0))
        self.fp.assign_add(tf.reduce_sum((1 - y_true_oh) * y_pred_oh, axis=0))
        self.fn.assign_add(tf.reduce_sum(y_true_oh * (1 - y_pred_oh), axis=0))

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall    = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1        = tf.math.divide_no_nan(2 * precision * recall,
                                          precision + recall)
        return tf.reduce_mean(f1)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(tf.zeros_like(v))
