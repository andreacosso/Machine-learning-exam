import tensorflow as tf
from tensorflow.keras import layers, regularizers
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
import numpy as np

TARGET_SIZE = (224, 224) # Default input size for ResNet50

# model builder for the classifier ResNet50
def build_class_ResNet50_model(
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        trainable_backbone: bool = False,
        unfrozen_blocks=("conv5_block3",),
        metrics=None,                       # <- NEW
):
    """
    Build & compile a ResNet50-based 10-class classifier.

    Args
    ----
    dropout_rate       : dropout after GAP layer.
    learning_rate      : Adam learning-rate.
    trainable_backbone : fine-tune backbone if True.
    unfrozen_blocks    : tuple of block names to unfreeze in the backbone.
    metrics            : list of Keras metrics to use in model.compile().
                         Default = ["accuracy"].

    Returns
    -------
    tf.keras.Model (compiled).
    """
    # ---------------- inputs ----------------
    inputs = tf.keras.Input(shape=(*TARGET_SIZE, 3))
    
    # ---------------- backbone ----------------
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*TARGET_SIZE, 3),
        input_tensor=inputs,
        pooling=None,                 #  we'll add GAP ourselves
    )
    
    # Freeze everything first
    backbone.trainable = False
    if trainable_backbone:
        for layer in backbone.layers:
            if any(layer.name.startswith(b) for b in unfrozen_blocks):
                layer.trainable = True

    # ---------------- head ----------------
    #x = backbone(inputs, training=trainable_backbone)   # BatchNorm update only if fine-tuning
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)  # GAP layer
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # ---------------- compile ----------------
    if metrics is None:
        metrics = ["accuracy"]           # default
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=metrics,
    )
    return model



def build_dnn_classifier(
    input_dim: int,
    num_classes: int,
    hidden_units: list[int] = [1024, 512, 256],
    dropout_rate: float = 0.3,
    weight_decay: float = 1e-4,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Builds & compiles a fully-connected classifier with BatchNormalization.
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    # Hidden layers
    for units in hidden_units:
        x = layers.Dense(
            units,
            activation=None,
            kernel_regularizer=regularizers.l2(weight_decay)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if dropout_rate and dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ------- 4. Define feature-head model for 3 normal classes -------
def build_feature_model(
    feature_dim: int = 32,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    trainable_backbone: bool = False,
    unfrozen_blocks=("conv5_block3",),
    num_classes: int = 3,
    metrics=None
):
    """
    Build & compile a ResNet50-based model that outputs:
      - 32-D feature vector
      - Logits for 'num_classes' normal classes
    """
    inputs = tf.keras.Input(shape=(*TARGET_SIZE, 3))

    # Backbone
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling=None
    )
    backbone.trainable = False 
    if trainable_backbone:
        for layer in backbone.layers:
            if any(layer.name.startswith(b) for b in unfrozen_blocks):
                layer.trainable = True

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    feature_vec = tf.keras.layers.Dense(feature_dim, activation=None, name="feature_32")(x)
    logits = tf.keras.layers.Dense(num_classes, activation="softmax", name="logits")(feature_vec)

    model = tf.keras.Model(inputs, logits, name="resnet_feature_head")

    # Compile
    if metrics is None:
        metrics = ["accuracy"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=metrics
    )
    return model




# MAF

def make_clamp_fn(made, min_ls, max_ls, dropout_rate=None):
    """
    Wraps a MADE instance, optionally applying dropout 
    to its outputs before splitting.
    """
    def clamp_fn(x, training=False):
        params = made(x)
        # Optional dropout
        if dropout_rate:
            # apply the same dropout mask to both halves
            params = tf.keras.layers.Dropout(dropout_rate)(params, training=training)
        shift, log_scale = tf.split(params, 2, axis=-1)
        log_scale = tf.clip_by_value(log_scale, min_ls, max_ls)
        return tf.concat([shift, log_scale], axis=-1)
    return clamp_fn

def build_clamped_maf_keras_model(
    num_flows: int,
    hidden_units: list[int],
    event_shape: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    clipnorm: float = 1.0,
    dropout_rate: float | None = None,
    use_permutation: bool = True,
    min_log_scale: float = -5.0,
    max_log_scale: float = 5.0,
    seed: int | None = 123, 
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(event_shape,), name="embeddings")
    bijectors = []
    #perm = list(reversed(range(event_shape)))
    rng = np.random.default_rng(seed)
    for _ in range(num_flows):
        # 1) MADE with L2 on its internal Dense layers
        made = tfb.AutoregressiveNetwork(
            params=2,
            hidden_units=hidden_units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        )
        # 2) clamp + optional dropout in the shift/log-scale fn
        clamp_fn = make_clamp_fn(made, min_log_scale, max_log_scale, dropout_rate)
        maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=clamp_fn)
        bijectors.append(maf)

        # 3) optional permutation
        if use_permutation:
            perm = rng.permutation(event_shape)
            bijectors.append(tfb.Permute(permutation=perm))

    # Drop trailing permutation so last bijector is a MAF
    if use_permutation:
        bijectors.pop()

    # 4) Build the full flow
    maf_chain = tfb.Chain(list(reversed(bijectors)))
    base = tfd.MultivariateNormalDiag(
        loc=tf.zeros(event_shape), scale_diag=tf.ones(event_shape)
    )
    flow = tfd.TransformedDistribution(distribution=base, bijector=maf_chain)

    log_prob = tf.expand_dims(flow.log_prob(inputs), axis=-1)
    model = tf.keras.Model(inputs=inputs, outputs=log_prob, name="clamped_maf")
    model.add_loss(-tf.reduce_mean(log_prob))
    model.flow = flow 
    model.maf_chain = maf_chain  # expose for debugging

    # 5) Compile with AdamW + gradient clipping + XLA
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clipnorm=clipnorm,
    )
    model.compile(optimizer=optimizer, jit_compile=True)
    return model



# RealNVP


def make_realnvp_fn(
    ndims: int,
    hidden_layers: list[int],
    weight_decay: float,
    min_ls: float,
    max_ls: float,
    clamp_log_scale: bool,
    name_prefix: str,
):
    """
    Returns shift_and_log_scale_fn(x, output_units, **kwargs) with unique layer names:
      - Dense(hidden_layers, relu) named {name_prefix}_dense_{i}
      - Dense(half*2) named {name_prefix}_dense_out
    """
    half = ndims // 2
    # build a fresh list of layers for this prefix
    layers = []
    for i, units in enumerate(hidden_layers):
        layers.append(tf.keras.layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name=f"{name_prefix}_dense_{i}"
        ))
    layers.append(tf.keras.layers.Dense(
        half * 2,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        name=f"{name_prefix}_dense_out"
    ))

    def fn(x, output_units, **kwargs):
        h = x
        for layer in layers:
            h = layer(h)
        shift, log_scale = tf.split(h, 2, axis=-1)
        if clamp_log_scale:
            log_scale = tf.clip_by_value(log_scale, min_ls, max_ls)
        return shift, log_scale

    return fn

def build_random_shuffle_realnvp_chain(
    ndims: int,
    num_bijectors: int,
    hidden_layers: list[int],
    weight_decay: float,
    clamp_log_scale: bool,
    min_log_scale: float,
    max_log_scale: float,
    seed: int | None = None 
) -> tfb.Chain:
    """[RealNVP→bi-perm→RealNVP→rnd-perm]×num_bijectors, drop final perm."""
    half = ndims // 2
    bi_perm = np.concatenate([np.arange(half, ndims), np.arange(0, half)]).astype(np.int32)

    rng = np.random.RandomState(seed)

    bijectors = []
    for k in range(num_bijectors):
        # first coupling (‘a’)
        bijectors.append(tfb.RealNVP(
            num_masked=half,
            shift_and_log_scale_fn=make_realnvp_fn(
                ndims, hidden_layers, weight_decay,
                min_log_scale, max_log_scale,
                clamp_log_scale,
                name_prefix=f"b{k}_a"
            ),
            name=f"realnvp_b{k}_a"
        ))
        bijectors.append(tfb.Permute(permutation=bi_perm, name=f"perm_bi_b{k}"))

        # second coupling (‘b’)
        bijectors.append(tfb.RealNVP(
            num_masked=half,
            shift_and_log_scale_fn=make_realnvp_fn(
                ndims, hidden_layers, weight_decay,
                min_log_scale, max_log_scale,
                clamp_log_scale,
                name_prefix=f"b{k}_b"
            ),
            name=f"realnvp_b{k}_b"
        ))
        rnd_perm = rng.permutation(ndims).astype(np.int32)
        bijectors.append(tfb.Permute(permutation=rnd_perm, name=f"perm_rnd_b{k}"))

    # drop the final permutation so we end on a RealNVP
    chain_list = list(reversed(bijectors[:-1]))
    return tfb.Chain(chain_list, name="random_shuffle_realnvp")

def build_realnvp_keras_model(
    num_flows: int,
    hidden_units: list[int],
    event_shape: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    clipnorm: float = 1.0,
    clamp_log_scale: bool = False,
    min_log_scale: float = -5.0,
    max_log_scale: float = 5.0,
    seed: int | None = None
) -> tf.keras.Model:
    """
    Keras Model wrapping RandomShuffle RealNVP flow (all tf.keras.layers
    with unique names).
    """
    x_in = tf.keras.Input(shape=(event_shape,), name="embeddings")

    bij = build_random_shuffle_realnvp_chain(
        ndims=event_shape,
        num_bijectors=num_flows,
        hidden_layers=hidden_units,
        weight_decay=weight_decay,
        clamp_log_scale=clamp_log_scale,
        min_log_scale=min_log_scale,
        max_log_scale=max_log_scale, 
        seed=seed
    )

    #base = tfd.MultivariateNormalDiag(
    #    loc=tf.zeros(event_shape),
    #    scale_diag=tf.ones(event_shape)
    #)

    base = tfd.MultivariateStudentTLinearOperator(
        df=2.0,                                        # heavier tails
        loc=tf.zeros(32),
        scale=tf.linalg.LinearOperatorIdentity(32)
    )
    
    flow = tfd.TransformedDistribution(base, bij)

    logp = tf.expand_dims(flow.log_prob(x_in), axis=-1)
    model = tf.keras.Model(x_in, logp, name="realnvp_random_shuffle")
    model.add_loss(-tf.reduce_mean(logp))
    model.flow = flow 
    opt = tf.keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clipnorm=clipnorm
    )
    model.compile(optimizer=opt, jit_compile=True)
    return model




class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, beta=0.3):
        super().__init__()
        # Encoder
        self.encoder_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(latent_dim * 2),
        ])
        # Decoder
        self.decoder_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(input_dim),
        ])
        self.latent_dim = latent_dim
        self.beta = beta  # KL divergence weight

    def encode(self, x):
        x = self.encoder_net(x)
        z_mean, z_logvar = tf.split(x, num_or_size_splits=2, axis=-1)
        return z_mean, z_logvar

    def reparameterize(self, z_mean, z_logvar):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps

    def decode(self, z):
        return self.decoder_net(z)

    def call(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_recon = self.decode(z)
        return x_recon

    def train_step(self, data):
        x = data  # only x is provided
        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encode(x)
            z = self.reparameterize(z_mean, z_logvar)
            x_recon = self.decode(z)
            # Reconstruction loss (MSE)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_recon), axis=-1))
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=-1))
            loss = recon_loss + self.beta*kl_loss
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
    
    def test_step(self, data):
        # exactly mirror train_step *without* the gradient tape
        x = data
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_recon = self.decode(z)

        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_recon), axis=-1))
        kl_loss    = -0.5 * tf.reduce_mean(tf.reduce_sum(
                         1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=-1))
        loss = recon_loss + self.beta * kl_loss

        # return a dict matching the names you track in train_step
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
