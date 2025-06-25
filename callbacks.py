
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions






class ParamLoggingCallback(tf.keras.callbacks.Callback):
    """
    Logs statistics of trainable parameters after each batch.
    """
    def __init__(self, log_every_n_batches=1):
        super().__init__()
        self.log_every_n = log_every_n_batches

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_every_n == 0:
            print(f"\n--- Batch {batch} parameter stats ---")
            for var in self.model.trainable_variables:
                values = var.numpy()
                name = var.name
                print(f"{name}: mean={np.mean(values):.4e}, std={np.std(values):.4e}, "
                      f"min={np.min(values):.4e}, max={np.max(values):.4e}")
            print("-------------------------------------")



class CatchAndDebugNaN(tf.keras.callbacks.Callback):
    """
    On NaN/Inf loss, prints:
      - Last good batch and breaking batch log_scale stats per MAF bijector.
      - Last good batch and breaking batch intermediate z stats after each bijector.
    """
    def __init__(self, train_dataset):
        super().__init__()
        self.train_dataset = train_dataset
        self.iterator = None
        self.current_batch = None

        # Snapshots for last good batch
        self.pre_logscales = None
        self.pre_intermediates = None
        self.last_good_batch = None

    def on_train_begin(self, logs=None):
        self.iterator = iter(self.train_dataset)

    def on_epoch_begin(self, epoch, logs=None):
        self.iterator = iter(self.train_dataset)

    def on_train_batch_begin(self, batch, logs=None):
        # fetch next batch data
        try:
            self.current_batch = next(self.iterator).numpy().astype(np.float32)
        except StopIteration:
            self.iterator = iter(self.train_dataset)
            self.current_batch = next(self.iterator).numpy().astype(np.float32)

    def on_train_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        if loss is None:
            return

        # 1) Propagate through flow: log_scales & intermediates
        '''
        z = self.current_batch
        logscales = []
        intermediates = []
        for bij in self.model.maf_chain.bijectors:
            if isinstance(bij, tfb.MaskedAutoregressiveFlow):
                fn = getattr(bij, 'shift_and_log_scale_fn',
                             getattr(bij, '_shift_and_log_scale_fn', None))
                if fn is not None:
                    params = fn(z)
                    D = params.shape[-1] // 2
                    logscales.append(params[..., D:])
            z = bij.forward(z).numpy()
            intermediates.append(z)
        '''
        z = self.current_batch
        logscales, intermediates = [], []
        for bij in self.model.maf_chain.bijectors:
            # if this is a MAF, record its log_scale and its output
            if isinstance(bij, tfb.MaskedAutoregressiveFlow):
                fn = getattr(bij, 'shift_and_log_scale_fn',
                             getattr(bij, '_shift_and_log_scale_fn'))
                params = fn(z)
                D = params.shape[-1] // 2
                logscales.append(params[..., D:])
                z = bij.forward(z).numpy()    # apply the MAF
                intermediates.append(z)       # record immediately after MAF
            else:
                # it’s a Permute (or other) — just forward, but don’t record
                z = bij.forward(z).numpy()

        # 2) Snapshot on finite loss
        if np.isfinite(loss):
            self.pre_logscales = logscales
            self.pre_intermediates = intermediates
            self.last_good_batch = batch
            return

        # 3) On NaN/Inf loss: print diagnostics
        print(f"\n Breaking Batch {batch}: loss = {loss}")

        # 3a) Last good log_scale stats
        if self.pre_logscales is not None:
            print(f"\n--- Last Good Batch {self.last_good_batch} log_scale stats ---")
            for idx, ls in enumerate(self.pre_logscales):
                arr = ls.numpy()
                print(f"Bijector {idx}: min={arr.min():.3e}, max={arr.max():.3e}, "
                      f"mean={arr.mean():.3e}, std={arr.std():.3e}")
        else:
            print("No last-good log_scale snapshot available.")

        # 3b) Breaking-batch log_scale stats
        print(f"\n--- Breaking Batch {batch} log_scale stats ---")
        for idx, ls in enumerate(logscales):
            arr = ls.numpy()
            print(f"Bijector {idx}: min={arr.min():.3e}, max={arr.max():.3e}, "
                  f"mean={arr.mean():.3e}, std={arr.std():.3e}")

        # 3c) Last good intermediate z stats
        if self.pre_intermediates is not None:
            print(f"\n--- Last Good Batch {self.last_good_batch} intermediate z stats ---")
            for idx, z_arr in enumerate(self.pre_intermediates):
                print(f"After Bijector {idx}: min={z_arr.min():.3e}, max={z_arr.max():.3e}, "
                      f"mean={z_arr.mean():.3e}, std={z_arr.std():.3e}")
        else:
            print("No last-good intermediate snapshot available.")

        # 3d) Breaking-batch intermediate z stats
        print(f"\n--- Breaking Batch {batch} intermediate z stats ---")
        for idx, z_arr in enumerate(intermediates):
            print(f"After Bijector {idx}: min={z_arr.min():.3e}, max={z_arr.max():.3e}, "
                  f"mean={z_arr.mean():.3e}, std={z_arr.std():.3e}")

        # 4) Halt
        self.model.stop_training = True
        print("Training halted due to NaN/Inf loss.")
