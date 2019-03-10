import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=300000,
        iters_per_checkpoint=5000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,

        n_mel_channels=769, # It meant as maximum length of ema output frame(I'm thinking it's no of outcomes, not sure if that's what it really means)


        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=110,

        # Encoder parameters
        encoder_kernel_size=3,
        encoder_n_convolutions=2,
        encoder_embedding_dim=110,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=40,
        prenet_dim=32,
        max_decoder_steps=12,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=32,
        attention_dim=15,

        # Location Layer parameters
        attention_location_n_filters=16,
        attention_location_kernel_size=15,

        # Mel-post processing network parameters
        postnet_embedding_dim=64,
        postnet_kernel_size=3,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=30,
        mask_padding=False # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams