import tensorflow as tf

default_params = {
  "data-dir":                   "./data/simple-examples/data",
  "valid-file":                 "ptb.valid.txt",
  "test-file":                  "ptb.test.txt",
  "train-file":                 "ptb.train.txt",
  "train-dir":                  "./results/reverse",
  "max-train-data-size":        0,
  "mini-batch-size":            64,
  "keepout":                    0.8,
  "learning-rate":              0.5,
  "learning-rate-decay-factor": 0.99,
  "max-gradient-norm":          5.0,
  "hidden-size":                64, #40K rows ~ (64^2)*4 (i/o/f/v)
  "num-layers":                 2,
  "max-epochs":                 30,
  "anneal-epochs":              2,
  "early-stop":                 4,
  "steps-per-loss-report":      128,
  "decode":                     False,
  "force-fresh-start":          False,
  "self-test":                  False,
}

tf.app.flags.DEFINE_string("data_dir", default_params["data-dir"],
                           "Data directory")
tf.app.flags.DEFINE_string("valid_file", default_params["valid-file"],
                           "Data file for validation.")
tf.app.flags.DEFINE_string("test_file", default_params["test-file"],
                           "Data file for test.")
tf.app.flags.DEFINE_string("train_file", default_params["train-file"],
                           "Data file for training.")
tf.app.flags.DEFINE_string("train_dir", default_params["train-dir"],
                           "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size",
                            default_params["max-train-data-size"],
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("mini_batch_size",
                            default_params["mini-batch-size"],
                            "Mini Batch size to use during training.")
tf.app.flags.DEFINE_float("keepout",
                          default_params["keepout"],
                          "Keepout used at the output layer of the model.")
tf.app.flags.DEFINE_float("learning_rate",
                          default_params["learning-rate"],
                          "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor",
                          default_params["learning-rate-decay-factor"],
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm",
                          default_params["max-gradient-norm"],
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("hidden_size", default_params["hidden-size"],
                            "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", default_params["num-layers"],
                            "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_epochs", default_params["max-epochs"],
  "# epochs after which training stops.")
tf.app.flags.DEFINE_integer("anneal_epochs", default_params["anneal-epochs"],
  "# epochs when learning rate reduced if no training loss improvement.")
tf.app.flags.DEFINE_integer("early-stop", default_params["early-stop"],
  "# epochs when training terminated if no validation loss improvement.")
tf.app.flags.DEFINE_integer(
  "steps_per_loss_report",
  default_params["steps-per-loss-report"],
  "# steps in an epoch when average training loss is reported.")
tf.app.flags.DEFINE_boolean("decode", default_params["decode"],
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean(
  "force_fresh_start", default_params["force-fresh-start"],
  "Initialize all parameters and force fresh start if this is set to True.")
tf.app.flags.DEFINE_boolean("self_test", default_params["self-test"],
                            "Run a self-test if this is set to True.")
FLAGS = tf.app.flags.FLAGS
