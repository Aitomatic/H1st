import abc
import math
import json
import os
import pprint
import sys
import shutil

# import tensorflow as tf
import tensorflow.compat.v1 as tf

import scipy.stats
import numpy as np

# from h1st.dl.util.logger import logger


import tensorflow_probability as tfp

tf.disable_v2_behavior()

tfd = tfp.distributions

layers = tf.keras.layers

_NAME_SCOPE = "h1st"


class BaseModelGraph(object):
    """
    Generic base model TF graph.

    Instantiate the model graph in TF for model training, evaluation or prediction,
    seting up the placeholders if necessary, combining embedding and actual features,
    then hooking up the input to the network topology depending on the `back_prop`, `compute_loss` and `apply_dropout` parameters.

    Args:
    input_size: number of features per time step
    target_size: how many target class
    back_prop: back-propagate to train the model weights, True for training, False for evaluation/prediction
    compute_loss: compute loss or not, should set to False for perform predictions, where labels are not available
    apply_dropout: apply dropouts or not, should set to False for prediction
    """

    def __init__(self,
                 input_size,
                 target_size,
                 config=None,
                 apply_dropout=True,
                 compute_loss=True,
                 back_prop=True,
                 input_tensors=None,
                 input_tensor_transform_fn=None):

        self.config = config

        self.input_size = input_size
        self.target_size = target_size

        self.apply_dropout = apply_dropout
        self.compute_loss = compute_loss
        self.back_prop = back_prop

        self.input_tensors = input_tensors
        self.input_tensor_transform_fn = input_tensor_transform_fn
        self._lr = tf.Variable(0.0, trainable=False)

    @property
    def lr(self):
        return self._lr

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def setup_input_placeholders_and_embedding_lookup(self,
                                                      input_size,
                                                      target_size,
                                                      config,
                                                      target_type=tf.float32,
                                                      with_time_axis=False,
                                                      apply_dropout=False,
                                                      input_tensors=None, input_tensor_transform_fn=None):
        """Setup input & target placeholders (if needed), then merge real-valued inputs with embedding lookups to return a single input tensor to the model.

        Args:
          target_type: target type, tf.int32 for classification and tf.float32 for regression.
          with_time_axis: specify the shape of input placeholder to be 2d of shape [batch_size x input_size] or 3D of shape [batch_size x num_steps x input_size] for time-series modeling.
          input_tensors: if not None, specifies already-created input tensors, else input/output placeholders will be created for feed-dict based training (default).
            Expected to be a dict mapping from names to TF tensors produced by a TF queue-reader input pipeline i.e. https://www.tensorflow.org/programmers_guide/reading_data#reading_from_files.
            The dict should have these mapping: {'input_data': input_tensors, 'time_invariant_input': time_invariant_input_tensors, 'targets': target_tensors}.
          input_tensor_transform_fn: optional, a function that transform input tensor before feeding into the model.
            The transformation is applied at the TF graph level and can be used in exported model for TF serving.

        Returns:
          A tuple of ([input_data, time_invariant_input, targets], model_input) where the first items are plaveholders and the second are the merged tensor
        """
        targets = None
        if input_tensors is None:
            if with_time_axis:
                input_data = tf.placeholder(tf.float32, [None, None, input_size])
                time_invariant_input = tf.placeholder(tf.int32, [None, None])
                if target_size:
                    targets = tf.placeholder(target_type, [None, None])
            else:
                input_data = tf.placeholder(tf.float32, [None, input_size])
                time_invariant_input = tf.placeholder(tf.int32, [None])
                if target_size:
                    targets = tf.placeholder(target_type, [None])
        else:
            input_data = input_tensors['input_data']
            time_invariant_input = input_tensors['time_invariant_input']
            if target_size:
                targets = input_tensors['targets']

        model_input = input_data
        if input_tensor_transform_fn:
            model_input = tf.Print(model_input, [model_input], "input_data pre input_tensor_transform_fn  = ",
                                   summarize=input_size, first_n=2)
            model_input = input_tensor_transform_fn(model_input)
        logger.debug("model_input = %s" % model_input)
        model_input = tf.Print(model_input, [model_input], "model_input = ", summarize=input_size, first_n=2)

        if config.embedding_size is not None:
            self.embeddings = embeddings = tf.get_variable("embeddings", [config.vocab_size, config.embedding_size],
                                                           initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, time_invariant_input)
            logger.debug("embedding_lookup = %s " % embed)

            model_input = tf.concat([model_input, embed], 1)
        else:
            model_input = model_input

        if apply_dropout and config.input_keep_prob < 1:
            model_input = tf.nn.dropout(model_input, config.input_keep_prob)
        logger.debug("model_input = %s" % model_input)

        return [input_data, time_invariant_input, targets], model_input

    @abc.abstractmethod
    def network_topology(self, inputs, input_size, target_size, apply_dropout):
        """Must be implemented by subclass.

        Should return the network outputs"""
        raise NotImplementedError("required by subclass")

    def save(self, session, checkpoint_path, global_step=0):
        """Save the model graph to disk using tf.train.Saver and also save the hyperparameter configuration to json"""
        # literally save everything in this session, have to get the right model name out
        # don't use interactively
        logger.info('saving to %s-%s' % (checkpoint_path, global_step))
        saver = tf.train.Saver(tf.global_variables())
        saver.save(session, checkpoint_path, global_step=global_step)

        config = self.config.__dict__

        # old method, not accurate
        # also doesn't work with zip import
        # config['class_name'] = qualname.qualname(self.__class__)

        # new method
        config['model_class_name'] = self.model_class_name

        config['input_size'] = self.input_size
        config['target_size'] = self.target_size

        # This sometimes caused serialization error when model trained on one machine (PENG worker) and loaded on another (Notebooks)
        # config['input_tensor_transform_fn'] = base64.b64encode(cloudpickle.dumps(self.input_tensor_transform_fn))

        dirname = os.path.dirname(checkpoint_path)
        with open(os.path.join(dirname, "model_config.json"), 'w') as f:
            json.dump(config, f)

    @classmethod
    def load(model_class, session, checkpoint_path, batch_size=None, name_scope="model", reuse=None, **kwargs):
        """Load model weights and hyper-parameter configuration from a checkpoint on disk"""
        logger.info("==== Load %s.ModelGraph from checkpoint %r" % (model_class.model_class_name, checkpoint_path))

        # assume sth like "run/lstm-aligned/checkpoints-30"
        dirname = os.path.dirname(checkpoint_path)
        with open(os.path.join(dirname, "model_config.json")) as f:
            params = json.load(f)

        try:
            assert params['model_class_name'] == model_class.model_class_name
            del params['model_class_name']
        except KeyError:
            # old method, not accurate
            # also doesn't work with zip import
            assert params['class_name'] == qualname.qualname(model_class)
            del params['class_name']

        input_size = params['input_size']
        target_size = params['target_size']
        del params['input_size']
        del params['target_size']

        input_tensor_transform_fn = kwargs.get("input_tensor_transform_fn")
        if 'input_tensor_transform_fn' in params:
            # if present in kwargs, then skip loading it from config
            # in case we want to specify input_tensor_transform_fn = None,
            # as cloudpickle might fail to unpickle the lambda.
            # XXX: this feature is a bit experimental.
            del params['input_tensor_transform_fn']
        # if input_tensor_transform_fn is not None:
        #   try:
        #     logger.debug(inspect.getsource(input_tensor_transform_fn))
        #   except Exception as e:
        #     pass
        logger.debug("input_tensor_transform_fn = %s" % input_tensor_transform_fn)

        if batch_size is not None:
            # special treatment
            params['batch_size'] = batch_size
        logger.debug(params)

        config = model_class.config_class(**params)
        logger.debug("==== Model parameters: %s" % pprint.pformat(config))

        kwargs['input_tensor_transform_fn'] = input_tensor_transform_fn

        with tf.variable_scope(name_scope, reuse=reuse):
            model = model_class(input_size=input_size,
                                target_size=target_size,
                                config=config,
                                **kwargs)

        saver = tf.train.Saver(tf.global_variables())

        # checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        # logger.debug("get_checkpoint_state = %s", checkpoint.model_checkpoint_path)
        saver.restore(session, checkpoint_path)

        return model

    @classmethod
    def export(model_class, session, checkpoint_path, dest, inputs_name="features", embed_name="embed_id",
               outputs_name="outputs", batch_size=None, name_scope=_NAME_SCOPE, reuse=None,
               input_tensor_transform_fn=None, output_transform_fn=None, output_probas=True):
        """Export the model graph with signature for TF serving.

        Args:
          * session: a tf.Session
          * dest: destination path on disk
          * batch_size: override config.batch_size, in case we want a different batch_size for eval/serving.
          * name_scope: the tf.variable_scope of the model variables
          * reuse: whether to reuse variables, set to True if the model variables already exist in the current tf.Session
        """
        from tensorflow.contrib.learn.python.learn.utils import export
        from tensorflow.contrib.session_bundle import exporter

        # assume sth like "run/lstm-aligned/checkpoints-30"
        # can also be from an already-exported model e.g. run/model/export
        try:
            global_step = checkpoint_path.split('/')[-1].split('-')[-1]
        except ValueError:
            global_step = 1

        m_pred = model_class.load(session, checkpoint_path,
                                  batch_size=batch_size,
                                  name_scope=name_scope,
                                  apply_dropout=False,
                                  compute_loss=False,
                                  reuse=reuse,
                                  input_tensor_transform_fn=input_tensor_transform_fn)

        if m_pred.config.embedding_size:
            sig = exporter.generic_signature({
                inputs_name: m_pred.input_data,
                embed_name: m_pred.time_invariant_input,
            })
        else:
            sig = exporter.generic_signature({inputs_name: m_pred.input_data})

        if output_probas:
            output_tensor = m_pred.output_probas
        else:
            output_tensor = m_pred.outputs
        if output_transform_fn:
            output_tensor = output_transform_fn(output_tensor)

        saver = tf.train.Saver()
        model_exporter = exporter.Exporter(saver)
        model_exporter.init(
            session.graph.as_graph_def(),
            named_graph_signatures={
                'inputs': sig,
                'outputs': exporter.generic_signature({outputs_name: output_tensor}
                                                      )})
        model_version_path = model_exporter.export(dest, tf.constant(global_step), session)

        shutil.copy(os.path.join(os.path.dirname(checkpoint_path), "model_config.json"), dest)

        logger.info("Exported to %r" % model_version_path)
        return model_version_path


class RegressorModelGraph(BaseModelGraph):
    """Base model graph for regressor with MAE loss function and Adam optimizer.
    """

    def __init__(self,
                 input_size,
                 target_size,
                 config=None,
                 apply_dropout=True,
                 compute_loss=True,
                 back_prop=True,
                 input_tensors=None,
                 input_tensor_transform_fn=None):
        super(RegressorModelGraph, self).__init__(input_size,
                                                  target_size,
                                                  config=config,
                                                  apply_dropout=apply_dropout,
                                                  compute_loss=compute_loss,
                                                  back_prop=back_prop,
                                                  input_tensors=input_tensors,
                                                  input_tensor_transform_fn=input_tensor_transform_fn)

        plchldrs, model_input = self.setup_input_placeholders_and_embedding_lookup(input_size=input_size,
                                                                                   target_size=target_size,
                                                                                   target_type=tf.float32,
                                                                                   config=config,
                                                                                   apply_dropout=apply_dropout,
                                                                                   input_tensors=input_tensors,
                                                                                   input_tensor_transform_fn=input_tensor_transform_fn)
        self._input_data, self._time_invariant_input, self._targets = plchldrs

        if config.embedding_size:
            outputs = self.network_topology(model_input, input_size + config.embedding_size, target_size, apply_dropout)
        else:
            outputs = self.network_topology(model_input, input_size, target_size, apply_dropout)

        target_size = target_size
        logger.debug("target_size = %s" % target_size)

        outputs = tf.reshape(outputs, [-1, target_size])
        outputs = tf.Print(outputs, [outputs], "outputs = ", summarize=24, first_n=5)
        self.outputs = outputs

        if not compute_loss:
            return

        targets = tf.reshape(self.targets, [-1])
        outputs = tf.reshape(self.outputs, [-1])
        cost = self.cost_function_def(targets, outputs)

        if config.scale_l1 > 0 and config.scale_l2 > 0:
            weights = tf.trainable_variables()
            reg = l1_l2_regularizer(scale_l1=config.scale_l1, scale_l2=config.scale_l2)
            penalty = tf.contrib.layers.apply_regularization(reg, weights)
            cost = cost + penalty
        logger.debug("cost = %s" % cost)

        cost_summary = tf.summary.scalar("cost", cost)
        cost = tf.Print(cost, [cost], "cost = ", summarize=24, first_n=5)
        self._cost = cost

        self.summaries = tf.summary.merge([cost_summary])

        if not back_prop:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        self._train_op = self.train_op_def(cost)

    @property
    def input_data(self):
        """The tensor Variable representing the neural network input data .
        """
        return self._input_data

    @property
    def time_invariant_input(self):
        return self._time_invariant_input

    @property
    def targets(self):
        return self._targets

    @property
    def cost(self):
        """The tensor Variable representing the neural network cost function .
        """
        return self._cost

    @property
    def train_op(self):
        """The tensor operation representing a single model training mini-batch update .
        """
        return self._train_op

    def train_batch(self, session, batch):
        feeds = {self.input_data: batch['x'],
                 self.time_invariant_input: batch['entity_embed_id'],
                 self.targets: batch['y']}

        loss, _, summary = session.run([self.cost, self.train_op, self.summaries], feed_dict=feeds)

        return loss, summary

    def batch_loss(self, session, batch):
        assert self.apply_dropout == False, "dropout is being applied for training, create a model instance with apply_dropout=False to compute loss"

        feeds = {self.input_data: batch['x'],
                 self.time_invariant_input: batch['entity_embed_id'],
                 self.targets: batch['y']}

        loss, summary = session.run([self.cost, self.summaries], feed_dict=feeds)

        return loss, summary

    def cost_function_def(self, targets, outputs):
        """Define the loss function for optimizaiton as MAE: tf.reduce_mean(tf.abs(targets - outputs)).
        Note that L1/L2 regularization will be added on top also.
        """
        return tf.reduce_mean(tf.abs(targets - outputs))

    def train_op_def(self, cost):
        # train_op = tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=True).minimize(self.cost)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        return train_op

    def predict_batch(self, session, batch):
        """Return regression outputs for given batch,
        """
        assert self.apply_dropout is False, "Should not make predictions from model instance when apply_dropout is True."

        feeds = {self.input_data: batch['x'],
                 self.time_invariant_input: batch['entity_embed_id']}

        outputs = session.run(self.outputs, feed_dict=feeds)

        return outputs


class ClassifierModelGraph(BaseModelGraph):
    """Base model graph for classifier with cross entropy softmax loss function and Adam optimizer.
    Also defines self.output_probas as a tensor in addition to the raw logits self.outputs.
    """

    def __init__(self,
                 input_size,
                 target_size,
                 config=None,
                 apply_dropout=True,
                 compute_loss=True,
                 back_prop=True,
                 input_tensors=None,
                 input_tensor_transform_fn=None):
        super(ClassifierModelGraph, self).__init__(input_size,
                                                   target_size,
                                                   config=config,
                                                   apply_dropout=apply_dropout,
                                                   compute_loss=compute_loss,
                                                   back_prop=back_prop,
                                                   input_tensors=input_tensors,
                                                   input_tensor_transform_fn=input_tensor_transform_fn)

        plchldrs, model_input = self.setup_input_placeholders_and_embedding_lookup(input_size=input_size,
                                                                                   target_size=target_size,
                                                                                   target_type=tf.int32,
                                                                                   config=config,
                                                                                   apply_dropout=apply_dropout,
                                                                                   input_tensors=input_tensors,
                                                                                   input_tensor_transform_fn=input_tensor_transform_fn)
        self._input_data, self._time_invariant_input, self._targets = plchldrs

        if config.embedding_size:
            outputs = self.network_topology(model_input, input_size + config.embedding_size, target_size,
                                            self.apply_dropout)
        else:
            outputs = self.network_topology(model_input, input_size, target_size, self.apply_dropout)

        target_size = target_size
        logger.debug("target_size = %s" % target_size)

        probas = tf.nn.softmax(outputs)
        self._output_probas = tf.Print(probas, [probas], "output_probas = ", summarize=24, first_n=5)

        outputs = tf.reshape(outputs, [-1, target_size])
        outputs = tf.Print(outputs, [outputs], "outputs = ", summarize=24, first_n=5)
        self.outputs = outputs

        if not compute_loss:
            return

        targets = tf.reshape(self.targets, [-1])
        with tf.control_dependencies([self.outputs, self.output_probas]):
            cost = self.cost_function_def(targets, self.outputs)

        if config.scale_l1 > 0 and config.scale_l2 > 0:
            weights = tf.trainable_variables()
            reg = l1_l2_regularizer(scale_l1=config.scale_l1, scale_l2=config.scale_l2)
            penalty = tf.contrib.layers.apply_regularization(reg, weights)
            cost = cost + penalty
        logger.debug("cost = %s" % cost)

        cost_summary = tf.summary.scalar("cost", cost)
        cost = tf.Print(cost, [cost], "cost = ", summarize=24, first_n=5)
        self._cost = cost

        self.summaries = tf.summary.merge([cost_summary])

        if not back_prop:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        self._train_op = self.train_op_def(cost)

    @property
    def input_data(self):
        """The tensor Variable representing the neural network input data .
        """
        return self._input_data

    @property
    def time_invariant_input(self):
        return self._time_invariant_input

    @property
    def output_probas(self):
        """The tensor Variable representing the neural network output probabilities .
        """
        return self._output_probas

    @property
    def targets(self):
        return self._targets

    @property
    def cost(self):
        """The tensor Variable representing the neural network cost function .
        """
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        """The tensor operation representing a single model training mini-batch update .
        """
        return self._train_op

    def cost_function_def(self, targets, outputs):
        """Define the loss function for optimizaiton as cross-entropy softmax: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=outputs)).
        Note that L1/L2 regularization will be added on top also.
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=outputs))

    def train_op_def(self, cost):
        # train_op = tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=True).minimize(self.cost)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        return train_op

    def train_batch(self, session, batch):
        feeds = {self.input_data: batch['x'],
                 self.time_invariant_input: batch['entity_embed_id'],
                 self.targets: batch['y']}

        loss, _, summary = session.run([self.cost, self.train_op, self.summaries], feed_dict=feeds)

        return loss, summary

    def batch_loss(self, session, batch):
        assert self.apply_dropout == False, "dropout is being applied for training, create a model instance with apply_dropout=False to compute loss"

        feeds = {self.input_data: batch['x'],
                 self.time_invariant_input: batch['entity_embed_id'],
                 self.targets: batch['y']}

        loss, summary = session.run([self.cost, self.summaries], feed_dict=feeds)

        return loss, summary

    def predict_batch(self, session, batch):
        assert self.apply_dropout is False, "Should not make predictions from model graph created apply_dropout is True."

        feeds = {self.input_data: batch['x'],
                 self.time_invariant_input: batch['entity_embed_id']}

        outputs, output_probas = session.run([self.outputs, self.output_probas], feed_dict=feeds)

        return outputs, output_probas


def softmax(x):
    """Compute softmax values for each sets of scores in x in numpy.
    """

    """
    >>> softmax([[1, -2, 3], [3, 4, 5]])
    array([[ 0.11849965,  0.00589975,  0.8756006 ],
           [ 0.09003057,  0.24472847,  0.66524096]])
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return (e_x / e_x.sum(axis=1, keepdims=True))


def leaky_relu(x, alpha=0.001):
    """
    Ref: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, He et al https://arxiv.org/pdf/1502.01852.pdf

    More practically, this helps with "dying ReLU" problems.
    http://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks
    """
    return tf.maximum(alpha * x, x)


def maxnorm_regularizer(threshold, axes=1, name="maxnorm", collection="maxnorm"):
    """This goes hand-in-hand with dropouts.

    Ref: Dropout: A Simple Way to Prevent Neural Networks from Overfitting, Srivastava et al http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

    Usage: http://stackoverflow.com/questions/34934303/renormalize-weight-matrix-using-tensorflow
    """

    def maxnorm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None  # there is no regularization loss term

    return maxnorm


class MDN(object):
    """
    Building blocks for Mixture Density Network.

    Ref:
      1. Mixture Density Networks, Bishop 94, http://research.microsoft.com/en-us/um/people/cmbishop/downloads/Bishop-NCRG-94-004.pdf
      2. http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/
    """
    oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)  # normalisation factor for gaussian

    @staticmethod
    def get_mixture_coef(output, KMIX, output_scale):
        out_pi, out_sigma, out_mu = tf.split(output, KMIX, axis=1)

        # out_pi = tf.Print(out_pi, [out_pi], "out_pi raw =", summarize=100, first_n=100)

        # Original impl
        # max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
        # out_pi = tf.subtract(out_pi, max_pi)
        # out_pi = tf.exp(out_pi)
        # normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
        # out_pi = tf.multiply(normalize_pi, out_pi)

        # NB(aht): "batch-normalize" the pi's otherwise
        # softmax will cause one of the pi to be 1.0 and all others 0.0,
        # for which no further training can happen
        out_pi = tf.abs(out_pi)
        out_pi = out_pi / tf.reduce_mean(out_pi, 1, keep_dims=True)
        out_pi = tf.nn.softmax(out_pi)
        print("out_pi = %s" % out_pi)

        out_pi = tf.Print(out_pi, [out_pi], "out_pi softmax =", summarize=100, first_n=100)

        output_scale = float(output_scale)
        out_sigma = tf.maximum(tf.abs(out_sigma) * output_scale, 1e-6)  # ensure positive
        out_mu = out_mu * output_scale

        print("out_sigma = %s" % out_sigma)
        print("out_mu = %s" % out_mu)

        with tf.control_dependencies([tf.assert_positive(out_sigma)]):
            out_sigma = tf.Print(out_sigma, [out_sigma], "out_sigma =", summarize=100, first_n=100)
        out_mu = tf.Print(out_mu, [out_mu], "out_mu =", summarize=100, first_n=100)

        return out_pi, out_sigma, out_mu

    @staticmethod
    def tf_normal(y, mu, sigma):
        result = tf.subtract(mu, tf.expand_dims(y, axis=1))
        result = tf.multiply(result, tf.reciprocal(sigma))
        result = -tf.square(result) / 2
        result = tf.multiply(tf.exp(result), tf.reciprocal(sigma)) * MDN.oneDivSqrtTwoPI
        return result

    @staticmethod
    def loss_function(out_pi, out_sigma, out_mu, y):
        result = MDN.tf_normal(y, out_mu, out_sigma)
        result = tf.Print(result, [result], "density =", summarize=100, first_n=100)
        result = tf.multiply(result, out_pi)
        result = tf.Print(result, [result], "density * out_pi =", summarize=100, first_n=100)
        # tf.histogram_summary("density * out_pi", result)

        result = tf.reduce_sum(result, 1, keep_dims=True)
        result = tf.Print(result, [result], "sum(density * out_pi) =", summarize=100, first_n=100)
        # tf.histogram_summary("sum(density * out_pi)", result)

        result = tf.maximum(result, 1e-20)  # avoid taking log of 0
        with tf.control_dependencies([tf.assert_positive(result)]):
            result = -tf.log(result)
        print("- log lik = %s" % result)
        return tf.reduce_mean(result)

    @staticmethod
    def neg_log_lik(ytrue, out_pi, out_mu, out_sigma, KMIX):
        """MDN loss function in numpy"""
        d = []
        for k in range(KMIX):
            d.append(scipy.stats.norm.pdf(ytrue, loc=out_mu[:, k], scale=out_sigma[:, k]))
        d = np.array(d).T
        return -np.log(np.mean(np.sum(d * out_pi, axis=1)))


def l1_l2_regularizer(scale_l1=1.0, scale_l2=1.0, scope=None):
    """Returns a function that can be used to apply L1 L2 regularizations.

    Args:
      scale_l1: A scalar multiplier `Tensor` for L1 regularization.
      scale_l2: A scalar multiplier `Tensor` for L2 regularization.
      scope: An optional scope name.

    Returns:
      A function with signature `l1_l2(weights)` that applies a weighted sum of
      L1 L2  regularization.

    Raises:
      ValueError: If scale is negative or if scale is not a float.

    Backport from TF master branch.
    """
    scope = scope or 'l1_l2_regularizer'
    return tf.contrib.layers.sum_regularizer([tf.contrib.layers.l1_regularizer(scale_l1),
                                              tf.contrib.layers.l2_regularizer(scale_l2)],
                                             scope=scope)


class _Dense_BN(tf.keras.Model):
    def __init__(self, networks, base_name):
        super(_Dense_BN, self).__init__()
        self.networks_list = []
        for idx, layer in enumerate(networks):
            self.dense = layers.Dense(
                layer, kernel_initializer="glorot_uniform", name=base_name + "_dense" + str(idx))
            self.networks_list.append(self.dense)
            self.bn = layers.BatchNormalization(name=base_name + "_bn" + str(idx))
            self.networks_list.append(self.bn)

    def call(self, input_tensor, training=False):
        x = self.networks_list[0](input_tensor)
        x = self.networks_list[1](x, training=training)
        x = tf.nn.tanh(x)
        for i in range(2, len(self.networks_list), 2):
            x = self.networks_list[i](x)
            x = self.networks_list[i + 1](x, training=training)
            x = tf.nn.tanh(x)  # tf.nn.elu(x)
        return x


class _Dense(tf.keras.Model):
    def __init__(self, networks, base_name):
        super(_Dense, self).__init__()
        self.networks_list = []
        for idx, layer in enumerate(networks):
            self.dense = layers.Dense(
                layer, kernel_initializer="glorot_uniform", name=base_name + "_dense" + str(idx))
            self.networks_list.append(self.dense)

    def call(self, input_tensor, training=False):
        x = self.networks_list[0](input_tensor)
        x = tf.nn.tanh(x)
        for i in range(1, len(self.networks_list)):
            x = self.networks_list[i](x)
            x = tf.nn.tanh(x)  # tf.nn.elu(x)
        return x


class VAE(tf.keras.Model):
    def __init__(self, encoding_networks, decoding_networks, n_input, n_z, n_rank=0, bn=False, alpha=0, drop_rate=0.0):
        super(VAE, self).__init__(name="vae")

        self.alpha = alpha
        self.drop_rate = drop_rate
        self.n_input = n_input
        self.n_rank = n_rank

        if bn:
            self.encoder = _Dense_BN(encoding_networks, "encoder")
        else:
            self.encoder = _Dense(encoding_networks, "encoder")
        self.get_z_mean = layers.Dense(
            n_z, kernel_initializer="glorot_uniform", name="z_mean")
        self.get_z_logvar = layers.Dense(
            n_z, kernel_initializer="glorot_uniform", name="z_sigma")
        self.z_prior_pdf = tfd.MultivariateNormalDiag(tf.zeros(n_z), tf.ones(n_z))

        if bn:
            self.decoder = _Dense_BN(decoding_networks, "decoder")
        else:
            self.decoder = _Dense(decoding_networks, "decoder")
        self.get_x_mean = layers.Dense(
            self.n_input, kernel_initializer="glorot_uniform", name="x_mean")
        self.get_x_logvar = layers.Dense(
            self.n_input, kernel_initializer="glorot_uniform", name="x_sigma")

        self.dropout_encoder = layers.Dropout(rate=self.drop_rate)
        self.dropout_decoder = layers.Dropout(rate=self.drop_rate)

        # LowRank Approximation
        if self.n_rank:
            self.get_scale_perturb_factor = layers.Dense(
                n_input * n_rank, kernel_initializer="glorot_uniform", name="get_scale_perturb_factor")
            self.get_scale_perturb_diag = layers.Dense(
                n_rank, kernel_initializer="glorot_uniform", name="get_scale_perturb_diag")

    def call(self, input_tensor, num_sample, training=False):
        z_post_pdf = self.get_z_post_pdf(input_tensor, training=training)
        log_prob_list = []
        for _ in range(num_sample):
            z_sample = z_post_pdf.sample()
            # epsilon = self.z_prior_pdf.sample()
            # z_sample = z_mean + z_sigma * epsilon
            x_pdf = self.get_x_pdf(z_sample, training=training)
            x_log_prob = x_pdf.log_prob(input_tensor)
            log_prob_list.append(x_log_prob)
        log_prob_vector = tf.stack(log_prob_list)
        return tf.reduce_mean(log_prob_vector, axis=0)

    def get_z_param(self, input_tensor, training=False):
        temp = self.encoder(input_tensor, training=training)
        dropout = self.dropout_encoder(temp, training=training)
        z_mean = self.get_z_mean(dropout)
        z_logvar = self.get_z_logvar(dropout)
        return [z_mean, z_logvar]

    def get_z_post_pdf(self, input_tensor, training=False):
        z_mean, z_logvar = self.get_z_param(input_tensor, training=training)
        z_sigma = tf.sqrt(tf.exp(z_logvar))
        z_post_pdf = tfd.MultivariateNormalDiag(z_mean, z_sigma)
        return z_post_pdf

    def get_x_param_from_z(self, z_sample, training=False):
        temp = self.decoder(z_sample, training=training)
        dropout = self.dropout_decoder(temp, training=training)
        x_mean = self.get_x_mean(dropout)
        x_logvar = self.get_x_logvar(dropout)
        return [x_mean, x_logvar]

    def get_x_pdf(self, z_sample, training=False):
        temp = self.decoder(z_sample, training=training)
        dropout = self.dropout_decoder(temp, training=training)
        x_mean = self.get_x_mean(dropout)
        x_logvar = self.get_x_logvar(dropout)
        x_sigma = tf.sqrt(tf.exp(x_logvar))

        if self.n_rank:
            m_scale = tf.nn.softplus(self.get_scale_perturb_diag(dropout)) + 1e-10
            u_scale = tf.nn.softplus(self.get_scale_perturb_factor(dropout)) + 1e-10
            u_scale = tf.reshape(u_scale, shape=[-1, self.n_input, self.n_rank])
            x_pdf = tfd.MultivariateNormalDiagPlusLowRank(
                loc=x_mean,
                scale_diag=x_sigma + self.alpha,
                scale_perturb_factor=u_scale,
                scale_perturb_diag=m_scale)
        else:
            x_pdf = tfd.MultivariateNormalDiag(x_mean, x_sigma + self.alpha)

        return x_pdf

    def get_x_sigma(self, z_sample, training=False):
        temp = self.decoder(z_sample, training=training)
        dropout = self.dropout_decoder(temp, training=training)
        x_logvar = self.get_x_logvar(dropout)
        x_sigma = tf.sqrt(tf.exp(x_logvar))
        m_scale = tf.nn.softplus(self.get_scale_perturb_diag(dropout)) + 1e-10
        u_scale = tf.nn.softplus(self.get_scale_perturb_factor(dropout)) + 1e-10
        u_scale = tf.reshape(u_scale, shape=[-1, self.n_input, self.n_rank])
        final_sigma = x_sigma + tf.linalg.diag_part(tf.matmul(tf.matmul(
            u_scale,
            tf.linalg.diag(m_scale)),
            tf.linalg.transpose(u_scale)))
        return final_sigma

    def get_prob_for_individual_sensor(self, input_tensor, num_sensors, num_sample, window_size=1, training=False):
        prob_for_each_sensor_list = []
        z_post_pdf = self.get_z_post_pdf(input_tensor, training=training)
        for _ in range(num_sample):
            # epsilon = self.z_prior_pdf.sample()
            # z_sample = z_mean + z_sigma * epsilon
            z_sample = z_post_pdf.sample()
            x_mean, x_logvar = self.get_x_param_from_z(z_sample, training=training)
            if self.n_rank:
                x_sigma = self.get_x_sigma(z_sample, training=training)
            else:
                x_sigma = tf.sqrt(tf.exp(x_logvar))
            normal_dist = tf.distributions.Normal(loc=x_mean, scale=x_sigma)
            prob_for_each_sensor = normal_dist.prob(input_tensor)
            prob_for_each_sensor_list.append(prob_for_each_sensor)
        prob_for_each_sensor_vector = tf.stack(prob_for_each_sensor_list)
        prob_for_each_sensor_final = tf.reduce_mean(prob_for_each_sensor_vector, axis=0)
        if window_size > 1:
            reshaped_prob = tf.reshape(prob_for_each_sensor_final, [-1, window_size, num_sensors])
            prob_for_each_sensor_final = tf.reduce_mean(reshaped_prob, axis=[1])
        return prob_for_each_sensor_final

    def get_analytic_loss_n_mae(self, input_tensor, beta=1, training=True):
        def log_likelihood(input_val, mean, logvar, raxis=1):
            log2pi = tf.log(2. * np.pi)
            return tf.reduce_sum(
                -.5 * ((input_val - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
                axis=raxis)

        def kl_divergence(mean, logvar, raxis=1):
            return tf.reduce_sum(
                .5 * (mean ** 2. + tf.exp(logvar) - logvar - 1),
                axis=raxis)

        # Get loss
        z_post_pdf = self.get_z_post_pdf(input_tensor, training=training)
        z_sample = z_post_pdf.sample()
        # epsilon = self.z_prior_pdf.sample()
        # z_sample = z_mean + z_sigma * epsilon
        x_mean, x_logvar = self.get_x_param_from_z(z_sample, training=training)
        ELBO = log_likelihood(input_tensor, x_mean, x_logvar) \
               - beta * kl_divergence(z_mean, z_logvar)
        loss = tf.reduce_mean(-ELBO)

        # Get MAE
        mae, mae_update_op = tf.metrics.mean_absolute_error(input_tensor, x_mean)

        return [loss, mae, mae_update_op]

    def get_mc_loss_n_mae(self, input_tensor, beta=1, training=True):
        # Get loss
        z_post_pdf = self.get_z_post_pdf(input_tensor, training=training)
        z_sample = z_post_pdf.sample()
        # epsilon = self.z_prior_pdf.sample()
        # z_sample = z_mean + z_sigma * epsilon
        x_mean, _ = self.get_x_param_from_z(z_sample, training=training)
        x_pdf = self.get_x_pdf(z_sample, training=training)
        x_log_prob = x_pdf.log_prob(input_tensor)
        ELBO = x_log_prob - beta * (z_post_pdf.log_prob(z_sample) - \
                                    self.z_prior_pdf.log_prob(z_sample))
        loss = tf.reduce_mean(-ELBO)

        # Get MAE
        mae, mae_update_op = tf.metrics.mean_absolute_error(input_tensor, x_mean, name="my_mae")

        return [loss, mae, mae_update_op]
