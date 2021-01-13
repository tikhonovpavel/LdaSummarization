"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division

import pickle

import gensim
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer

from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric

from models.reporter import Statistics

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

nltk.download('wordnet')

try:
    with open('f:/workspace/LdaSummarization/lda_model_large_2020_12_08.pkl', 'rb') as f:
        lda_model, tm_dictionary = pickle.load(f)
except FileNotFoundError:
    with open('/content/LdaSummarization/lda_model_large_2020_12_08.pkl', 'rb') as f:
        lda_model, tm_dictionary = pickle.load(f)

wn_lemmatizer = nltk.WordNetLemmatizer()

lda_topics = lda_model.show_topics(num_topics=-1, num_words=10)

topics_words = []
filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]

step_n = 0

for topic in lda_topics:
    print(topic)
    topics_words.append(preprocess_string(topic[1], filters))


def lemmatize(text):#lemmatize_stemming(text):
    return wn_lemmatizer.lemmatize(text, pos='v')  # stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize(token))
    return result


preprocessed_vocab = [preprocess(x) for x in tokenizer.vocab.keys()]
preprocessed_vocab = [x[0] if len(x) > 0 else None for x in preprocessed_vocab]

topics_words_indexes = [[
    next(index for index, word in enumerate(preprocessed_vocab) if word == x) for x in y
] for y in topics_words]


def abs_loss(generator, topical_output, symbols, vocab_size, device, train=True, label_smoothing=0.0):
    compute = NMTLossCompute(
        generator, topical_output, symbols, vocab_size,
        label_smoothing=label_smoothing if train else 0.0)
    compute.to(device)
    return compute



class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, topical_output, pad_id):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.topical_output = topical_output
        self.padding_idx = pad_id



    def _make_shard_state(self, batch, output,  attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        shard_state = self._make_shard_state(batch, output)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output,
                              shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = Statistics()
        shard_state = self._make_shard_state(batch, output)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

        self.cosine_embedding_loss = nn.CosineEmbeddingLoss()

    def pgloss(self, src, trg, reward):
        """
        Returns a policy gradient loss
        :param src: seq_len
        :param trg: seq_len
        :param reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
        :return loss: policy loss
        """

        srq_len, batch_size = src.size()

        out = self.forward(src, trg,
                           teacher_forcing_ratio=0.5).permute(1, 0, 2)  # batch * seq * vocab
        out = F.log_softmax(out, dim=2)
        target_onehot = F.one_hot(trg.permute(1, 0), self.encoder.input_dim).float()  # batch * seq * vocab
        pred = torch.sum(out * target_onehot, dim=-1)  # batch_size * seq_len
        pred = torch.sum(pred, dim=-1)
        loss = -torch.sum(pred * reward)

        return loss / batch_size

    def forward(self, output_param, target_param, generator, topical_output, src_topics):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        loss = 0

        for i in range(output_param.shape[0]):
            output = generator(output_param[i])
            target = target_param[i]

            model_prob = self.one_hot.repeat(target.size(0), 1)
            model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
            model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

            # ---- calculate reward ----
            output_text = tokenizer.convert_ids_to_tokens(output.argmax(1).tolist())

            output_bow_vector = tm_dictionary.doc2bow(preprocess(' '.join(output_text)))
            output_article_topic = sorted(lda_model[output_bow_vector], key=lambda tup: -1 * tup[1])

            target_topics_one_hot = torch.zeros(len(topics_words))
            output_topics_one_hot = torch.zeros(len(topics_words))

            top_target_topic = sorted(src_topics, key=lambda tup: -1 * tup[1])[0]
            target_topics_one_hot[top_target_topic[0]] = torch.FloatTensor([1])
            # for index, value in src_topics:
            #     target_topics_one_hot[index] = torch.FloatTensor([value])

            top_output_topic = sorted(output_article_topic, key=lambda tup: -1 * tup[1])[0]
            output_topics_one_hot[top_output_topic[0]] = torch.FloatTensor([1])
            # for index, value in output_article_topic:
            #     output_topics_one_hot[index] = torch.FloatTensor([value])

            # more divergence - less reward
            reward = -(F.kl_div(output_topics_one_hot, target_topics_one_hot) ** 2)
            # ---- end of reward calculation ----

            vanilla_loss = F.kl_div(output, model_prob, reduction='sum')

            rl_loss = torch.sum(output * reward, dtype=torch.float)
            #rl_loss = -0.05 * rl_loss

            rl_loss_percentage = 0.5

            # rl_loss * x = rl_loss_percentage * vanilla_loss  =>  x = rl_loss_percentage * vanilla_loss / rl_loss
            rl_coeff = float(rl_loss_percentage * vanilla_loss / rl_loss)

            loss += (vanilla_loss * (1 - rl_loss_percentage) + rl_loss * rl_coeff)

            global step_n
            if step_n % 1000 == 0:
                target_text = tokenizer.convert_ids_to_tokens(target.tolist())
                print('Loss: vanilla: {} ({}), rl: {} ({}), {}\nTarget:\n{}\n\nOutput:\n{}'.format(
                    vanilla_loss, (1 - rl_loss_percentage) * vanilla_loss, rl_loss, rl_loss * rl_coeff, loss, ' '.join(target_text), ' '.join(output_text)
                    ), '\n')

            step_n += 1

        return loss


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, topical_output, symbols, vocab_size,
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, topical_output, symbols['PAD'])
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)

        if label_smoothing > 0:
            criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            raise NotImplementedError()
            # criterion = nn.NLLLoss(
            #     ignore_index=self.padding_idx, reduction='sum'
            # )

        self.criterion = criterion

    def _make_shard_state(self, batch, output):
        return {
            "output": output,
            "target": batch.tgt[:,1:],
        }

    def _compute_loss(self, batch, output, target):
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output)
        gtruth =target.contiguous().view(-1)

        loss = self.criterion(output, target, self.generator, self.topical_output, batch.topics)

        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
