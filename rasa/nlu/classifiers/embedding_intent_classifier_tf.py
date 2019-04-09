# create by fanfan on 2019/3/27 0027
import io
import logging
import os
from tqdm import tqdm
from typing import List,Text,Any,Optional,Dict


from rasa.nlu.classifiers import INTENT_RANKING_LENGTH
from rasa.nlu.components import Component
from multiprocessing import cpu_count
import numpy as np

import pickle

logger = logging.getLogger(__name__)

import tensorflow as tf
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message

from rasa.nlu.classifiers.tf_models import classify_cnn_model,base_classify_model
from rasa.nlu.utils.tfrecord_api import _int64_feature,_bytes_feature
import os

_START_VOCAB = ['_PAD', '_GO', "_EOS", '<UNK>']
class EmbeddingIntentClassifierTf(Component):
    name = 'intent_classifier_tf_embedding'

    provides = ['intent','intent_ranking']

    requires = ['text_features']

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["tensorflow"]

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                'Failed to import `tensorflow`. '
                'Please install `tensorflow`. '
                'For example with `pip install tensorflow`.')

    def __init__(self,component_config=None,vocabulary_list=None,intent_list=None,sess=None,input_node=None,output_node=None):
        self._check_tensorflow()
        super(EmbeddingIntentClassifierTf,self).__init__(component_config)

        self.vocabulary_list = vocabulary_list
        self.intent_list = intent_list
        self.sess = sess
        self.input_node = input_node
        self.output_node = output_node

        if self.intent_list != None:
            self.inv_intent_dict = {index:key for index,key in enumerate(intent_list)}
        if self.vocabulary_list != None:
            self.vocabulary = {key:index for index,key in enumerate(self.vocabulary_list)}

    # training data helpers:
    @staticmethod
    def _create_intent_dict(training_data):
        distinct_intents = set([example.get('intent') for example in training_data.intent_examples])
        return {intent:idx for idx,intent in enumerate(sorted(distinct_intents))},sorted(distinct_intents)

    @staticmethod
    def _create_vocab_dict(training_data,min_freq=3):
        vocab = {}
        for tokens in training_data.intent_examples:
            real_tokens = [token.text for token in tokens.data.get('tokens')]
            for word in real_tokens:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab = {key: value for key, value in vocab.items() if value >= min_freq}
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        vocab_dict = {key:index for index,key in enumerate(vocab_list)}
        return vocab_dict,vocab_list

    @staticmethod
    def pad_sentence(sentence, max_sentence,vocabulary):
        '''
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

        参数：
        - sentence
        '''
        UNK_ID = vocabulary.get('<UNK>')
        PAD_ID = vocabulary.get('_PAD')
        sentence_batch_ids = [vocabulary.get(w, UNK_ID) for w in sentence]
        if len(sentence_batch_ids) > max_sentence:
            sentence_batch_ids = sentence_batch_ids[:max_sentence]
        else:
            sentence_batch_ids = sentence_batch_ids + [PAD_ID] * (max_sentence - len(sentence_batch_ids))

        if max(sentence_batch_ids) == 0:
            print(sentence_batch_ids)
        return sentence_batch_ids




    def make_tfrecord_files(self, training_data, intent_dict,classify_config):
        X = [[token.text for token in e.data.get('tokens')] for e in training_data.intent_examples]
        intents_for_X = [intent_dict[e.get('intent')] for e in training_data.intent_examples]
        self.vocabulary,self.vocab_list = self._create_vocab_dict(training_data,classify_config.min_freq)

        # tfrecore 文件写入
        tfrecord_save_path = os.path.join(classify_config.save_path,"train.tfrecord")
        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_save_path)
        for sentence,intent in zip(X,intents_for_X):
            sentence_ids = self.pad_sentence(sentence,classify_config.max_sentence_length,self.vocabulary)
            #sentence_ids_string = np.array(sentence_ids).tostring()
            train_feature_item = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(intent),
                'sentence':_int64_feature(sentence_ids,need_list=False)
            }))
            tfrecord_writer.write(train_feature_item.SerializeToString())
        tfrecord_writer.close()


    def input_fn(self, classify_config, shuffle_num, mode,epochs):
        """
         build tf.data set for input pipeline

        :param classify_config: classify config dict
        :param shuffle_num: type int number , random select the data
        :param mode: type string ,tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
        :return: set() with type of (tf.data , and labels)
        """
        def parse_single_tfrecord(serializer_item):
            features = {
                'label': tf.FixedLenFeature([],tf.int64),
                'sentence' : tf.FixedLenFeature([classify_config.max_sentence_length],tf.int64)
            }

            features_var = tf.parse_single_example(serializer_item,features)

            labels = tf.cast(features_var['label'],tf.int64)
            #sentence = tf.decode_raw(features_var['sentence'],tf.uint8)
            sentence = tf.cast(features_var['sentence'],tf.int64)
            return sentence,labels



        tf_record_filename = os.path.join(classify_config.save_path,'train.tfrecord')
        if not os.path.exists(tf_record_filename):
            raise FileNotFoundError("tfrecord not found")
        tf_record_reader = tf.data.TFRecordDataset(tf_record_filename)


        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = tf_record_reader.map(parse_single_tfrecord).shuffle(shuffle_num).batch(classify_config.batch_size).repeat(epochs)
        else:
            dataset = tf_record_reader.map(parse_single_tfrecord).batch(classify_config.batch_size)
        iterator = dataset.make_one_shot_iterator()
        data, labels = iterator.get_next()
        data = tf.reshape(data,[-1,classify_config.max_sentence_length])
        return data, labels

    def train(self,training_data,cfg=None,**kwargs):
        classify_config = classify_cnn_model.CNNConfig()
        if kwargs['project'] == None:
            projectName = 'default'
        else:
            projectName = kwargs['project']
        classify_config.save_path = os.path.join(kwargs['path'],projectName,kwargs['fixed_model_name'])
        if not os.path.exists(classify_config.save_path):
            os.makedirs(classify_config.save_path)
        self.intent_dict,self.intent_list = self._create_intent_dict(training_data)
        if len(self.intent_dict) < 2:
            logger.error("Can not train an intent classifier. "
                         "Need at least 2 different classes. "
                         "Skipping training of intent classifier.")
            return

        self.inv_intent_dict = {v:k for k,v in self.intent_dict.items()}
        classify_config.label_nums = len(self.intent_dict.keys())

        self.make_tfrecord_files(training_data,self.intent_dict,classify_config)
        classify_config.vocab_size = len(self.vocabulary.keys())

        os.environ["CUDA_VISIBLE_DEVICES"] = classify_config.CUDA_VISIBLE_DEVICES
        with tf.Graph().as_default():
            training_input_x,training_input_y = self.input_fn(classify_config,
                                                              shuffle_num=500000,
                                                              mode=tf.estimator.ModeKeys.TRAIN,
                                                              epochs=classify_config.epochs)

            self.classify_model = classify_cnn_model.ClassifyCnnModel(classify_config)
            self.classify_model.train(training_input_x,training_input_y)



    def process(self, message, **kwargs):
        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if self.sess is None:
            logger.error("There is no trained tf.session: "
                         "component is either not trained or "
                         "didn't receive enough training data")

        else:
            # get features (bag of words) for a message
            # noinspection PyPep8Naming
            X = [token.text for token in message.data.get('tokens')]
            X_ids = np.array(self.pad_sentence(X,50,self.vocabulary)).reshape((1,50))

            intent_pre = self.sess.run(self.output_node,feed_dict={self.input_node:X_ids})
            intent_ids = np.argmax(intent_pre,axis=1)

            # if X contains all zeros do not predict some label
            if intent_ids.size > 0:
                intent = {"name": self.inv_intent_dict[intent_ids[0]],
                          "confidence": 1}
                intent_pre = [round(val,2) for val in intent_pre[0]]
                ranking = list(zip(intent_pre, self.intent_list))
                intent_ranking = [{"name": intent_name,
                                   "confidence": score}
                                  for score,intent_name in ranking]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self,filename,model_dir):
        with open(os.path.join(model_dir,'label.txt'),'w',encoding='utf-8') as fwrite:
            for label in self.intent_list:
                fwrite.write(label + "\n")

        with open(os.path.join(model_dir,'vocab.txt'),'w',encoding='utf-8') as fwrite:
            for word in self.vocab_list:
                fwrite.write(word + "\n")

        with open(os.path.join(model_dir,'opname.txt'),'w',encoding='utf-8') as fwrite:
            fwrite.write("input:"+ base_classify_model.input_node_name + "\n")
            fwrite.write("output:" + base_classify_model.output_node_logit + "\n")


        model_pb = self.classify_model.make_pb_file(model_dir)
        return {"classifier_file":model_pb}

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> EmbeddingIntentClassifierTf

        #meta = model_metadata.for_component(cls.name)
        if model_dir:
            pb_file_path = os.path.join(model_dir,'classify.pb')
            sess,input_node,output_node = classify_cnn_model.ClassifyCnnModel.load_model_from_pb(pb_file_path)

            intent_list = []
            if os.path.exists(os.path.join(model_dir,'label.txt')):
                with open(os.path.join(model_dir,'label.txt'),'r',encoding='utf-8') as fr:
                    for line in fr:
                        intent_list.append(line.strip())

            vocabulary_list = []
            with open(os.path.join(model_dir, 'vocab.txt'), 'r', encoding='utf-8') as fr:
                for line in fr:
                    vocabulary_list.append(line.strip())
            return EmbeddingIntentClassifierTf(component_config=meta,vocabulary_list=vocabulary_list,intent_list=intent_list,sess=sess,input_node=input_node,output_node=output_node)













