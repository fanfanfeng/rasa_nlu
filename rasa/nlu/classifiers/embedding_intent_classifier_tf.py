# create by fanfan on 2019/3/27 0027
import io
import logging
import os
from tqdm import tqdm
from typing import List,Text,Any,Optional,Dict
import shutil


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

from rasa.nlu.classifiers.tf_models import classify_cnn_model,constant
from rasa.nlu.classifiers.tf_models.base_classify_model import ClassifyConfig
from rasa.nlu.classifiers.tf_utils import data_utils,data_process
import os

_START_VOCAB = ['_PAD', '_GO', "_EOS", '<UNK>']
class EmbeddingIntentClassifierTf(Component):
    name = 'EmbeddingIntentClassifierTf'

    provides = ['intent','intent_ranking']

    requires = []




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







    def train(self,training_data,cfg=None,**kwargs):
        class component_config_bject:
            def __init__(self,dict):
                self.__dict__.update(dict)

        component_config_bject = component_config_bject(self.component_config)
        if not os.path.exists(component_config_bject.output_path):
            os.mkdir(component_config_bject.output_path)


        if component_config_bject.use_bert:
            # bert_make_tfrecord_files(argument_dict)
            # bert_train(argument_dict)
            pass
        else:
            data_utils.make_tfrecord_files(component_config_bject)

        if component_config_bject.data_type == 'default':
            data_processer = data_process.NormalData(component_config_bject.origin_data, output_path=component_config_bject.output_path)
        else:
            data_processer = data_process.RasaData(component_config_bject.origin_data, output_path=component_config_bject.output_path)
        classify_config = ClassifyConfig(vocab_size=None)
        classify_config.output_path = component_config_bject.output_path
        classify_config.max_sentence_length = component_config_bject.max_sentence_len
        if not os.path.exists(classify_config.output_path):
            os.makedirs(classify_config.output_path)

        vocab, self.vocab_list, self.intent_list = data_processer.load_vocab_and_intent()

        classify_config.vocab_size = len(self.vocab_list)
        classify_config.num_tags = len(self.intent_list)

        os.environ["CUDA_VISIBLE_DEVICES"] = component_config_bject.device_map
        with tf.Graph().as_default():
            training_input_x, training_input_y = data_utils.input_fn(os.path.join(classify_config.output_path, "train.tfrecord"),
                                                          classify_config.batch_size,
                                                          classify_config.max_sentence_length,
                                                          mode=tf.estimator.ModeKeys.TRAIN)

            classify_model = classify_cnn_model.ClassifyCnnModel(classify_config)
            classify_model.train(training_input_x, training_input_y)

            self.component_config['pb_path'] = classify_model.make_pb_file(classify_config.output_path)



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
            intent_pre = intent_pre.tolist()
            intent_ids = np.argmax(intent_pre,axis=1)

            # if X contains all zeros do not predict some label
            if len(intent_ids) > 0:
                intent = {"name": self.inv_intent_dict[intent_ids[0]],
                          "confidence": 1}
                intent_pre = [float(round(val,2))for val in intent_pre[0]]
                ranking = list(zip(intent_pre, self.intent_list))
                intent_ranking = [{"name": intent_name,
                                   "confidence": score}
                                  for score,intent_name in ranking]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self,filename,model_dir):
        save_model_path = os.path.join(model_dir, self.name)
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        with open(os.path.join(save_model_path,'label.txt'),'w',encoding='utf-8') as fwrite:
            for label in self.intent_list:
                fwrite.write(label + "\n")

        with open(os.path.join(save_model_path,'vocab.txt'),'w',encoding='utf-8') as fwrite:
            for word in self.vocab_list:
                fwrite.write(word + "\n")

        with open(os.path.join(save_model_path,'opname.txt'),'w',encoding='utf-8') as fwrite:
            fwrite.write("input:"+ constant.INPUT_NODE_NAME + "\n")
            fwrite.write("output:" + constant.OUTPUT_NODE_LOGIT + "\n")

        save_pb_path = os.path.join(save_model_path,'classify.pb')
        shutil.copy(self.component_config['pb_path'],save_pb_path)
        return {"classifier_file":save_pb_path}

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
            save_model_path = os.path.join(model_dir, cls.name)
            pb_file_path = os.path.join(save_model_path,'classify.pb')
            sess,input_node,output_node = classify_cnn_model.ClassifyCnnModel.load_model_from_pb(pb_file_path)

            intent_list = []
            if os.path.exists(os.path.join(save_model_path,'label.txt')):
                with open(os.path.join(save_model_path,'label.txt'),'r',encoding='utf-8') as fr:
                    for line in fr:
                        intent_list.append(line.strip())

            vocabulary_list = []
            with open(os.path.join(save_model_path, 'vocab.txt'), 'r', encoding='utf-8') as fr:
                for line in fr:
                    vocabulary_list.append(line.strip())
            return EmbeddingIntentClassifierTf(component_config=meta,vocabulary_list=vocabulary_list,intent_list=intent_list,sess=sess,input_node=input_node,output_node=output_node)













