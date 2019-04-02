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
from rasa.nlu.utils.vocabprocessor import VocabularyProcessor
import os

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

    def __init__(self,component_config=None,vocabprocess=None,intent_list=None,sess=None,input_node=None,output_node=None):
        self._check_tensorflow()
        super(EmbeddingIntentClassifierTf,self).__init__(component_config)

        self.vocabprocessor = vocabprocess
        self.intent_list = intent_list
        self.sess =sess
        self.input_node = input_node
        self.output_node = output_node

        if self.intent_list != None:
            self.inv_intent_dict = {index:key for index,key in enumerate(intent_list)}

    # training data helpers:
    @staticmethod
    def _create_intent_dict(training_data):
        distinct_intents = set([example.get('intent') for example in training_data.intent_examples])
        return {intent:idx for idx,intent in enumerate(sorted(distinct_intents))},sorted(distinct_intents)

    def _prepare_data_for_training(self, training_data, intent_dict):
        X = [" ".join([token.text for token in e.data.get('tokens')]) for e in training_data.intent_examples]
        X_ids = np.array(list(self.vocabprocessor.fit_transform(X)))
        intents_for_X = np.array([intent_dict[e.get('intent')] for e in training_data.intent_examples])

        return X_ids,intents_for_X

    def input_fn(self,features, labels, batch_size, shuffle_num, mode,epochs):
        """
         build tf.data set for input pipeline

        :param features: type dict() , define input x structure for parsing
        :param labels: type np.array input label
        :param batch_size: type int number ,input batch_size
        :param shuffle_num: type int number , random select the data
        :param mode: type string ,tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
        :return: set() with type of (tf.data , and labels)
        """
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(shuffle_num).batch(batch_size).repeat(epochs)
        else:
            dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        data, labels = iterator.get_next()
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
        self.vocabprocessor = VocabularyProcessor(classify_config.max_sentence_length,min_frequency=3)
        self.intent_dict,self.intent_list = self._create_intent_dict(training_data)
        if len(self.intent_dict) < 2:
            logger.error("Can not train an intent classifier. "
                         "Need at least 2 different classes. "
                         "Skipping training of intent classifier.")
            return

        self.inv_intent_dict = {v:k for k,v in self.intent_dict.items()}
        classify_config.label_nums = len(self.intent_dict.keys())

        X,Y = self._prepare_data_for_training(training_data,self.intent_dict)
        classify_config.vocab_size = len(self.vocabprocessor.vocabulary_)

        os.environ["CUDA_VISIBLE_DEVICES"] = classify_config.CUDA_VISIBLE_DEVICES
        with tf.Graph().as_default():
            training_input_x,training_input_y = self.input_fn(X,
                                                              Y,
                                                              classify_config.batch_size,
                                                              shuffle_num=5000,
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
            X = [" ".join([token.text for token in message.data.get('tokens')])]
            X_ids = np.array(list(self.vocabprocessor.fit_transform(X)))

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
            for word in self.vocabprocessor.vocabulary_.get_vocab_list():
                fwrite.write(word + "\n")

        with open(os.path.join(model_dir,'opname.txt'),'w',encoding='utf-8') as fwrite:
            fwrite.write("input:"+ base_classify_model.input_node_name + "\n")
            fwrite.write("output:" + base_classify_model.output_node_logit + "\n")

        self.vocabprocessor.save(os.path.join(model_dir,'vocabprocessor.pkl'))

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

            vocabprocess =None
            if os.path.exists(os.path.join(model_dir,'vocabprocessor.pkl')):
                vocabprocess = VocabularyProcessor.restore(os.path.join(model_dir,'vocabprocessor.pkl'))

            return EmbeddingIntentClassifierTf(component_config=meta,vocabprocess=vocabprocess,intent_list=intent_list,sess=sess,input_node=input_node,output_node=output_node)













