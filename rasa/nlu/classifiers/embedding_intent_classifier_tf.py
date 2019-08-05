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

from rasa.nlu.classifiers.tf_models.classify_cnn_model import ClassifyCnnModel
from rasa.nlu.classifiers.tf_models.classify_bilstm_model import  ClassifyBilstmModel
from rasa.nlu.classifiers.tf_models.classify_rcnn_model import ClassifyRcnnModel
from rasa.nlu.classifiers.tf_models.bert_classify_model import BertClassifyModel
from rasa.nlu.classifiers.tf_models.classify_ensemble_model import ClassifyEnsembleModel
from rasa.nlu.classifiers.tf_models import constant
from rasa.nlu.classifiers.tf_utils import data_utils,data_process
from rasa.nlu.classifiers.tf_utils import data_process
from rasa.nlu.classifiers.tf_utils.data_utils import input_fn,make_tfrecord_files
from rasa.nlu.classifiers.tf_utils.bert_data_utils import input_fn as bert_input_fn
from rasa.nlu.classifiers.tf_utils.bert_data_utils import make_tfrecord_files as bert_make_tfrecord_files
from rasa.nlu.classifiers.tf_utils.bert_data_utils import pad_sentence as bert_pad_sentence

from rasa.third_models.bert import modeling as bert_modeling
from sklearn.metrics import f1_score
import os
import tqdm
from rasa.nlu.classifiers.tf_models.params import Params

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

    def __init__(self,component_config=None,vocabulary_list=None,intent_list=None,sess=None,input_node=None,output_node=None,input_mask_node=None):
        self._check_tensorflow()
        super(EmbeddingIntentClassifierTf,self).__init__(component_config)

        self.vocabulary_list = vocabulary_list
        self.intent_list = intent_list
        self.sess = sess
        self.input_node = input_node
        self.input_mask_node = input_mask_node
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

    def bert_train(self,params):
        if params.data_type == 'default':
            data_processer = data_process.NormalData(params.origin_data, output_path=params.output_path)
        else:
            data_processer = data_process.RasaData(params.origin_data, output_path=params.output_path)
        if not os.path.exists(params.output_path):
            os.makedirs(params.output_path)

        vocab, self.vocab_list, self.intent_list = data_processer.load_vocab_and_intent()

        params.vocab_size = len(self.vocab_list)
        params.num_tags = len(self.intent_list)

        bert_config = bert_modeling.BertConfig.from_json_file(os.path.join(params.bert_model_path, "bert_config.json"))
        if params.max_sentence_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (params.max_sentence_length, bert_config.max_position_embeddings)
            )

        os.environ["CUDA_VISIBLE_DEVICES"] = params.device_map
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                bert_input_dict = bert_input_fn(os.path.join(params.output_path, "train.tfrecord"),
                                                params.batch_size,
                                                params.max_sentence_length,
                                                params.shuffle_num,
                                                mode=tf.estimator.ModeKeys.TRAIN)

                classify_model = BertClassifyModel(params, bert_config)

                loss, global_step, train_op, merger_op = classify_model.make_train(bert_input_dict['input_ids'],
                                                                                   bert_input_dict['input_mask'],
                                                                                   bert_input_dict['segment_ids'],
                                                                                   bert_input_dict['label_ids'])

                # 初始化所有变量
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                classify_model.model_restore(sess, saver)

                best_f1 = 0
                for i in range(params.total_train_steps):
                    sess_loss, steps, _ = sess.run([loss, global_step, train_op])
                    if i % 20 == 0:
                        print("training step:%s ,loss: %s" % (steps,sess_loss))
                    if steps % params.evaluate_every_steps == 0:
                        test_input_dict = bert_input_fn(os.path.join(params.output_path, "test.tfrecord"),
                                                        params.batch_size,
                                                        params.max_sentence_length,
                                                        params.shuffle_num,
                                                        mode=tf.estimator.ModeKeys.EVAL)
                        loss_test, predict_test = classify_model.make_test(test_input_dict['input_ids'],
                                                                           test_input_dict['input_mask'],
                                                                           test_input_dict['segment_ids'],
                                                                           test_input_dict['label_ids'])

                        predict_var = []
                        train_y_var = []
                        loss_total = 0
                        num_batch = 0
                        try:
                            while 1:
                                loss_, predict_, test_input_y_ = sess.run(
                                    [loss_test, predict_test, test_input_dict['label_ids']])
                                loss_total += loss_
                                num_batch += 1
                                predict_var += predict_.tolist()
                                train_y_var += test_input_y_.tolist()
                        except tf.errors.OutOfRangeError:
                            print("eval over")
                        if num_batch > 0:

                            f1_val = f1_score(train_y_var, predict_var, average='micro')
                            print("current step:%s ,loss:%s , f1 :%s" % (steps, loss_total / num_batch, f1_val))

                            if f1_val >= best_f1:
                                saver.save(sess, params.model_path, steps)
                                print("new best f1: %s ,save to dir:%s" % (f1_val, params.output_path))
                                best_f1 = f1_val
                self.component_config['pb_path'] = classify_model.make_pb_file(params.output_path)

    def norm_train(self,params):
        if params.data_type == 'default':
            data_processer = data_process.NormalData(params.origin_data, output_path=params.output_path)
        else:
            data_processer = data_process.RasaData(params.origin_data, output_path=params.output_path)
        if not os.path.exists(params.output_path):
            os.makedirs(params.output_path)

        vocab, self.vocab_list, self.intent_list = data_processer.load_vocab_and_intent()

        params.vocab_size = len(self.vocab_list)
        params.num_tags = len(self.intent_list)

        os.environ["CUDA_VISIBLE_DEVICES"] = params.device_map
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                training_input_x, training_input_y = input_fn(os.path.join(params.output_path, "train.tfrecord"),
                                                              params.batch_size,
                                                              params.max_sentence_length,
                                                              params.shuffle_num,
                                                              mode=tf.estimator.ModeKeys.TRAIN)

                if params.category_type == 'cnn':
                    classify_model = ClassifyCnnModel(params)
                elif params.category_type == "bilstm":
                    classify_model = ClassifyBilstmModel(params)
                elif params.category_type == "rcnn":
                    classify_model = ClassifyRcnnModel(params)
                elif params.category_type == 'ensemble':
                    classify_model = ClassifyEnsembleModel(params)
                else:
                    raise ValueError("category_type error")

                loss, global_step, train_op, merger_op = classify_model.make_train(training_input_x, training_input_y)

                # 初始化所有变量
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                classify_model.model_restore(sess, saver)

                best_f1 = 0
                for _ in tqdm.tqdm(range(params.total_train_steps), desc="steps", miniters=10):
                    sess_loss, steps, _ = sess.run([loss, global_step, train_op])

                    if steps % params.evaluate_every_steps == 0:
                        test_input_x, test_input_y = input_fn(os.path.join(params.output_path, "test.tfrecord"),
                                                              params.batch_size,
                                                              params.max_sentence_length,
                                                              params.shuffle_num,
                                                              mode=tf.estimator.ModeKeys.EVAL)
                        loss_test, predict_test = classify_model.make_test(test_input_x, test_input_y)

                        predict_var = []
                        train_y_var = []
                        loss_total = 0
                        num_batch = 0
                        try:
                            while 1:
                                loss_, predict_, test_input_y_ = sess.run([loss_test, predict_test, test_input_y])
                                loss_total += loss_
                                num_batch += 1
                                predict_var += predict_.tolist()
                                train_y_var += test_input_y_.tolist()
                        except tf.errors.OutOfRangeError:
                            print("eval over")
                        if num_batch > 0:

                            f1_val = f1_score(train_y_var, predict_var, average='micro')
                            print("current step:%s ,loss:%s , f1 :%s" % (steps, loss_total / num_batch, f1_val))

                            if f1_val >= best_f1:
                                saver.save(sess, params.model_path, steps)
                                print("new best f1: %s ,save to dir:%s" % (f1_val, params.output_path))
                                best_f1 = f1_val

                self.component_config['pb_path'] = classify_model.make_pb_file(params.output_path)


    def train(self,training_data,cfg=None,**kwargs):

        params = Params()
        params.update_dict(self.component_config)
        if not os.path.exists(params.output_path):
            os.mkdir(params.output_path)


        if params.use_bert:
            bert_make_tfrecord_files(params)
            self.bert_train(params)
        else:
            data_utils.make_tfrecord_files(params)
            self.norm_train(params)





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
            if self.input_mask_node == None:
                X_ids = np.array(self.pad_sentence(X,self.component_config['max_sentence_length'],self.vocabulary)).reshape((1,self.component_config['max_sentence_length']))
                intent_pre = self.sess.run(self.output_node,feed_dict={self.input_node:X_ids})
            else:
                X_ids,_,input_mask = bert_pad_sentence("".join(X),self.component_config['max_sentence_length'],self.vocabulary)
                X_ids = np.array(X_ids).reshape((1,self.component_config['max_sentence_length']))
                input_mask = np.array(input_mask).reshape((1,self.component_config['max_sentence_length']))
                intent_pre = self.sess.run(self.output_node, feed_dict={self.input_node: X_ids,self.input_mask_node:input_mask})
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
            if meta['use_bert'] == 0:
                sess,input_node,output_node = ClassifyCnnModel.load_model_from_pb(pb_file_path)
                input_mask_node = None
            else:
                sess, input_node, input_mask_node, output_node = BertClassifyModel.load_model_from_pb(pb_file_path)

            intent_list = []
            if os.path.exists(os.path.join(save_model_path,'label.txt')):
                with open(os.path.join(save_model_path,'label.txt'),'r',encoding='utf-8') as fr:
                    for line in fr:
                        intent_list.append(line.strip())

            vocabulary_list = []
            with open(os.path.join(save_model_path, 'vocab.txt'), 'r', encoding='utf-8') as fr:
                for line in fr:
                    vocabulary_list.append(line.strip())
            return EmbeddingIntentClassifierTf(component_config=meta,vocabulary_list=vocabulary_list,intent_list=intent_list,sess=sess,input_node=input_node,output_node=output_node,input_mask_node=input_mask_node)













