from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import re
import io

import typing
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

from builtins import str
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa.nlu.config import InvalidConfigError, RasaNLUModelConfig
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message

from rasa.nlu.extractors.tf_utils import data_process
from rasa.nlu.extractors.tf_models.bilstm import BiLSTM
from rasa.nlu.extractors.tf_models.idcnn import IdCnn
from rasa.nlu.extractors.tf_models import constant
from rasa.nlu.extractors.tf_models.bert_ner_model import BertNerModel
from rasa.nlu.extractors.tf_models.base_ner_model import BasicNerModel
from rasa.nlu.extractors.tf_utils.data_utils import input_fn,make_tfrecord_files
from rasa.nlu.extractors.tf_utils.bert_data_utils import input_fn as bert_input_fn
from rasa.nlu.extractors.tf_utils.bert_data_utils import make_tfrecord_files as bert_make_tfrecord_files
from rasa.third_models.bert import modeling as bert_modeling
import shutil
from rasa.nlu.extractors.tf_models.params import Params,TestParams
from sklearn.metrics import f1_score
import tqdm



logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import numpy as np
    import tensorflow as tf
    import tensorflow.contrib

try:
    import tensorflow as tf
except ImportError:
    tf = None


class DictToObject(object):
    def __init__(self, dict):
        self.__dict__.update(dict)


class TfEntityExtractor(EntityExtractor):
    name = "TfEntityExtractor"

    provides = ["entities"]

    requires = []


    def __init__(self,component_config=None,vocabulary_list=None,labels_list=None,sess=None,input_node=None,output_node=None,input_node_mask=None):
        super(TfEntityExtractor, self).__init__(component_config)

        self.component_config = component_config
        self.vocabulary_list = vocabulary_list
        self.labels_list = labels_list
        self.sess = sess
        self.input_node = input_node
        self.output_node = output_node
        self.input_node_mask = input_node_mask

        if self.labels_list != None:
            self.inv_labels_dict = {index: key for index, key in enumerate(labels_list)}
        if self.vocabulary_list != None:
            self.vocabulary = {key: index for index, key in enumerate(self.vocabulary_list)}

    def normal_train(self,params):
        if params.data_type == 'default':
            data_processer = data_process.NormalData(params.origin_data, output_path=params.output_path)
        else:
            data_processer = data_process.RasaData(params.origin_data, output_path=params.output_path)

        vocab, self.vocab_list, self.labels_list = data_processer.load_vocab_and_labels()

        params.vocab_size = len(self.vocab_list)
        params.num_tags = len(self.labels_list)
        if not os.path.exists(params.output_path):
            os.makedirs(params.output_path)

        os.environ["CUDA_VISIBLE_DEVICES"] = params.device_map
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                training_input_x, training_input_y = input_fn(os.path.join(params.output_path, 'train.tfrecord'),
                                                              shuffle_num=params.shuffle_num,
                                                              mode=tf.estimator.ModeKeys.TRAIN,
                                                              batch_size=params.batch_size,
                                                              max_sentence_length=params.max_sentence_length,
                                                              )
                if params.ner_type == "idcnn":
                    ner_model = IdCnn(params)
                else:
                    ner_model = BiLSTM(params)
                loss, global_step, train_op, merger_op = ner_model.make_train(training_input_x, training_input_y)
                # 初始化所有变量
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                ner_model.model_restore(sess, saver)

                best_f1 = 0
                for _ in tqdm.tqdm(range(params.total_train_steps), desc="steps", miniters=10):
                    sess_loss, steps, _ = sess.run([loss, global_step, train_op])

                    if steps % params.evaluate_every_steps == 1:
                        test_input_x, test_input_y = input_fn(os.path.join(params.output_path, "test.tfrecord"),
                                                              shuffle_num=params.shuffle_num,
                                                              batch_size=params.batch_size,
                                                              max_sentence_length=params.max_sentence_length,
                                                              mode=tf.estimator.ModeKeys.EVAL)
                        loss_test, predict_test, sentence_length = ner_model.make_test(test_input_x, test_input_y)

                        predict_var = []
                        train_y_var = []
                        loss_total = 0
                        num_batch = 0
                        try:
                            while 1:
                                loss_, predict_, test_input_y_, length = sess.run(
                                    [loss_test, predict_test, test_input_y, sentence_length])
                                loss_total += loss_
                                num_batch += 1
                                for p_, t_, len_ in zip(predict_.tolist(), test_input_y_.tolist(), length.tolist()):
                                    predict_var += p_[:len_]
                                    train_y_var += t_[:len_]
                        except tf.errors.OutOfRangeError:
                            print("eval over")
                        if num_batch > 0:

                            f1_val = f1_score(train_y_var, predict_var, average='micro')
                            print("current step:%s ,loss:%s , f1 :%s" % (steps, loss_total / num_batch, f1_val))

                            if f1_val >= best_f1:
                                saver.save(sess, params.model_path, steps)
                                print("new best f1: %s ,save to dir:%s" % (f1_val, params.output_path))
                                best_f1 = f1_val

                self.component_config['pb_path'] = ner_model.make_pb_file(params.output_path)

    def bert_train(self,params):
        os.environ['CUDA_VISIBLE_DEVICES'] = params.device_map
        if params.data_type == 'default':
            data_processer = data_process.NormalData(params.origin_data, output_path=params.output_path)
        else:
            data_processer = data_process.RasaData(params.origin_data, output_path=params.output_path)

        vocab, self.vocab_list, self.labels_list = data_processer.load_vocab_and_labels()

        bert_config = bert_modeling.BertConfig.from_json_file(os.path.join(params.bert_model_path, "bert_config.json"))
        params.vocab_size = len(self.vocab_list)
        params.num_tags = len(self.labels_list)
        if params.max_sentence_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (params.max_sentence_length, bert_config.max_position_embeddings)
            )

        if not os.path.exists(params.output_path):
            os.makedirs(params.output_path)

        with tf.Graph().as_default():
            bert_input = bert_input_fn(os.path.join(params.output_path, 'train.tfrecord'),
                                       mode=tf.estimator.ModeKeys.TRAIN,
                                       batch_size=params.batch_size,
                                       max_sentence_length=params.max_sentence_length
                                       )

            model = BertNerModel(params, bert_config)
            model.train(bert_input['input_ids'], bert_input['input_mask'], bert_input['segment_ids'],
                        bert_input['label_ids'])
            self.component_config['pb_path'] = model.make_pb_file(params.output_path)


    def train(self, training_data, config, **kwargs):
        params = Params()
        params.update_dict(self.component_config)

        if not os.path.exists(params.output_path):
            os.mkdir(params.output_path)

        if params.use_bert:
            bert_make_tfrecord_files(params)
            self.bert_train(params)
        else:
            make_tfrecord_files(params)
            self.normal_train(params)

    def result_to_json(self,string, tags):
        item = {
            "string": string,
            "entities": []
        }
        entity_name = ""
        entity_start = 0
        current_entity_type = ""
        idx = 0
        for char, tag in zip(string, tags):
            if current_entity_type != "" and tag != "O":
                new_entity_type = tag.replace("B-","").replace("I-","")
                if new_entity_type != current_entity_type:
                    item["entities"].append(
                        {"value": entity_name, "start": entity_start, "end": idx, "entity": current_entity_type})
                    entity_name = ""
                    entity_start = 0
                    current_entity_type = ""
            if tag[0] == "B":
                entity_name += char
                entity_start = idx
                current_entity_type = tag.replace("B-","")
            elif tag[0] == "I":
                entity_name += char
                current_entity_type = tag.replace("I-", "")
            else:
                if current_entity_type != "":
                    item["entities"].append(
                        {"value": entity_name, "start": entity_start, "end": idx-1, "entity": current_entity_type})
                entity_name = ""
                entity_start = 0
                current_entity_type = ""

            idx += 1

        if current_entity_type != "":
                item["entities"].append(
                    {"value": entity_name, "start": entity_start, "end": idx-1, "entity": current_entity_type})
        return item

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        extracted = self.extract_entities(message.text)
        message.set("entities", message.get("entities", []) + extracted['entities'], add_to_output=True)



    def pad_sentence(self,sentence, max_sentence,vocabulary):
        '''
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

        参数：
        - sentence
        '''
        UNK_ID = vocabulary.get('<UNK>')
        PAD_ID = 0
        sentence_batch_ids = [vocabulary.get(w, UNK_ID) for w in sentence]
        if len(sentence_batch_ids) > max_sentence:
            sentence_batch_ids = sentence_batch_ids[:max_sentence]
        else:
            sentence_batch_ids = sentence_batch_ids + [PAD_ID] * (max_sentence - len(sentence_batch_ids))
        return sentence_batch_ids

    def extract_entities(self, text):
        component_config_object = DictToObject(self.component_config)


        if component_config_object.use_bert:
            words = ['[CLS]'] + list(text.lower()) + ['[SEP]']
            tokens = np.array(self.pad_sentence(words, component_config_object.max_sentence_length, self.vocabulary)).reshape((1, component_config_object.max_sentence_length))
            text_len = len(words)
            tokens_mask = []
            for i in range(component_config_object.max_sentence_length):
                if i < text_len:
                    tokens_mask.append(1)
                else:
                    tokens_mask.append(0)
            tokens_mask = np.array(tokens_mask).reshape((1, component_config_object.max_sentence_length))
            predict_ids = self.sess.run(self.output_node,
                                        feed_dict={self.input_node: tokens, self.input_node_mask: tokens_mask})
            predict_ids = predict_ids[0].tolist()[1:text_len - 1]
        else:
            words = list(text)
            tokens = np.array(self.pad_sentence(words, component_config_object.max_sentence_length, self.vocabulary)).reshape((1, component_config_object.max_sentence_length))

            predict_ids = self.sess.run(self.output_node, feed_dict={self.input_node: tokens})
            predict_ids = predict_ids[0].tolist()[:len(words)]

        entities = self.result_to_json(text, [self.inv_labels_dict[i] for i in predict_ids])
        return entities
    @classmethod
    def load(cls,
             meta,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[CRFEntityExtractor]
             **kwargs  # type: **Any
             ):
        if model_dir:
            save_model_path = os.path.join(model_dir, cls.name)
            pb_file_path = os.path.join(save_model_path,'ner.pb')

            if meta['use_bert'] == 0:
                sess, input_node, output_node = BasicNerModel.load_model_from_pb(pb_file_path)
                input_mask_node = None
            else:
                sess, input_node, input_mask_node, output_node = BertNerModel.load_model_from_pb(pb_file_path)

            labels_list = []
            if os.path.exists(os.path.join(save_model_path,'label.txt')):
                with open(os.path.join(save_model_path,'label.txt'),'r',encoding='utf-8') as fr:
                    for line in fr:
                        labels_list.append(line.strip())

            vocabulary_list = []
            with open(os.path.join(save_model_path, 'vocab.txt'), 'r', encoding='utf-8') as fr:
                for line in fr:
                    vocabulary_list.append(line.strip())
            return TfEntityExtractor(component_config=meta,vocabulary_list=vocabulary_list,labels_list=labels_list,sess=sess,input_node=input_node,output_node=output_node,input_node_mask=input_mask_node)

    def persist(self,filename,model_dir):
        save_model_path = os.path.join(model_dir,self.name)
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        with open(os.path.join(save_model_path,'label.txt'),'w',encoding='utf-8') as fwrite:
            for label in self.labels_list:
                fwrite.write(label + "\n")

        with open(os.path.join(save_model_path,'vocab.txt'),'w',encoding='utf-8') as fwrite:
            for word in self.vocab_list:
                fwrite.write(word + "\n")

        with open(os.path.join(save_model_path,'opname.txt'),'w',encoding='utf-8') as fwrite:
            fwrite.write("input:"+ constant.INPUT_NODE_NAME + "\n")
            fwrite.write("output:" + constant.OUTPUT_NODE_NAME + "\n")
            fwrite.write("input_mask:" + constant.INPUT_MASK_NAME + "\n")

        save_pb_path = os.path.join(save_model_path,'ner.pb')
        shutil.copy(self.component_config['pb_path'],save_pb_path)
        return {"classifier_file":save_pb_path}
