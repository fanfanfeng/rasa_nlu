# create by fanfan on 2019/4/24 0024
import tensorflow as tf
from rasa.third_models.bert import modeling as bert_modeling
from rasa.nlu.extractors.tf_models.bilstm import BiLSTM
from rasa.nlu.extractors.tf_models.idcnn import IdCnn
from rasa.nlu.extractors.tf_models import constant
from tensorflow.contrib import crf
from sklearn.metrics import f1_score
import tqdm
import os


class BertNerModel(object):
    def __init__(self,params,bert_config):
        self.params = params
        self.bert_config = bert_config


    def create_model(self,input_ids, input_mask, segment_ids,is_training,dropout,use_one_hot_embeddings=False):
        bert_model_layer = bert_modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
        embedding_input_x = bert_model_layer.get_sequence_output()

        if self.params.ner_type == "idcnn":
            self.ner_model = IdCnn(self.params)
        else:
            self.ner_model = BiLSTM(self.params)
        real_sentece_length = self.ner_model.get_setence_length(input_ids)
        logits,_,trans = self.ner_model.create_model(embedding_input_x,dropout,already_embedded=True,real_sentence_length=real_sentece_length)
        return logits,real_sentece_length,trans



    def train(self, input_ids, input_mask, segment_ids,label_ids):
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            dropout = tf.placeholder(dtype=tf.float32, name='dropout')

            logits,real_sentence_length,trans = self.create_model(input_ids, input_mask,segment_ids,is_training=True,dropout=dropout)

            globalStep = tf.Variable(0, name="globalStep", trainable=False)
            with tf.variable_scope('loss'):
                loss = self.ner_model.crf_layer_loss(logits, label_ids, real_sentence_length,trans)
                optimizer = tf.train.AdamOptimizer(self.params.learning_rate)

                grads_and_vars = optimizer.compute_gradients(loss)
                trainOp = optimizer.apply_gradients(grads_and_vars, globalStep)

            pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans,
                                         sequence_length=real_sentence_length)
            pred_ids = tf.identity(pred_ids, name=constant.OUTPUT_NODE_NAME)
            with tf.variable_scope('summary'):
                tf.summary.scalar("loss", loss)
                summary_op = tf.summary.merge_all()

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            sess.run(tf.global_variables_initializer())
            tvars = tf.trainable_variables()
            # 加载BERT模型
            bert_init_checkpoint = os.path.join(self.params.bert_model_path,'bert_model.ckpt')
            if os.path.exists(self.params.bert_model_path):
                (assignment_map, initialized_variable_names) = bert_modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                           bert_init_checkpoint)
                tf.train.init_from_checkpoint(bert_init_checkpoint, assignment_map)

            tf_save_path = os.path.join(self.params.output_path, 'tf')
            #try:
            best_f1 = 0
            for _ in tqdm.tqdm(range(self.params.total_train_steps), desc="steps", miniters=10):
                sess_loss, predict_var, steps, _, real_sentence, input_y_val = sess.run(
                    [loss, pred_ids, globalStep, trainOp, real_sentence_length, label_ids],
                    feed_dict={dropout: 0.8}
                )

                if steps % self.params.evaluate_every_steps == 0:
                    train_labels = []
                    predict_labels = []
                    for train_, predict_, len_ in zip(input_y_val, predict_var, real_sentence):
                        train_labels += train_[:len_].tolist()
                        predict_labels += predict_[:len_].tolist()
                    f1_val = f1_score(train_labels, predict_labels, average='micro')
                    print("current step:%s ,loss:%s ,f1 :%s" % (steps, sess_loss, f1_val))

                    if f1_val > best_f1:
                        saver.save(sess, tf_save_path, steps)
                        print("new best f1: %s ,save to dir:%s" % (f1_val, self.params.output_path))
                        best_f1 = f1_val
            #except tf.errors.OutOfRangeError:
                #print("training end")

    def make_pb_file(self,model_dir):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                input_ids = tf.placeholder(dtype=tf.int32,shape=(None,self.params.max_sentence_length),name=constant.INPUT_NODE_NAME)
                input_mask = tf.placeholder(dtype=tf.int32,shape=(None,self.params.max_sentence_length),name=constant.INPUT_MASK_NAME)
                dropout = tf.placeholder_with_default(1.0,shape=(), name='dropout')
                logits,real_sentence_length,trans = self.create_model(input_ids, input_mask, segment_ids=None, is_training=False, dropout=dropout)
                pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans,
                                             sequence_length=real_sentence_length)
                pred_ids = tf.identity(pred_ids, name=constant.OUTPUT_NODE_NAME)

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                checkpoint = tf.train.latest_checkpoint(model_dir)
                if checkpoint:
                    saver.restore(sess,checkpoint)
                else:
                    raise FileNotFoundError("模型文件未找到")

                output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,[constant.OUTPUT_NODE_NAME])

                with tf.gfile.GFile(os.path.join(model_dir,'ner.pb'),'wb') as gf:
                    gf.write(output_graph_with_weight.SerializeToString())
        return os.path.join(model_dir,'ner.pb')

    @staticmethod
    def load_model_from_pb(model_path):
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        ))

        with tf.gfile.GFile(model_path, 'rb') as fr:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fr.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name="")

        input_node = sess.graph.get_tensor_by_name(constant.INPUT_NODE_NAME).outputs[0]
        input_mask_node = sess.graph.get_tensor_by_name(constant.INPUT_MASK_NAME).outputs[0]
        logit_node = sess.graph.get_tensor_by_name(constant.OUTPUT_NODE_NAME).outputs[0]
        return sess, input_node,input_mask_node, logit_node