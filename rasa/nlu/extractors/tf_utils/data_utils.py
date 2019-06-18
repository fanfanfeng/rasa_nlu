# create by fanfan on 2019/4/10 0010
import tensorflow as tf
import os
from rasa.nlu.extractors.tf_utils import data_process
from rasa.nlu.utils.tfrecord_api import _int64_feature
_START_VOCAB = ['_PAD', '_GO', "_EOS", '<UNK>']


def pad_sentence(sentence, max_sentence,vocabulary,label_dict):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence
    '''
    UNK_ID = vocabulary.get('<UNK>')
    PAD_ID = vocabulary.get('_PAD')

    sentence_ids = []
    label_ids = []
    for token in sentence:
        word_and_type = token.split("\\")
        sentence_ids.append(word_and_type[0])
        if len(word_and_type) == 2:
            label_ids.append(word_and_type[1])
        else:
            label_ids.append('O')

    sentence_ids = [vocabulary.get(w, UNK_ID) for w in sentence_ids]
    label_ids = [label_dict.get(w, 0) for w in label_ids]
    if len(sentence_ids) > max_sentence:
        sentence_ids = sentence_ids[:max_sentence]
        label_ids = label_ids[:max_sentence]
    else:
        sentence_ids = sentence_ids + [PAD_ID] * (max_sentence - len(sentence_ids))
        label_ids = label_ids + [0] * (max_sentence - len(label_ids))
    return sentence_ids,label_ids

def pad_sentence_rasa(sentence,labels, max_sentence,vocabulary,label_dict):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence
    '''
    UNK_ID = vocabulary.get('<UNK>')

    sentence_ids = [vocabulary.get(w, UNK_ID) for w in sentence]
    label_ids = [label_dict.get(w, 0) for w in labels]
    if len(sentence_ids) > max_sentence:
        sentence_ids = sentence_ids[:max_sentence]
        label_ids = label_ids[:max_sentence]
    else:
        sentence_ids = sentence_ids + [0] * (max_sentence - len(sentence_ids))
        label_ids = label_ids + [0] * (max_sentence - len(label_ids))
    return sentence_ids,label_ids


def input_fn(input_file, batch_size,max_sentence_length, mode=tf.estimator.ModeKeys.TRAIN):
    """
     build tf.data set for input pipeline

    :param input_file: classify config dict
    :param shuffle_num: type int number , random select the data
    :param mode: type string ,tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
    :return: set() with type of (tf.data , and labels)
    """
    def parse_single_tfrecord(serializer_item):
        features = {
            'label': tf.FixedLenFeature([max_sentence_length],tf.int64),
            'sentence' : tf.FixedLenFeature([max_sentence_length],tf.int64)
        }

        features_var = tf.parse_single_example(serializer_item,features)

        labels = tf.cast(features_var['label'],tf.int64)
        sentence = tf.cast(features_var['sentence'],tf.int64)
        return sentence,labels


    if not os.path.exists(input_file):
        raise FileNotFoundError("tfrecord not found")


    tf_record_reader = tf.data.TFRecordDataset(input_file)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf_record_reader = tf_record_reader.repeat()
        tf_record_reader = tf_record_reader.shuffle(buffer_size=batch_size*2000)
    dataset = tf_record_reader.apply(tf.data.experimental.map_and_batch(lambda record:parse_single_tfrecord(record),
                                                   batch_size,num_parallel_calls=8))

    iterator = dataset.make_one_shot_iterator()
    data, labels = iterator.get_next()
    return data, labels



def make_tfrecord_files(args):
    if args.data_type == 'default':
        data_processer = data_process.NormalData(args.origin_data,output_path=args.output_path)
    else:
        data_processer = data_process.RasaData(args.origin_data, output_path=args.output_path)
    if os.path.exists(os.path.join(args.output_path,'vocab.txt')):
        vocab,vocab_list,labels = data_processer.load_vocab_and_labels()
    else:
        vocab,vocab_list,labels = data_processer.create_vocab_dict()

    labels_ids = {key:index for index,key in enumerate(labels)}
    # tfrecore 文件写入
    tfrecord_save_path = os.path.join(args.output_path,"train.tfrecord")
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_save_path)

    def thread_write_to_file(file):
        for sentence,labels in data_processer.load_single_file(file):
            if args.data_type == 'default':
                sentence_ids,sentence_labels_ids = pad_sentence(sentence, args.max_sentence_len, vocab,labels_ids)
            else:
                sentence_ids, sentence_labels_ids = pad_sentence_rasa(sentence, labels,args.max_sentence_len, vocab, labels_ids)

            # sentence_ids_string = np.array(sentence_ids).tostring()

            train_feature_item = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(sentence_labels_ids,need_list=False),
                'sentence': _int64_feature(sentence_ids, need_list=False)
            }))
            tfrecord_writer.write(train_feature_item.SerializeToString())

    #pool = threadpool.ThreadPool(20)

    #args = [((file,intent,tfrecord_writer),None) for file,intent in data_processer.getTotalfiles()]
    #requests = threadpool.makeRequests(thread_write_to_file,args)
    #[pool.putRequest(req) for req in requests]
    #pool.wait()
    for file  in data_processer.getTotalfiles():
        thread_write_to_file(file)
    tfrecord_writer.close()

