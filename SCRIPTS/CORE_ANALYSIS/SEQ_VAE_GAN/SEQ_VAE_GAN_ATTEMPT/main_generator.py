#MAIN GENEREATOR RUN PROGRAM
import tensorflow as tf
from generator import GetToken, Discriminator
from dataloader import Gen_Data_loader, Dis_dataloader
import pickle
from tensorboard.plugins.hparams import api as hp


BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))
Z_SIZE = hp.HParam('z_size', hp.Discrete([64]))
SEQ_LEN = hp.HParam('seq_len', hp.Discrete([8]))
EMB_DIM = hp.HParam('emb_dim', hp.Discrete([64]))
VOCAB_SIZE = hp.HParam('vocab_size', hp.Discrete([5000]))
HIDDEN_DIM = hp.HParam('hidden_dim', hp.Discrete([32]))
DROPOUT_KEEP_PROB = hp.HParam('dropout_keep_prob', hp.Discrete([0.75]))
FREE_BITS = hp.HParam('free_bits', hp.Discrete([0]))
ENCODER_LEARNING_RATE = hp.HParam('encoder_learning_rate', hp.Discrete([0.01]))
ENC_RNN_SIZE = hp.HParam('enc_rnn_size', hp.Discrete([32, 32]))
RESIDUAL_ENCODER = hp.HParam('residual_encoder', hp.Discrete([True]))
DENCODER_LEARNING_RATE = hp.HParam('dencoder_learning_rate', hp.Discrete([0.01]))
COMPUTE_REWARDS_STEP = hp.HParam('compute_rewards_step', hp.Discrete([1]))
DEC_UPDATE_RATE = hp.HParam('dec_update_rate', hp.Discrete([0.80]))
ROLLOUT_NUM = hp.HParam('rollout_num', hp.Discrete([16]))
DIS_LEARNING_RATE = hp.HParam('dis_learning_rate', hp.Discrete([0.0001]))
DIS_TRAIN_FREQ = hp.HParam('dis_train_freq', hp.Discrete([5]))
DIS_FILTER_SIZES = hp.HParam('dis_filter_sizes', hp.Discrete([1, 2, 3, 4, 6, 8]))
DIS_NUM_FILTERS = hp.HParam('dis_num_filters', hp.Discrete([100, 200]))

batch_size = BATCH_SIZE.domain.values[0]
z_size = Z_SIZE.domain.values
seq_len = SEQ_LEN.domain.values[0]
emb_dim = EMB_DIM.domain.values
vocab_size = VOCAB_SIZE.domain.values
hidden_dim = HIDDEN_DIM.domain.values
dropout_keep_prob = DROPOUT_KEEP_PROB.domain.values
free_bits = FREE_BITS.domain.values
encoder_learning_rate = ENCODER_LEARNING_RATE.domain.values
dencoder_learning_rate = DENCODER_LEARNING_RATE.domain.values
dec_update_rate = DEC_UPDATE_RATE.domain.values
enc_rnn_size = ENC_RNN_SIZE.domain.values
residual_encoder = RESIDUAL_ENCODER.domain.values
compute_rewards_step = COMPUTE_REWARDS_STEP.domain.values
rollout_num = ROLLOUT_NUM.domain.values
dis_learning_rate = DIS_LEARNING_RATE.domain.values
dis_train_freq = DIS_TRAIN_FREQ.domain.values
dis_filter_sizes = DIS_FILTER_SIZES.domain.values
dis_num_filters = DIS_NUM_FILTERS.domain.values
hparams = {
    "BATCH_SIZE": batch_size,
    "Z_SIZE": z_size[0],
    "SEQ_LEN": seq_len,
    "EMB_DIM": emb_dim[0],
    "VOCAB_SIZE": vocab_size[0],
    "HIDDEN_DIM": hidden_dim[0],
    "DROPOUT_KEEP_PROB": dropout_keep_prob[0],
    "FREE_BITS": free_bits[0],
    "ENCODER_LEARNING_RATE": encoder_learning_rate[0],
    "ENC_RNN_SIZE": enc_rnn_size[0],
    "RESIDUAL_ENCODER": residual_encoder[0],
    "DENCODER_LEARNING_RATE": dencoder_learning_rate[0],
    "COMPUTE_REWARDS_STEP": compute_rewards_step[0],
    "DEC_UPDATE_RATE": dec_update_rate[0],
    "ROLLOUT_NUM": rollout_num[0],
    "DIS_LEARNING_RATE": dis_learning_rate[0],
    "DIS_TRAIN_FREQ": dis_train_freq[0],
    "DIS_FILTER_SIZES": dis_filter_sizes,
    "DIS_NUM_FILTERS": dis_num_filters,
}

tf.compat.v1.disable_eager_execution()

root = str('/home/mda/PycharmProjects/SEQVAEGANAUDIO_3/Data/seq_vaegan_dir')
seed = int(88)
target_params_file = str('target_params.pkl')
generated_num = int(10000)
start_token = int(0)
positive_file_txt = str('real_data_8.csv')
positive_file_npy = str('real_data_8.csv')
negative_file = str('generator_sample')

token = GetToken(seed, root, start_token, target_params_file, generated_num,
                 positive_file_txt, negative_file)()

target_params_file = token['TARGET_PARAMS']
target_params = pickle.load(open(target_params_file, 'rb'))

discriminator = Discriminator(
    sequence_length=hparams['SEQ_LEN'],
    num_classes=2, vocab_size=hparams['VOCAB_SIZE'],
    embedding_size=hparams['EMB_DIM'],
    filter_sizes=hparams['DIS_FILTER_SIZES'],
    num_filters=hparams['DIS_NUM_FILTERS'],
    learning_rate=hparams['DIS_LEARNING_RATE'])

# CalculateMean cross-entropy loss
gen_data_loader = Gen_Data_loader(batch_size, seq_len)

d_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
losses = tf.nn.softmax_cross_entropy_with_logits

dis_data_loader = Dis_dataloader(batch_size, seq_len)
dis_data_loader.load_train_data('real_data_8_numpy', 'generator_sample')
for _ in range(3):
    dis_data_loader.reset_pointer()
    for _ in range(dis_data_loader.num_batch):
        x_batch, y_batch = dis_data_loader.next_batch()

discriminator.compile(loss=losses, optimizer=d_optimizer)
discriminator.fit([x_batch, y_batch])
