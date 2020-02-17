"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
import tensorflow as tf
#from common_private import  ModelOwner, LogisticRegression, XOwner, YOwner
from common_private import  LogisticRegression, XOwner, YOwner
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import argparse
import sys
import time



def run(input):
    batch_size = 100
    epoch_num=2


    #data_owner_0 = XOwner(   batch_size )
    #data_owner_1 = YOwner( batch_size )

    #train_batch_num = epoch_num*data_owner_0.sample_num// batch_size
    train_batch_num=100
    #test_batch_num = data_owner_0.test_sample_num // batch_size
    test_batch_num=30

    if len(sys.argv) >= 2:
      # config file was specified
      config_file = sys.argv[1]
      config = tfe.RemoteConfig.load(config_file)
    else:
      # default to using local config
      config = tfe.LocalConfig([
          'XOwner',
          'YOwner',
          'RS'
      ])
    tfe.set_config(config)
    players = ['XOwner', 'YOwner', 'RS']
    prot = tfe.protocol.SecureNN(*tfe.get_config().get_players(players))
    tfe.set_protocol(prot)
    session_target = sys.argv[2] if len(sys.argv) > 2 else None




    # tfe.set_protocol(tfe.protocol.Pond(
    #     tfe.get_config().get_player(data_owner_0.player_name),
    #     tfe.get_config().get_player(data_owner_1.player_name)
    # ))

    @tfe.local_computation("XOwner")
    def provide_training_data(path):
        """Preprocess training dataset

        Return single batch of training dataset
        """
        train_x = get_embed_op_5w_x(batch_size=self.batch_size, test_flag=False)
        return train_x
    x_train = data_owner_0.provide_training_data()
    y_train = data_owner_1.provide_training_data()

    x_test = data_owner_0.provide_testing_data()
    y_test = data_owner_1.provide_testing_data()



    print("x_train:", x_train)
    print("y_train:", y_train)

    print("x_test:", x_test)
    print("y_test:", y_test)



    model = LogisticRegression(data_owner_0.feature_num,learning_rate=0.01)

    with tfe.Session() as sess:

      sess.run(tfe.global_variables_initializer(),
               tag='init')
      start_time=time.time()
      model.fit(sess, x_train, y_train, train_batch_num)

      train_time=time.time()-start_time
      print("train_time=", train_time)


      model.get_KS(sess, x_test, y_test, test_batch_num)
      #sess.run(reveal_weights_op, tag='reveal')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1E-2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--regularizationL2', type=float, default=1E-3)
    parser.add_argument('--maxIter', type=int, default=5000)
    #parser.add_argument('--used_data_percent', type=float, default=1.0)

    args = parser.parse_args()