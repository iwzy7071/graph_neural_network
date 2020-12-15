import argparse


class ArgsInit(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='DeeperGCN')
        # dataset
        parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                            help='dataset name (default: ogbg-molhiv)')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='number of workers (default: 0)')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='input batch size for training (default: 32)')
        parser.add_argument('--feature', type=str, default='full',
                            help='two options: full or simple')
        parser.add_argument('--add_virtual_node', action='store_true')
        # training & eval settings
        parser.add_argument('--use_gpu', action='store_true')
        parser.add_argument('--device', type=int, default=0,
                            help='which gpu to use if any (default: 0)')
        parser.add_argument('--epochs', type=int, default=300,
                            help='number of epochs to train (default: 300)')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='learning rate set for optimizer.')
        parser.add_argument('--dropout', type=float, default=0.5)
        # model
        parser.add_argument('--num_layers', type=int, default=3,
                            help='the number of layers of the networks')
        parser.add_argument('--mlp_layers', type=int, default=1,
                            help='the number of layers of mlp in conv')
        parser.add_argument('--hidden_channels', type=int, default=256,
                            help='the dimension of embeddings of nodes and edges')
        parser.add_argument('--block', default='res+', type=str,
                            help='graph backbone block type {res+, res, dense, plain}')
        parser.add_argument('--conv', type=str, default='gen',
                            help='the type of GCNs')
        parser.add_argument('--gcn_aggr', type=str, default='max',
                            help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
        parser.add_argument('--norm', type=str, default='batch',
                            help='the type of normalization layer')
        parser.add_argument('--num_tasks', type=int, default=1,
                            help='the number of prediction tasks')
        # learnable parameters
        parser.add_argument('--t', type=float, default=1.0,
                            help='the temperature of SoftMax')
        parser.add_argument('--p', type=float, default=1.0,
                            help='the power of PowerMean')
        parser.add_argument('--learn_t', action='store_true')
        parser.add_argument('--learn_p', action='store_true')
        # message norm
        parser.add_argument('--msg_norm', action='store_true')
        parser.add_argument('--learn_msg_scale', action='store_true')
        # encode edge in conv
        parser.add_argument('--conv_encode_edge', action='store_true')
        # graph pooling type
        parser.add_argument('--graph_pooling', type=str, default='mean',
                            help='graph pooling method')
        # save model
        parser.add_argument('--model_save_path', type=str, default='model_ckpt',
                            help='the directory used to save models')
        parser.add_argument('--save', type=str, default='EXP', help='experiment name')
        # load pre-trained model
        parser.add_argument('--model_load_path', type=str, default='ogbg_molhiv_pretrained_model.pth',
                            help='the path of pre-trained model')

        self.args = parser.parse_args()

    def save_exp(self):
        return self.args
