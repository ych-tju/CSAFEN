import torch

import utility
import data
import model
import loss
from option import args
from trainer1 import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters())/1000000.0))
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()



# CUDA_VISIBLE_DEVICES=1 python main1.py --rgb_range 1 --save_models --decay 500-1000-1500-2000 --epochs 2500 --save_results --res_scale 0.1 --batch_size 4 --patch_size 48 --lr 2e-4 --model WZH12 --scale 2 --save WZH12x2 --n_resblocks 3 --n_MFMblocks 10 --n_class 32 --data_train DIV2K --data_test Set5+Set14+Urban100+B100

# python main1.py --rgb_range 1 --save_models --decay 500-1000-1500-2000 --epochs 2500 --save_results --res_scale 0.1 --batch_size 4 --patch_size 48 --lr 2e-4 --model CVT --scale 2 --save CVTx2 --data_train DIV2K --data_test Set5+Set14+B100+Urban100
# python main1.py --rgb_range 1 --save_models --decay 500-1000-1500-2000 --epochs 2500 --save_results --res_scale 0.1 --batch_size 4 --patch_size 48 --lr 2e-4 --model CSNLN --scale 2 --save CSNLNx2 --n_feats 128 --depth 12  --data_train DIV2K --data_test Set5

# python main1.py --dir_data dataset/ --model WZH12  --n_resblocks 3 --n_MFMblocks 10 --n_class 32 --data_test Set5+Set14+B100+Urban100 --chop --save_results --rgb_range 1  --scale 2  --res_scale 0.1  --pre_train model_best.pt --test_only

