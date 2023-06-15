import argparse
import logging
import  torch

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from utils import sample, set_seed
from dataset import CharDataset




logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset_path",default="/home/ltl/snn_gpt/红楼梦.txt",type=str)
    parser.add_argument("--ckpt_path",default="/home/ltl/save_model.pth",type=str)
    parser.add_argument("--block_size",default=128,type=int)
    parser.add_argument("--time_step",default=4,type=int)
    parser.add_argument("--hidden_dim",default=512,type=int)
    parser.add_argument("--num_heads",default=8,type=int)
    parser.add_argument("--depths",default=12,type=int)
    parser.add_argument("--epochs",default=2,type=int)
    parser.add_argument("--batch_size",default=16,type=int)
    parser.add_argument("--learning_rate",default=3e-4,type=float)
    parser.add_argument("--grad_norm_clip",default=1.0,type=float)
    parser.add_argument("--lr_decay",default=0.1,type=float)
    parser.add_argument("--context",default="林黛玉哭着说",type=str)
    parser.add_argument("--steps",default=2000,type=int)
    parser.add_argument("--temperature",default=1.0,type=float)
    parser.add_argument("--sample",default=True,type=bool)
    parser.add_argument("--top_k",default=20,type=int)
    
    args = parser.parse_args()
    return args


def main(args):

    # data
    train_dataset = CharDataset(args.dataset_path, args.block_size)

    # model
    mconfig = GPTConfig(time_step=args.time_step,
                        hidden_dim=args.hidden_dim,
                        vocab_size=train_dataset.vocab_size,
                        block_size=args.block_size,
                        num_heads=args.num_heads,
                        depths=args.depths)
    model = GPT(mconfig)

    # train
    tconf = TrainerConfig(max_epochs=args.epochs,
                          batch_size=args.batch_size,
                          learning_rate=args.learning_rate,
                          grad_norm_clip=args.grad_norm_clip,
                          lr_decay=args.lr_decay,
                          ckpt_path=args.ckpt_path)
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()
    
    # test
    # while True:
    #     context = "how to make a cookie?"
    #     x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)

    #     y = sample(model, x, 2000, temperature=0.9, sample=True, top_k=5)[0]
    #     completion = ''.join([train_dataset.itos[int(i)] for i in y])
    #     print(completion)
    context = args.context
    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
    y = sample(model, x, args.steps, temperature=args.temperature, sample=args.sample, top_k=None)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    print(completion)




if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    
    # seed
    set_seed(_args.seed)
    
    # main
    main(_args)


















