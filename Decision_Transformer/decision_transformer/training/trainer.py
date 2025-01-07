import numpy as np
import torch
import os

import time
# from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('Decision_Transformer/tb_record/')
class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, iter, env_name, model_type, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for i in range(num_steps):
            train_loss = self.train_step()
            #writer.add_scalar(str(iter)+ '/' + env_name + '/' + model_type + '/loss/training', train_loss, i)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        eval_start = time.time()

        # save model
        save_path = 'Decision_Transformer/preTrained/{}'.format(env_name)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), save_path+'/{}.pth'.format(iter))
        
        #self.model.eval()
        #for eval_fn in self.eval_fns:
        #    outputs = eval_fn(self.model)
        #    for k, v in outputs.items():
        #        logs[f'evaluation/{k}'] = v
        #        #writer.add_scalar(env_name + '/' + model_type + f'evaluation/{k}', v, iter)

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        #writer.add_scalar(env_name + '/' + model_type + '/loss/train_loss_mean', np.mean(train_losses), iter)
        #writer.add_scalar(env_name + '/' + model_type + '/loss/train_loss_mean', np.std(train_losses), iter)
        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
