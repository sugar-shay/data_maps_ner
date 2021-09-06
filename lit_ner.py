# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:43:50 2021

@author: Shadow
"""

import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForTokenClassification, AutoConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, classification_report
from pytorch_lightning.callbacks import EarlyStopping
import os 
import itertools

class LIT_NER(pl.LightningModule):
    def __init__(self, 
                 num_classes, 
                 id2tag,
                 tag2id,
                 hidden_dropout_prob=.5,
                 attention_probs_dropout_prob=.2,
                 encoder_name = 'bert-base-uncased',
                 save_fp='best_model.pt'):
       
        super(LIT_NER, self).__init__()
        
        self.num_classes = num_classes
        self.id2tag = id2tag
        self.tag2id = tag2id
        
        self.build_model(hidden_dropout_prob, attention_probs_dropout_prob, encoder_name)
        
        self.training_stats = {'train_losses':[],
                               'val_losses':[],
                               'train_accs':[],
                               'val_accs':[],
                               'gt_probs':[],
                               'correctness':[],
                               'train_labels':[]}
        

        self.save_fp = save_fp
    
    def build_model(self, hidden_dropout_prob, attention_probs_dropout_prob, encoder_name):
        config = AutoConfig.from_pretrained(encoder_name, num_labels=self.num_classes)
        #These are the only two dropouts that we can set
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.encoder = AutoModelForTokenClassification.from_pretrained(encoder_name, config=config)
        
    def save_model(self):
        
        '''
        print()
        print('Class Attributes: ', self.__dict__)
        print()
        '''
        torch.save(self.state_dict(), self.save_fp)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        return outputs

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-5)
        return optimizer
    
    #WE MAY NOT NEED
    def conf_avg(self, active_gt_probs, active_preds, active_labels):
        
        gt_probs, correct = [], []
        for probs, preds, labels in zip(active_gt_probs, active_preds, active_labels):
            
            avg_gt_prob = np.mean(probs)
            
            num_correct = 0
            for pred, label in zip(preds, labels):
                if pred == label:
                    num_correct += 1
            
            correctness = num_correct/len(preds)
            
            #each of these lists have length [batch size, ]
            gt_probs.append(avg_gt_prob)
            correct.append(correctness)
        
        return np.array(gt_probs), np.array(correct)
    
    def conf_token_level(self, active_gt_probs, active_preds, active_labels):
        
        correct = []
        for preds, labels in zip(active_preds, active_labels):
            
            for pred, label in zip(preds, labels):
                if pred == label:
                    correct.append(True)
                else:
                    correct.append(False)
            
            
            #each of these lists have length [total # tokens, ]
            
        active_gt_probs = list(itertools.chain(*active_gt_probs))
        
        return np.array(active_gt_probs), np.array(correct)

    def training_step(self, batch, batch_idx):
        
        #batch['labels'] has shape [batch_size, MAX_LEN]
        #batch['num_slot'] has shape [batch_size]
        
        
        # Run Forward Pass
        outputs = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels = batch['labels'])

        # Compute Loss (Cross-Entropy)
        loss = outputs.loss
        
        logits = outputs.logits
        
        logits = torch.nn.functional.softmax(logits, dim=-1)
        
        #print()
        #print('Labels Shape: ', batch['labels'].shape)
        #print('Num Slots Shape: ', batch['num_slots'].shape)
        
        '''
        print()
        print('Train Loss: ', loss.detach().cpu().numpy())
        
        # Getting logits 
        logits = outputs.logits
        #print('Train logits shape: ', logits.shape)
        
        #Below is the right code to get loss from logits 
        our_loss = None
        loss_fct = CrossEntropyLoss()
        attention_mask = batch['attention_mask']
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_classes)
            active_labels = torch.where(
                active_loss, batch['labels'].view(-1), torch.tensor(loss_fct.ignore_index).type_as(batch['labels'])
            )
            
            #active_logits has shape (batch_size * MAX LEN, # classes)
            #active_labels has shape (batch_size * MAX LEN)
            #active labels has -100, 0, 1, 2 -100 means we ignore that index
            print('Active Logits shape: ', active_logits.shape)
            print('Active Labels Shape: ', active_labels.shape)
            print('Active Labels: ', active_labels.detach().cpu().numpy())
            our_loss = loss_fct(active_logits, active_labels)
        
        print('Our Loss', our_loss.detach().cpu().numpy())
        
        '''
        
        #attention mask has shape [batch_size, max_len]
        attention_mask = batch['attention_mask']
        
        #the below return a tensor that is size [batch_size * max_len] but we want shape [batch_size, max_len]
        #active_loss = attention_mask.view(-1) == 1
        
        #active loss here has shape [batch_size, max_len]
        #active loss is MASK comprised of True (token in seq) or False (padding)
        active_loss = attention_mask == 1
        #print('Active Loss Shape: ', active_loss.shape)
        
        
        #label_masks = batch['labels'] != -100
        #print('Label Mask Shape: ', label_masks.shape)
        #print('Label Mask: ', label_masks)
        
        
        #print()
        
        '''
        active_logits = logits[active_loss]
        print('Active Logits Shape: ', active_logits.shape)
        '''
        
        #were getting the ground truth probs. using amax()
        gt_probs = torch.amax(logits, dim = -1)
        
        #STILL NEED TO MASK PREDS
        preds = torch.argmax(logits, dim = -1)
        #print('GT Probs Shape: ', gt_probs.shape)
        
        #WE NEED ACTIVE PREDS TO HAVE SHAPE [BATCH SIZE, ACTUAL # TOKENS IN EACH SEQUENCE]
        #active loss is the mask for the loss (i.e. removes the padding) => we can also use it to mask for both gt_probs and preds 
        #gt_probs, active_loss, preds = gt_probs.detach().cpu().numpy(), active_loss.detach().cpu().numpy(), preds.detach().cpu().numpy()
        labels = batch['labels'].detach().cpu().numpy()
        gt_probs, preds = gt_probs.detach().cpu().numpy(), preds.detach().cpu().numpy()

        
        
        #the active_loss mask and the id2tag mask that we use have different shapes => active loss doesnt correspond to the number of tokens in the original sequence 
        
        num_slots = batch['num_slots'].detach().cpu().numpy()
        
        active_preds = [[self.id2tag[p] for (p, l) in zip(pred, label) if l != -100] 
              for pred, label in zip(preds, labels)]
    
        active_labels = [[self.id2tag[l] for (p, l) in zip(pred, label) if l != -100]
                  for pred, label in zip(preds, labels)]
        
        active_gt_probs = [[p for (p, l) in zip(prob, label) if l != -100]
                  for prob, label in zip(gt_probs, labels)]
        
        active_gt_probs, active_correct = self.conf_token_level(active_gt_probs, active_preds, active_labels)
        
        acc = accuracy_score(list(itertools.chain(*active_labels)), list(itertools.chain(*active_preds)))

        '''
        #Active_preds is a list of lists we have 32 lists with varying number of tokens in each
        active_gt_probs, active_preds, active_labels = [], [], []
        
        for i in range(gt_probs.shape[0]):
            
            logit_mask = active_loss[i, :]
            #label_mask = label_masks[i, :]
            
            token_probs = gt_probs[i,:]
            token_preds = preds[i,:]
            #seq_labels = labels[i, :]
            
            
            active_gt_probs.append(token_probs[logit_mask].tolist())
            active_preds.append(token_preds[logit_mask].tolist())
            active_labels.append(labels[i, logit_mask].tolist())
            
            print('seq active preds shape: ', token_probs[logit_mask].shape)
            print('seq active labels shape: ', labels[i, logit_mask].shape)
            print('# Token Slots: ', num_slots[i])
        '''
        
        
        # will havve shape 32 because its a list of lists 
        #print('Active GT Probs Shape: ', len(active_gt_probs))
    
        if self.current_epoch == 0:
            active_labels = list(itertools.chain(*active_labels))
            print('Active Labe Head: ', active_labels[:5])
            active_labels = [self.id2tag[label] for label in active_labels]
            return {"loss": loss, 'train_loss': loss, "gt_probs": active_gt_probs, "correct": active_correct, 'train_acc':acc, 'train_labels': active_labels}
        else:
            return {"loss": loss, 'train_loss': loss, "gt_probs": active_gt_probs, "correct": active_correct, 'train_acc':acc}
    
    def training_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.training_stats['train_losses'].append(avg_loss.detach().cpu())
        
        avg_acc = np.stack([x["train_acc"] for x in outputs]).mean()
        self.training_stats['train_accs'].append(avg_acc)
        
        #both of these have shape [# examples]
        gt_probs = np.concatenate([x['gt_probs'] for x in outputs])
        
        correctness = np.concatenate([x['correct'] for x in outputs])
        
        print('GT Probs shape: ', gt_probs.shape)
        print('Correctness shape: ', correctness.shape)
        
        self.training_stats['gt_probs'].append(gt_probs)
        self.training_stats['correctness'].append(correctness)
    
        if self.current_epoch == 0:
            self.training_stats['train_labels'] =  np.concatenate([x['train_labels'] for x in outputs])
        self.log('train_loss', avg_loss)
        
    def validation_step(self, batch, batch_idx):

        # Run Forward Pass
        outputs = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels = batch['labels'])
        
        # Compute Loss (Cross-Entropy)
        loss = outputs.loss
        
        logits = outputs.logits
        
        logits = torch.nn.functional.softmax(logits, dim=-1)
        
        preds = torch.argmax(logits, dim = -1)
        
        labels, preds = batch['labels'].detach().cpu().numpy(), preds.detach().cpu().numpy()
        
        active_preds = [[self.id2tag[p] for (p, l) in zip(pred, label) if l != -100] 
              for pred, label in zip(preds, labels)]
    
        active_labels = [[self.id2tag[l] for (p, l) in zip(pred, label) if l != -100]
                  for pred, label in zip(preds, labels)]
        
        acc = accuracy_score(list(itertools.chain(*active_labels)), list(itertools.chain(*active_preds)))
        
       
        return {"val_loss": loss, 'val_acc': acc}

        
    def validation_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        print('Val Loss: ', avg_loss.detach().cpu().numpy())
        
        avg_loss_cpu = avg_loss.detach().cpu().numpy()
        if len(self.training_stats['val_losses']) == 0 or avg_loss_cpu<np.min(self.training_stats['val_losses']):
            self.save_model()
            
        self.training_stats['val_losses'].append(avg_loss_cpu)
        
        avg_acc =  np.stack([x["val_acc"] for x in outputs]).mean()
        self.training_stats['val_accs'].append(avg_acc)
        
        self.log('val_loss', avg_loss)

        


def train_LitModel(model, train_data, val_data, max_epochs, batch_size, patience = 3, num_gpu=1):
    
    #
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle=False)#, num_workers=8)#, num_workers=16)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle = False)
    
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=False, mode="min")
    
    trainer = pl.Trainer(gpus=num_gpu, max_epochs = max_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)
    
    
    model.training_stats['gt_probs'], model.training_stats['correctness'] = (np.array(model.training_stats['gt_probs'])).T, (np.array(model.training_stats['correctness'])).T
    model.training_stats['train_losses'], model.training_stats['val_losses'] = np.array(model.training_stats['train_losses']), np.array(model.training_stats['val_losses'])
    
    return model


def model_testing(model, test_dataset):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = model.to(device)
    
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    total_preds, total_labels = [], []
    
    model.eval()
    for idx, batch in enumerate(test_dataloader):
        
        seq = (batch['input_ids']).to(device)
        mask = (batch['attention_mask']).to(device)
        labels = batch['labels']
        
        outputs = model(input_ids=seq, attention_mask=mask, labels=None)
        
        logits = outputs.logits
        logits = torch.nn.functional.softmax(logits, dim=-1)
        
        preds = torch.argmax(logits, dim=-1)
        preds = preds.detach().cpu().numpy()
        
        labels = labels.detach().cpu().numpy()
        
        active_preds = [[model.id2tag[p] for (p, l) in zip(pred, label) if l != -100] 
              for pred, label in zip(preds, labels)]
    
        active_labels = [[model.id2tag[l] for (p, l) in zip(pred, label) if l != -100]
                  for pred, label in zip(preds, labels)]
        
        total_preds.extend(active_preds)
        total_labels.extend(active_labels)
    
    #Total Preds is list of lists with length [# sequences] by [# tokens]
    #print('Len of Total Preds: ', len(total_preds))
        
        
    cr = classification_report(list(itertools.chain(*total_labels)), list(itertools.chain(*total_preds)))
    return cr
        

