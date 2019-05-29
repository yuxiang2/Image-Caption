import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import softmax
import random
import copy

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        b_size = images.size(0)
        features = self.resnet(images).view(b_size, -1, 49)
        features = features.transpose(1,2).contiguous()
        return features
    

class Attention(nn.Module):
    def __init__(self, encoder_size, decoder_size, attn_size):
        super(Attention, self).__init__()        
        self.encoder_query_transformer = nn.Linear(encoder_size, attn_size)
        self.encoder_value_transformer = nn.Linear(encoder_size, attn_size)
        self.decoder_transformer = nn.Linear(decoder_size, attn_size)
        self.encoder_queries = None
        self.encoder_values = None
        
    def forward(self, decoder_hidden):
        decoder_query = self.decoder_transformer(decoder_hidden).unsqueeze(-1) # (B, N, 1)
        attn_energy = torch.bmm(self.encoder_queries, decoder_query) # (B, L, 1)
        attn_energy = attn_energy.squeeze(-1).unsqueeze(1) # (B, 1, L)
        attn_energy = torch.tanh(attn_energy) # (B, 1, L)
        attn_weights = softmax(attn_energy, dim=2) # (B, 1, L)
        context = torch.bmm(attn_weights, self.encoder_values)  # (B, 1, N)
        return context.squeeze(1) # (B,N)
    
    def get_encoder_features(self, encoder_outputs):
        bsize, leng, _ = encoder_outputs.size()
        encoder_outputs_flatten = encoder_outputs.view(bsize * leng, -1)
        self.encoder_queries = self.encoder_query_transformer(encoder_outputs_flatten).view(bsize, leng, -1)
        self.encoder_values = self.encoder_value_transformer(encoder_outputs_flatten).view(bsize, leng, -1)
        
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, attn_size=512, dropout=0.0):
        super(DecoderRNN, self).__init__()
        
        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, embed_size),
            nn.Dropout(dropout, inplace=True)
        )
        self.attention = Attention(2048, hidden_size, attn_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + attn_size, hidden_size + attn_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size + attn_size),
            nn.Linear(hidden_size + attn_size, vocab_size),
        )
        
        self.teacher_force_rate = 1.0
        self.v_size = vocab_size
    
    def step(self, prev_token, last_hidden):
        embedded = self.embed(prev_token).unsqueeze(0)  # (1,B,N)
        embedded = embedded.cuda() 
        outputs, (h,c) = self.rnn(embedded)
        outputs = outputs.squeeze(0)  # (1,B,N) -> (B,N)
        
        # Calculate context vector
        context = self.attention(h[-1])
        
        # Get output logits
        outputs = torch.cat((outputs, context), dim=1) # (B, N + attn_N)
        logits = self.fc(outputs)
        return logits, h
    
    def change_teacher_force_rate(self, multiplier=0.997):
        self.teacher_force_rate = max(0.75, multiplier * self.teacher_force_rate)
    
    def forward(self, features, captions):
        # calculate attention features
        self.attention.get_encoder_features(features)
        prev_token = captions[:,0]
        hidden = None
        
        logits_seqs = []
        for t in range(1, len(captions[0])):
            logits, hidden = self.step(prev_token, hidden)
            if random.random() < self.teacher_force_rate:
                prev_token = captions[:,t]
            else:
                _, prev_token = torch.max(logits, 1)
            logits_seqs.append(logits)
        logits_seqs = torch.stack(logits_seqs, dim=1) # (B, L, V)
        
        return logits_seqs
        

    def sample(self, features, max_len=10, beam_width=5, sos=0, eos=1, unk=2):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        assert(beam_width>2)
        prev_token = torch.LongTensor([sos]).cuda()
        hidden = None
        bad_tokens = (sos, unk)
        
        # best replacing token
        def get_best_token(tokens, bad_tokens):
            for token in tokens:
                if token not in bad_tokens and token != eos:
                    return token
        
        # initial step for seq2seq model
        self.attention.get_encoder_features(features)
        logits, hidden = self.step(prev_token, hidden)
        p = softmax(logits.view(-1), dim=0)
        logp = torch.log(p)
        
        klogp, greedy_kwords = torch.topk(logp, beam_width)
        klogp = klogp.cpu().tolist() 
        greedy_kwords = greedy_kwords.cpu().tolist()
        best_token = get_best_token(greedy_kwords, bad_tokens)
        
        bestk_paths = []
        for logp,init_word in zip(klogp,greedy_kwords):
            if init_word in bad_tokens:
                bestk_paths.append(Beam_path(eos, logp, init_word, hidden, best_token))
            else:
                bestk_paths.append(Beam_path(eos, logp, init_word, hidden))
                
        # initial step done, steps afterwards
        for i in range(1, max_len):
            new_paths = []
            for beam_path in bestk_paths:
                # if the beam path has ended, don't expand it
                if beam_path.is_done():
                    new_paths.append(beam_path)
                    continue
                
                prev_hidden = beam_path.prev_hidden
                prev_token = torch.LongTensor([beam_path.prev_word]).cuda()
                logits, hidden = self.step(prev_token, prev_hidden)
                p = softmax(logits.view(-1), dim=0)
                logp = torch.log(p)
                klogp, greedy_kwords = torch.topk(logp, beam_width)
                klogp = klogp.cpu().tolist()
                greedy_kwords = greedy_kwords.cpu().tolist()
                best_token = get_best_token(greedy_kwords, bad_tokens)
                new_paths.extend(beam_path.get_new_paths(greedy_kwords, bad_tokens, best_token, klogp, hidden))
                
            bestk_paths = get_bestk_paths(new_paths, beam_width)
            
        best_idx = -1
        best_score = -99999.99
        for i,path in enumerate(bestk_paths):
            if len(path.path) == max_len or len(path.path) == 1:
                path.score = path.score - 100.0
            if path.score > best_score:
                best_score = path.score
                best_idx = i
            
        return bestk_paths[best_idx].path
        

def get_bestk_paths(paths, k):
    sorted_paths = sorted(paths, key=lambda x: x.score / len(x.path))
    return sorted_paths[-k:]

        
class Beam_path(object):
    def __init__(self, eos=None, logp=0, cur_word=None, prev_hidden=None, replace_word=None):
        self.score = logp
        self.path = [cur_word] if replace_word == None else [replace_word]
        self.prev_word = cur_word
        self.prev_hidden = prev_hidden 
        self.eos = eos
    
    def _copy(self):
        path = Beam_path()
        path.score = self.score 
        path.path = copy.copy(self.path)
        path.eos = self.eos
        return path
        
    def _update(self, cur_word, logp, hidden, replace_word=None):
        self.score += logp
        self.path.append(cur_word if replace_word == None else replace_word)
        self.prev_word = cur_word 
        self.prev_hidden = hidden
        
    def is_done(self):
        return self.prev_word == self.eos
        
    def get_new_paths(self, branches, bad_tokens, replace_word, logps, hidden):
        N = len(branches)
        new_paths = []
        for i in range(N):
            new_paths.append(self._copy())
        for new_path,branch,logp in zip(new_paths,branches,logps):
            if branch in bad_tokens:
                new_path._update(branch,logp,hidden,replace_word)
            else:
                new_path._update(branch,logp,hidden)
        return new_paths
        
    def __repr__(self):
        return str(self.path) + str(self.score)