import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel

from models.transformer.gpt_decoder_visualGPT import GPT2LMHeadModel
from models.transformer.config import GPT2Config
#
from models.transformer.load_gptmodel import load_weight
from transformers import GPT2Tokenizer

state_dict = torch.load('./saved_models/gpt2/gpt2-pytorch_model.bin', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')


class Transformer_visualgpt(CaptioningModel):

    def __init__(self, encoder, gpt2_type='gpt',n_layer=12,tau=0):
        super(Transformer_visualgpt, self).__init__()
        
        # n_layer=4
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({'pad_token': "+="})
        tokenizer.add_special_tokens({'bos_token': "<?"})
        self.tokenizer = tokenizer

        self.vocab_size= tokenizer.vocab_size
        self.bos_idx = tokenizer.bos_token_id
        self.eos_idx = tokenizer.eos_token_id
        self.pad_idx = tokenizer.pad_token_id
        # self.encoder = nn.DataParallel(encoder)
        self.encoder = encoder
        self.gpt2_type = gpt2_type

        self.config = GPT2Config()
        self.config.n_layer = n_layer
   
        if gpt2_type =="random":

             
            decoder= GPT2LMHeadModel(self.config, tau=tau)
            self.decoder = decoder

        else:

            decoder = GPT2LMHeadModel(self.config,tau=tau)
            decoder = load_weight(decoder, state_dict)
                        
            self.decoder = decoder

        self.vocab_size = self.config.vocab_size
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        
        if self.gpt2_type =="random":
            for p in self.parameters():
                if p.dim()>1:
                    nn.init.xavier_uniform_(p)
        else:
            for p in self.encoder.parameters():
                if p.dim()> 1:
                    nn.init.xavier_uniform_(p)




    def forward(self, images, seq, *args):
        enc_output, mask_enc = self.encoder(images)
        # enc_output, mask_enc = self.encoder.module(images)

        dec_output, past = self.decoder(seq, enc_output, mask_enc)
        return dec_output,past

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]


    def sample(self, visual, seq_len, prev_ids=None, **kwargs):

        batch_size = visual.shape[0]
        output_word = None
        past = None 
        output_word_probs = None
         
        if prev_ids != None:
            prev_ids_seq_len = prev_ids.size(1)
            output_ids = prev_ids[:, 1:]
            # remove bos token
            if prev_ids_seq_len > seq_len:
                raise Exception("prev_ids_seq_len can not longer than seq_len")

        else:
            output_ids = prev_ids

        if prev_ids == None:
            for t in range(seq_len):
                word_logprob, past = self.step(t, output_word, visual, None, past, mode='feedback', **kwargs)
                word_logprob = word_logprob.view(batch_size, self.vocab_size)
                output_word = word_logprob.exp().multinomial(1)
                output_word_prob = torch.gather(word_logprob, 1, output_word)
                if output_ids == None:
                    output_ids = output_word  
                    output_word_probs = output_word_prob  
                else:
                    output_ids = torch.cat((output_ids, output_word), dim=-1)
                    output_word_probs = torch.cat((output_word_probs, output_word_prob), dim=-1)  

                # print(word_logprob.shape, output_word.shape, output_ids.shape)

        # when prev_ids_seq_len < seq_len + bos_token
        elif prev_ids != None and prev_ids_seq_len < seq_len + 1:

            log_prob, past = self.forward(visual, prev_ids)
            # prev_ids_lis = prev_ids.chunk(prev_ids.size(1), dim=1)
            # for t in range(prev_ids_seq_len):
                # word_logprob, past = self.step(t, prev_ids_lis[t], visual, None, past, mode='feedback', **kwargs)

            word_logprob = log_prob[:, -1]
            output_word = word_logprob.exp().multinomial(1)
            output_word_prob = torch.gather(word_logprob, 1, output_word)

            # add last token since remove bos token
            for t in range(prev_ids_seq_len, seq_len+1):
                output_ids = torch.cat((output_ids, output_word), dim=-1)
                if output_word_probs == None:
                    output_word_probs = output_word_prob
                else:
                    output_word_probs = torch.cat((output_word_probs, output_word_prob), dim=-1)
                word_logprob, past = self.step(t, output_word, visual, None, past, mode='feedback', **kwargs)
                word_logprob = word_logprob.view(batch_size, self.vocab_size)
                output_word = word_logprob.exp().multinomial(1)
                output_word_prob = torch.gather(word_logprob, 1, output_word)
                
        # return output_ids output_word_probs
        return output_ids


    def step(self, t, prev_output, visual, seq, past, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(visual)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                # 如果第一次输入, 那就产生一次enc_output和mask_enc
                if self.enc_output == None and self.mask_enc == None:
                    self.enc_output, self.mask_enc = self.encoder(visual)

                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc,past=past)

    

