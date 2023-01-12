import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import clip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class BaseDataset(Dataset):

    def __init__(self, args):
        self.args = args

    def getBottomUpAttentionFeature(self, ID):
    
        data = np.load(os.path.join(self.args.image_bottom_up_attention_feature_root, ID + '.npz'))
        precomp_data = data['x']
        max_feature_length = 50

        delta = max_feature_length - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:max_feature_length]

        return torch.tensor(precomp_data.astype(np.float32)) 
        



class DPC2022(BaseDataset):
    
    def __init__(self, df, text_tokenizer, args):
        super(DPC2022, self).__init__(args)

        self.args = args
        self.data_root = args.data_root
        self.img_root = args.image_root
        self.max_sentence_length = args.max_sentence_length
   
        self.text_tokenizer = text_tokenizer
        # 过滤一些无用的数据

        if args.batch_weighted_ce == 'cs_csocre':
            df = df[df['sum_score']>1]
        self.data = df

        


    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):


        data_row = self.data.iloc[idx]
        
        ID = str(int(data_row['ID']))
        # ID = '128'
        

        imgfeat = self.getBottomUpAttentionFeature(ID)

        # if self.args.batch_weighted_ce == 'constant':
        cscore = torch.ones(1).expand(1, self.max_sentence_length-1)
        
        if self.args.batch_weighted_ce == 'cs_csocre':
            cscore = self.getCommentInformativeScores(torch.tensor(data_row['sum_score']), torch.tensor(data_row['norm_sum_score'])).expand(1, self.max_sentence_length-1)
        

        comment = self.text_tokenizer.bos_token + ' ' + str(data_row['comment']) + ' ' + self.text_tokenizer.eos_token
        
        
        return imgfeat, comment, cscore, ID 
     

    def collate_fn(self, batch):
        
        # images, ascores, comments, cscores, IDs =  list(zip(*batch))
        imgfeats, comments, cscores, IDs =  list(zip(*batch))
        
        cscores = torch.cat(cscores, dim=0)
        imgfeats = torch.stack(imgfeats, dim=0)

        text_encoding = self.text_tokenizer.batch_encode_plus(
            list(comments),
            add_special_tokens=True,
            max_length=self.max_sentence_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        
        if self.args.batch_weighted_ce == 'length':
            sentence_lenth = text_encoding['attention_mask'].sum(dim=1)
            cscores = self.getLengthScores(sentence_lenth).unsqueeze(1).expand(text_encoding['attention_mask'].size(0), self.max_sentence_length-1)

        return dict(
            imgfeats=imgfeats,
            text_encoding=text_encoding,
            cscores=cscores,
            IDs=IDs,
        )

    def getLengthScores(self, length):
        
        score = length * (9/38)+10/9

        return 1+10*torch.clamp(score, 0, 1)

    def getCommentInformativeScores(self, cscore, norm_cscore):
        
        # mean=2.1351
        # std=1.9914
 
        return 1+10*torch.clamp(norm_cscore, 0, 1)
        

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class DPC2022_for_generate(BaseDataset):
    
    def __init__(self, df, gpt2_tokenizer, args):
        super(DPC2022_for_generate, self).__init__(args)

        self.args = args
        self.data_root = args.data_root
        self.img_root = args.image_root
        self.max_sentence_length = args.max_sentence_length
        self.gpt2_tokenizer = gpt2_tokenizer
        self.clip_tokenizer = clip.tokenize

        # clip_preprocessing
        self.clip_preprocessing = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.args.imgSize,self.args.imgSize)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.data = df



    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):


        data_row = self.data.iloc[idx]
        ascore = torch.tensor(data_row['score'])
        ID = str(int(data_row['ID']))
        img_path = os.path.join(self.img_root, ID+'.jpg')
        image = Image.open(img_path).convert("RGB")
        clip_image = self.clip_preprocessing(image)
        image = self.transform(image)

        imgfeat = self.getBottomUpAttentionFeature(ID)
       

        return imgfeat, image, clip_image, ascore, ID
        # return dict(
        #     images=image,
        #     ascores=example['ascore'],
        #     comments=example['comment'],
        #     cscores=example['cscore'],
        #     IDs=example['ID'],
        # )

    def collate_fn(self, batch):

        imgfeats, images, clip_images, ascores, IDs =  list(zip(*batch))

        images = torch.stack(images, dim=0)
        clip_images = torch.stack(clip_images, dim=0)
        imgfeats = torch.stack(imgfeats, dim=0)
        ascores = torch.stack(ascores, dim=0)
        
        return dict(
            imgfeats = imgfeats,
            images = images,
            clip_images = clip_images,
            IDs=IDs,
            ascores = ascores,
        )

    def readData_from_id(self, ID):

        comment_list = []
        cscore_list = []
        norm_cscore_list = []
        with open(os.path.join(self.data_root, ID+'.json'), 'r') as f:
            data_dict = json.load(f)


        # score comments
        comment_dict_list = data_dict['score_comments']
        for i in range(len(comment_dict_list)):
            comment = str(comment_dict_list[i]['comment']['comment']).lower()
            cscore = comment_dict_list[i]['sum_score']
            norm_sum_score = comment_dict_list[i]['norm_sum_score']
            # 过滤掉无用的comment
            # if cscore > 1:
            comment_list.append(comment)
            cscore_list.append(cscore)
            norm_cscore_list.append(norm_sum_score)

        return comment_list, cscore_list, norm_cscore_list

    def readGenData_from_id(self, data_root, ID):

        comment_list = []
        cscore_list = []
        norm_cscore_list = []
        with open(os.path.join(data_root, ID+'.json'), 'r') as f:
            data_dict = json.load(f)

        comment_list = data_dict['gen_comments']

        return comment_list
            
                
