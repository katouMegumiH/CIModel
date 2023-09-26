The expected structure of files is:
```
#CIModel
 |-- ckpt
 |    |-- REModel
 |    |-- Twitter2015Model
 |    |    |--unimodal
 |    |    |--multimodal
 |    |-- Twitter2017Model
 |    |    |--unimodal
 |    |    |--multimodal
 |-- data
 |    |-- NER_data
 |    |    |--twitter15_caption.csv
 |    |    |--twitter17_caption.csv
 |    |    |-- twitter2015 
 |    |    |    |--unimodal
 |    |    |    |   |-- train.txt
 |    |    |    |   |-- test.txt
 |    |    |    |   |-- valid.txt
 |    |    |    |-- train.txt
 |    |    |    |-- valid.txt
 |    |    |    |-- test.txt
 |    |    |    |-- twitter2015_train_dict.pth 
 |    |    |    |-- ...
 |    |    |-- twitter2015_images
 |    |    |-- twitter2015_aux_images
 |    |    |-- twitter2017
 |    |    |    |--unimodal
 |    |    |    |   |-- train.txt
 |    |    |    |   |-- test.txt
 |    |    |    |   |-- valid.txt
 |    |    |-- twitter2017_images
 |    |    |-- twitter2017_aux_images
 |    |-- RE_data
 |    |    |-- caption.csv
 |    |    |-- img_org          
 |    |    |-- img_vg          
 |    |    |-- txt            
 |    |    |-- ours_rel2id.json
 |-- run_multimodal.py 
 |-- run_unimodal.py 
```
