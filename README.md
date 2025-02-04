# HebrewKWS
keyword spotting in a few-shot scenario.   

pre-prosses and data loaders generation:
This part has two code files, 
pre_prosses_dataloaders.py - 
handles generation from a folder within the root folder of the project.
to generate training dataset, download google commands from touch, then run:
train_loader, val_loader = create_dataloaders(data_dir, validation_split=0.2, batch_size=32)
with data_dir the dircotory where the dataset is w.r.t root folder, default is:
data_dir = 'SpeechCommands/speech_commands_v0.02' 

To generate test loader upload a folder that has diffrent folders for each wanted keyword (at least 5 utternce for each word preferable 1 sec ). then in test_audioproj.py assert the wanted labels and data-dir onto the reqrired testing method (   metrics_gamma1 , etc.. )  

To generate auxliry data, one should first gain a token for https://huggingface.co/ivrit-ai , then  
to load it  use load_ivrit_ai_dataset(token) from general_utils.py , then ivrit_ai_gen_pkl_files(ds,K_cnt=100)
where K_cnt is the amount of .pkl files to use (each .pkl file has 1000 audio files)
paste all of the .pkl files onto folder. 

if you want to generate a triplet dataset, use process_and_dump_files(source_folder, dest_folder) , where the source folder is the folder with .pkl files (with original data). then use  featch_triplet_aux_dataloader(folder_path,batch_size). where the folder path is the dest_folder from process_and_dump_files

Training the model: the training loop is in main_DL_audioproj.py. note that if you want to do training with auxiliary data fill training_type='AUX' , and  aux_val_loader,aux_train_loader should be inserted also. 

Testing the model: 
use test_audioproj.py with the prefeneced method, you could generate your own data or load the .pkl file in the results folder



