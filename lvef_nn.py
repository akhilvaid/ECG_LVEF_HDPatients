#!/bin/python

import os

import tqdm
import torch
import torch.nn.functional as F
import torchvision
import pandas as pd
import numpy as np

from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from efficientnet_pytorch import EfficientNet
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from config import Config
from stratifiedgroupkf import StratifiedGroupKFold

torch.manual_seed(Config.random_state)
torch.cuda.manual_seed(Config.random_state)
np.random.seed(Config.random_state)


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, df, output_channels, return_original=False):
        # Create one iterable that get be __getitemed__
        self.image_dir = image_dir
        self.df = df
        self.output_channels = output_channels
        self.return_original = return_original

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.df.iloc[index]['IDENTIFIER'] + '.png')

        clinical_tensor = torch.tensor(
            self.df.iloc[index][['AGE', 'GENDER', 'VENTRATE', 'ATRIALRATE', 'PR', 'QTC']],
            dtype=torch.float32)
        lvef = torch.tensor(
            self.df.iloc[index][['LVEF']],
            dtype=torch.long)

        # Goes to image_tensor below
        image = Image.open(image_path)

        # Transforms
        # Efficientnet B4: 380
        # Efficientnet B5: 456
        if Config.resize:
            resize = torchvision.transforms.Resize(384)
            image = resize(image)

        if self.return_original:  # Does not go to GS/normalization

            gscale = torchvision.transforms.Grayscale(num_output_channels=1)
            image = gscale(image)
            image_tensor = torchvision.transforms.functional.to_tensor(image)

        else:
            normalize0, normalize1 = [0.5], [0.5]

            if self.output_channels == 3:
                normalize0, normalize1 = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

            # Grayscale
            gscale = torchvision.transforms.Grayscale(num_output_channels=self.output_channels)
            image = gscale(image)

            # to_tensor supports PIL images and numpy arrays
            image_tensor = torchvision.transforms.functional.to_tensor(image)
            normalize = torchvision.transforms.Normalize(normalize0, normalize1)
            image_tensor = normalize(image_tensor)

        return clinical_tensor, image_tensor, lvef


class ClinicalMLP(torch.nn.Module):
    def __init__(self, n_clinical_vars):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_clinical_vars, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        return out


class CombinerMLP(torch.nn.Module):
    def __init__(self, feature_modalities):
        super().__init__()
        print('Total feature modalities:', feature_modalities)
        self.fc1 = torch.nn.Linear(64 + 32, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 3)

    def forward(self, clinical_tensor, image_tensor):
        x = torch.cat((clinical_tensor, image_tensor), dim=1)

        out = F.relu(self.fc1(x))
        if Config.dropout_rate:
            out = torch.nn.Dropout(Config.dropout_rate)(out)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out


class Combonet(torch.nn.Module):
    def __init__(self, n_clinical_vars, img_model_type):
        super().__init__()
        # Init the Clinical MLP
        self.fc_clinical = ClinicalMLP(n_clinical_vars)

        # Init the image model
        # This should take care of both classes and channels
        if img_model_type == 'efficientnet':
            print('Image model: Efficientnet B4 (Imagenet)')
            self.image_model = EfficientNet.from_pretrained(
                'efficientnet-b4', in_channels=1, num_classes=64)

        elif img_model_type == 'efficientnetB4UT':
            print('Image model: Efficientnet B4 (Un-pretrained)')
            self.image_model = EfficientNet.from_name(
                'efficientnet-b4', in_channels=1, num_classes=64)

        elif img_model_type == 'efficientnetB4_FT':
            print('Image model: Efficientnet B4 (Starting to Fine tune)')
            self.image_model = EfficientNet.from_name(
                'efficientnet-b4', in_channels=1, num_classes=64)

        else:
            print('Model name not recognized')
            raise NotImplementedError

        # Init the combiner model
        self.combiner = CombinerMLP(2)

    def forward(self, clinical_data, image_data):
        clinical_out = self.fc_clinical(clinical_data)
        image_out = self.image_model(image_data)
        combined_out = self.combiner(clinical_out, image_out)

        return combined_out


class LVEFEstimator:
    def __init__(self, img_model_type, epoch_offset=0, saved_model_path=None):
        self.img_model_type = img_model_type
        print('Image model:', img_model_type)

        # Image channels
        self.output_channels = 1
        if img_model_type.startswith('BiT'):
            self.output_channels = 3

        print('Channels:', self.output_channels)
        self.epoch_offset = epoch_offset

        self.ecg_plot_dir = Config.dir_ecg_plots
        ecg_metrics = Config.file_ecg_metrics
        processed_metrics = Config.file_ecg_metrics_p

        if not os.path.exists(processed_metrics):
            print('Sit back and relax')

            self.df_patients = pd.read_pickle(ecg_metrics)
            available_images = set([i.split('.')[0] for i in os.listdir(self.ecg_plot_dir)])
            self.df_patients = self.df_patients.query('IDENTIFIER in @available_images')

            # Reduce to a non-obscene VentRate
            self.df_patients = self.df_patients.query('45 <= VENTRATE <= 140')

            # Also rearranges columns
            self.preprocess_tabular_data()

            # Save dataframe
            # print('Saving imputed / scaled dataframe')
            self.df_patients.to_pickle(processed_metrics)
        else:
            print('Using processed dataframe')
            self.df_patients = pd.read_pickle(processed_metrics)

        # Just in case
        self.df_patients = self.df_patients.dropna()

        # Outcome
        # Finalize outcome
        self.df_patients['LVEF'] = pd.cut(
            self.df_patients['LVEF'],
            [0, 40, 50, 100],
            labels=[0, 1, 2]
        )

        print('Batch size:', Config.batch_size)

        # Continue training checkpointed model and explainability
        self.saved_model_path = saved_model_path

        self.device = torch.device('cuda')

        # Make directories
        os.makedirs('OutputProbabilities', exist_ok=True)
        os.makedirs('Models', exist_ok=True)

    def image_dataloader_generator(self, df, output_channels):
        image_dataset = CustomImageDataset(self.ecg_plot_dir, df, output_channels)
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=Config.batch_size,
            shuffle=True, num_workers=os.cpu_count() - 1, pin_memory=True)

        return image_dataloader

    def preprocess_tabular_data(self):
        print('Preprocessing dataframe')
        self.df_patients = self.df_patients.set_index(
            ['MRN', 'ECHODate', 'SITENAME', 'IDENTIFIER']).sort_index()
        self.df_patients = self.df_patients.drop(
            ['FILENAME', 'SAMPLERATE'], axis=1)

        # KNN imputation for a dataset this size takes twice the age
        # of the universe. I'm typing this after the big crunch
        if os.path.exists(Config.file_imputation_temp):
            print('Using pre imputed dataframe')
            df = pd.read_pickle(Config.file_imputation_temp)
        else:
            print('Starting imputation')
            df = self.df_patients.copy()
            df = df.drop('TIME_DELTA', axis=1)
            imputed = KNNImputer().fit_transform(df)
            df = pd.DataFrame(imputed, index=df.index, columns=df.columns)
            df.to_pickle(Config.file_imputation_temp)

        # Leave the following out of the scaling
        drop_cols = ['GENDER', 'LVEF']
        df = df.drop(drop_cols, axis=1)

        print('Scaling')
        scaler = pd.read_pickle('Data/StandardScaler.pickle')
        scaled = scaler.transform(df)

        # scaled = power_transform(StandardScaler().fit_transform(df))
        # scaled = MinMaxScaler().fit_transform(df)
        df = pd.DataFrame(scaled, index=df.index, columns=df.columns)

        # Restore to usability
        # NOTE The reset_index is crucial
        drop_cols.append('TIME_DELTA')
        df[drop_cols] = self.df_patients[drop_cols]
        self.df_patients = df.reset_index()

    def eval_model(self, model, dataloader, test_size, mode, epoch=None, fold=99):
        all_lvef = []
        all_pred = []
        all_prob = []

        epoch_loss = 0
        criterion = torch.nn.CrossEntropyLoss()

        model.eval()
        for batch in tqdm.tqdm(dataloader):

            with torch.no_grad():

                with autocast():
                    clinical = batch[0].cuda()
                    waves = batch[1].cuda()
                    lvef = batch[2].cuda()

                    pred = model(clinical, waves)
                    try:
                        loss = criterion(pred.squeeze(), lvef.squeeze())
                    except IndexError:
                        breakpoint()

                    epoch_loss += loss.item() * pred.shape[0]

            all_lvef.extend(lvef.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
            all_prob.extend(torch.softmax(pred, dim=1).cpu().numpy())

        # Testing loss
        eval_loss = epoch_loss / test_size

        y = [np.eye(3)[i[0]] for i in all_lvef]
        y_pred = [i for i in all_prob]
        df_y = pd.DataFrame([y, y_pred]).T
        df_y = df_y.dropna()

        # This is to save space for the sake of the following
        y_true_stacked = np.stack(df_y[0])
        y_pred_stacked = np.stack(df_y[1])

        metrics = {
            'macro_ovo': roc_auc_score(
                y_true_stacked, y_pred_stacked, average='macro', multi_class='ovo'),
            'macro_ovr': roc_auc_score(
                y_true_stacked, y_pred_stacked, average='macro', multi_class='ovr'),
            'weighted_ovo': roc_auc_score(
                y_true_stacked, y_pred_stacked, average='weighted', multi_class='ovo'),
            'weighted_ovr': roc_auc_score(
                y_true_stacked, y_pred_stacked, average='weighted', multi_class='ovr')
        }

        # Save output probabilities
        if epoch is not None:
            os.makedirs(f'OutputProbabilities/{self.img_model_type}', exist_ok=True)
            df_y.to_pickle(
                f'OutputProbabilities/{self.img_model_type}/C_{mode}_F{fold}_E{epoch}.pickle')

        # Just return the one metric to prevent downstream complications
        return eval_loss, metrics['weighted_ovo']

    def get_splits(self, df):
        print('Stratified Group K Fold Splitting')
        gskf = StratifiedGroupKFold(
            n_splits=Config.cross_val_folds,
            shuffle=True,
            random_state=Config.random_state)
        splitter = gskf.split(
            df.drop('LVEF', axis=1),
            df['LVEF'],
            groups=df['MRN'])

        train_test_indices = []
        for train_idx, test_idx in splitter:
            train_test_indices.append((train_idx, test_idx))

        return train_test_indices

    def gaping_maw(self, dict_dataframes, fold):
        print('Starting fold', fold)

        df_train = dict_dataframes['train']
        df_test = dict_dataframes['test']

        # Track performance - Config.patience epochs must be considered before
        # stopping training and moving to the next fold
        performance_track = []

        # Model
        if self.saved_model_path is not None:
            print('Loading saved model for continued training')
            model_l = torch.load(self.saved_model_path, map_location='cpu')
            state_dict = model_l.state_dict()
            prefix = 'module.'
            n_clip = len(prefix)
            adapted_dict = {
                k[n_clip:]: v for k, v in state_dict.items()
                if k.startswith(prefix)}

            # Still architecture dependent for a load
            model = Combonet(6, self.img_model_type)
            model.load_state_dict(adapted_dict)

        else:
            model = Combonet(6, self.img_model_type)

        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model.cuda()

        extreme_metric = 0
        df_vc = df_train['LVEF'].value_counts(normalize=True)
        pos_weight = df_vc.loc[0] / df_vc.loc[1]
        print('Total training samples:', df_train.shape[0], '|', 'pos_weight:', pos_weight)

        criterion = torch.nn.CrossEntropyLoss()

        if self.img_model_type.startswith('big_'):
            optim = torch.optim.SGD(model.parameters(), lr=1e-3)
        else:
            optim = torch.optim.Adam(model.parameters(), lr=1e-4)

        scaler = GradScaler()

        train_dataloader = self.image_dataloader_generator(
            df_train, self.output_channels)
        test_dataloader = self.image_dataloader_generator(
            df_test, self.output_channels)

        for epoch in range(Config.epochs):
            epoch_loss = 0

            model.train()
            for batch in tqdm.tqdm(train_dataloader):

                # Same as optim.zero_grad()
                for param in model.parameters():
                    param.grad = None

                with autocast():
                    clinical_batch = batch[0].cuda()
                    image_batch = batch[1].cuda()
                    lvef_batch = batch[2].cuda()

                    pred = model(clinical_batch, image_batch)
                    loss = criterion(pred.squeeze(), lvef_batch.squeeze())

                    epoch_loss += loss.item() * pred.shape[0]

                # Gradient scaling for AMP
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

            # Overall epoch loss
            training_loss = epoch_loss / df_train.shape[0]

            # Get progress every epoch
            print(f'Evaluating @ Epoch {epoch}')

            testing_loss, testing_metric = self.eval_model(
                model, test_dataloader, df_test.shape[0],
                'Testing', epoch=epoch, fold=fold)

            # Store results
            result_outfile = 'ClassificationResults_LVEF.csv'
            condition = testing_metric > extreme_metric
            model_outfile = f'Models/C_{self.img_model_type}_F{fold}_{testing_metric}.pth'

            if condition:
                extreme_metric = testing_metric
                if Config.save_models:
                    print('Saving model')
                    torch.save(model, model_outfile)

            results = f'Training: {training_loss} | Testing: {testing_metric} | MT: {extreme_metric}'
            print(results)

            df_results = pd.DataFrame([
                self.img_model_type, fold,
                epoch + self.epoch_offset, training_loss,
                testing_loss, testing_metric]).T
            df_results.to_csv(
                result_outfile,
                mode='a',
                header=False)

            # Patience
            performance_track.append(testing_metric)
            if len(performance_track) >= Config.patience + 1:
                perf_eval = performance_track[(len(performance_track) - Config.patience):]

                _yardstick = max(perf_eval)  # Tracking AUROC
                if _yardstick == perf_eval[0]:
                    print(f'Patience threshold exceeded @E {epoch} @TP {perf_eval[0]} > {perf_eval[-1]}')
                    return

    def hammer_time(self):
        df_int = self.df_patients.copy()

        # Stratified cross val per group
        if os.path.exists(Config.file_splits):
            print('Using pickled splits')
            splits = pd.read_pickle(Config.file_splits)
        else:
            splits = self.get_splits(df_int)
            pd.to_pickle(splits, Config.file_splits)

        for fold, (train_idx, test_val_idx) in enumerate(splits):
            df_train = df_int.loc[train_idx]
            df_val_test = df_int.loc[test_val_idx]

            # Deduplicate for test set
            df_val_test = df_val_test.sort_values(['MRN', 'TIME_DELTA'])
            df_val_test = df_val_test.reset_index().groupby(['MRN', 'ECHODate']).first()
            df_val_test = df_val_test.drop_duplicates(subset=['IDENTIFIER'])
            df_val_test = df_val_test.reset_index().set_index('index')

            stratify = df_val_test['LVEF']

            df_test, df_val = train_test_split(
                df_val_test,
                random_state=Config.random_state,
                shuffle=True,
                test_size=0.25,
                stratify=stratify)

            dict_dataframes = {
                'train': df_train,
                'test': df_test,
                'int_val': df_val
            }

            # Train as usual - with awareness of folds and patience
            try:
                self.gaping_maw(dict_dataframes, fold)
            except KeyboardInterrupt:
                print('Continuing to next fold')
                continue
