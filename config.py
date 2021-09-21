#!/bin/python

class Config:
    dir_ecg_xml = 'Data/ECGs'
    dir_ecg_plots = 'Data/Plotted'

    file_ecg_query = 'Data/ECGQuery.pickle'
    file_ecg_metrics = 'Data/DerivedMetrics.pickle'
    file_ecg_metrics_p = 'Data/DerivedMetricsProcessed.pickle'
    file_imputation_temp = 'Data/DerivedMetricsImputed.pickle'
    file_scaling = 'Data/LeadScaling.pickle'  # Per lead scaling metrics
    file_splits = 'Data/Splits.pickle'

    ecg_echo_timedelta = 15
    ecg_dialysis_timedelta = 15

    # Dataset
    plot_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # External validation - by facility
    ext_val_hospital = None

    # Training related
    random_state = 42
    resize = True
    dropout_rate = False
    epochs = 301
    save_models = True
    patience = 3
    cross_val_folds = 5

    # Batch sizes
    batch_size = 150

    # Debug
    debug = False
