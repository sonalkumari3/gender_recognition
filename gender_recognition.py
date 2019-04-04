###@Author Email: sonal.kumari1910@gmail.com


#### Importing Python libraries
import pandas as pd
import numpy as np
import os
import re
from os.path import join, exists, isdir, isfile
import tarfile  ##to untar a directory
import scipy.stats as stats
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt


## Set the value of probability cut-off threshold for prediction: Global variable
Prob_thresh = 0.40



female_pattern = re.compile('[Ff]emale')
male_pattern = re.compile('[Mm]ale')


def get_features(frequencies):
    """
    This function extract different features from the frequency information.
    Parameters
    ----------
        frequencies:      List of dominant frequencies computed from wav files
    Returns: list of features extracted from frequency list
    -------
        mean_freq:  Mean frequency
        std_freq:   Standard deviation of frequency
        median_freq:  Median of frequency
        mode_freq:  Mode of frequency
        peak_freq:  Peak frequency
        Q25:  25th percentile/quantile of frequency
        Q75:  75th percentile/quantile of frequency
        IQR:  interquartile range of frequency
        skew:  skewness of frequency
        kurto:  kurtosis of frequency
    """
    mean_freq = np.mean(frequencies)
    std_freq = np.std(frequencies)
    median_freq = np.median(frequencies)
    mode_freq = stats.mode(frequencies).mode[0]
    peak_freq = np.max(frequencies)
    Q25 = np.percentile(frequencies, 25)
    Q75 = np.percentile(frequencies, 75)
    IQR = Q75 - Q25
    skew = stats.skew(frequencies)
    kurto = stats.kurtosis(frequencies)
    return mean_freq, std_freq, median_freq, mode_freq, peak_freq, Q25, Q75, IQR, skew, kurto


def untar(tar_directory, untar_directory="untar_data", file="output_untar.txt"):
    """
    This function untar all ".tgz" files existing in "tar_directory" directory.
    It also creates a directory named "untar_data" to write the extracted data files in the disk.

    Parameters
    ----------
        tar_directory:      It contains all .tgz data files
        untar_directory:    Name of directory where all untar files will be written
        file:               Name of the file to write the results
    """
    if not exists(tar_directory): file.write("Raw data directory does not exist, download the data or check directory name"+'\n')
    if not exists(untar_directory): os.makedirs(untar_directory)
    for f in os.listdir(tar_directory):
        if f.endswith('.tgz'):
            try:
                tar = tarfile.open(join(tar_directory, f))
                tar.extractall(untar_directory)
                tar.close()
            except:
                file.write("There was an error in opening & extracting the tar file: "+f+'\n')


def preprocess_data(file="output_preprocess_data.txt"):
    '''
    This function untar the data files if not done previously and then extract the class-label and features from wav files present in each directory
    The list of features considered are as follow: 'mean_freq', 'std_freq', 'median_freq', 'Q25', 'Q75', 'IQR', 'skew', 'kurto', 'mode_freq', 'peak_freq', 'label'

    Returns
    -------
        the pandas dataframe with the extracted features from wav files and class label information

    Parameters
    ----------
        file:               Name of the file to write the results
    '''
    ##Path of output data directory to write the extracted data files
    data_directory = './untar_data'

    ##Check if data has been already extracted. If it does not exist, untar .tgz data files. It is one time operation
    if not exists(data_directory): untar('./8kHz_16bit', file)

    ## Copying all the data files in the data_directory
    all_files = [f for f in os.listdir(data_directory) if isdir(join(data_directory, f))]
    number_of_files = len(all_files)
    file.write("Total number of files: " + str(number_of_files) + "\n")

    ## Defining column based on feature-set
    column_list = ['mean_freq', 'std_freq', 'median_freq', 'Q25', 'Q75', 'IQR', 'skew', 'kurto', 'mode_freq',
                   'peak_freq', 'label']
    ## create empty dataframe with column_list
    df = pd.DataFrame(columns=column_list)

    ##Traversing the data directory to extract useful information
    for i in range(number_of_files):
        temp = all_files[i]
        curr_directory = join(data_directory, temp)

        # Extracting gender information from Readme file
        readme_file = join(curr_directory, 'etc', 'README')
        gender = 'Unknown'  ##initialize with Unknown
        if isfile(readme_file):
            for line in open(readme_file):
                if line.startswith("Gender:"):
                    gender = line.split(":")[1].strip()
                    if female_pattern.search(gender):
                        gender = 'Female'
                    elif male_pattern.search(gender):
                        gender = 'Male'

        # Extracting frequency information from wav file
        ## Extracting a combined overall frequency information from all the audio files in a directory
        audio_directory = join(curr_directory, 'wav')
        if isdir(audio_directory):  # few directory does not have wav file
            if os.listdir(audio_directory):
                frequencies_list = []
                for file in os.listdir(audio_directory):  ##traverse the wav directory to extract frequency features
                    try:
                        sampFreq, snd = wavfile.read(join(audio_directory, file))
                        step = int(sampFreq / 5)
                        window_frequencies = []
                        for i in range(0, len(snd), step):
                            ft = np.fft.fft(snd[i:i + step])
                            freqs = np.fft.fftfreq(len(ft))
                            imax = np.argmax(np.abs(ft))
                            freq = freqs[imax]
                            freq_in_hz = abs(freq * sampFreq)
                            window_frequencies.append(freq_in_hz)
                        frequencies_list.append(window_frequencies)
                    except ValueError:
                        file.write("Oops! ValueError: Unexpected end of file. Process next file..."+'\n')
                frequencies = [item for sublist in frequencies_list for item in sublist]
                mean_freq, std_freq, median_freq, mode_freq, peak_freq, Q25, Q75, IQR, skew, kurto = get_features(frequencies)
                ##Appending each data sample in the dataframe
                df = df.append(
                    {'mean_freq': mean_freq, 'std_freq': std_freq, 'median_freq': median_freq, 'Q25': Q25, 'Q75': Q75,
                     'IQR': IQR, 'skew': skew, 'kurto': kurto, 'mode_freq': mode_freq, 'peak_freq': peak_freq,
                     'label': gender}, ignore_index=True)

            else:
                file.write(temp+ " directory does not have any wav file in its etc directory" + '\n')
        else:
            file.write(temp+ " directory does not have etc directory" + '\n')


    ## Saving dataframe to the disk (to reuse it for model tuning)
    df.to_csv('df_wav_features.csv', index=False)
    print("Pre-processing of data successfully completed"+ '\n')
    return df

def prepare_train_test_data(df, features):
    """
        This function split the data in 70:30 trai-test ratio and then standardize the datasets.

        Parameters
        ----------
            df:      pandas dataframe on which splitting happens
            features:      list of features will be used for modelling

        Returns:
        -------
            X_train:  Train-dataset
            X_test:   Test-dataset
            Y_train:  Class-label for train-dataset
            Y_test:  Class-label for test-dataset
    """

    # train-test split is done in 70:30 ratio
    train, test = train_test_split(df, random_state=21, test_size=0.3)
    # print("Train data size is: " + str(train.shape) + '\n')
    # print("Test data size is: " + str(test.shape) + '\n')

    # Standardize the input features before apply any modelling
    scaler = StandardScaler()
    scaler.fit(train[features])
    X_train = scaler.transform(train[features])
    X_test = scaler.transform(test[features])
    Y_train = list(train['label'].values)
    Y_test = list(test['label'].values)
    return X_train, X_test, Y_train, Y_test


def plot_feature_importances(model, X_train, model_name=""):
    '''
    Parameters
    ----------
        model:      data mining model
        X_train:    Train-data used to fit the model
        model_name: Name of the model

    Returns
    -------
        the pandas dataframe with the features and their importance
    '''


    feat_imp = pd.DataFrame({'importance': model.feature_importances_})
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    #
    # feat_imp.sort_values(by='importance', inplace=True)
    # feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title="Feature Importances", figsize=(10, 10))
    plt.xlabel('Feature Importance Score')
    # plt.show()
    plt.savefig(model_name+"_FeatureImportance.png")
    return feat_imp


def model_performance_evaluation(model, X_train, X_test, Y_train, Y_test, model_name="", file="modelling_result_performance_metric.txt"):
    """
    This function predicts the gender for the given tran & test sets and computes different performance metrics for model accuracy.
    Parameters
    ----------
        model:      data mining model for which evaluation will be performed
        X_train:    Train-data used to fit the model
        X_test:     Test-data used to validate the model
        Y_train:    Class label for train-data used to fit the model
        Y_test:     Class label for test-data used to validate the model
        model_name: Name of the model
        file:       Name of the file to write the results
    """
    ##Print the feature-importance : (knn does not have feature_importances_)
    # feat_imp = plot_feature_importances(model, X_train, model_name)
    ### Model accuracy on train set
    Y_pred = model.predict(X_train)
    file.write("Kappa-score on train data: " + str(cohen_kappa_score(Y_train, Y_pred)) + '\n')
    file.write("F1-score on train data: " + str(f1_score(Y_train, Y_pred)) + '\n')

    ### Model accuracy on train set with user defined probability cut-off threshold
    file.write("Train data accuracy on probability threshold: " + str(Prob_thresh) + '\n')
    Y_pred = (model.predict_proba(X_train)[:, 1] > Prob_thresh).astype(int)
    file.write("Kappa-score on train data: " + str(cohen_kappa_score(Y_train, Y_pred)) + '\n')
    file.write("F1-score on train data: " + str(f1_score(Y_train, Y_pred)) + '\n')

    ### Model validation on test set
    Y_pred = model.predict(X_test)
    file.write("\nClassification-report on test data: " + '\n' + classification_report(Y_test, Y_pred) + '\n')
    file.write("Kappa-score on test data: " + str(cohen_kappa_score(Y_test, Y_pred)) + '\n')
    file.write("F1-score on test data: " + str(f1_score(Y_test, Y_pred)) + '\n')

    ### Model accuracy on test set with user defined probability cut-off threshold
    file.write("Test data accuracy on probability threshold: " + str(Prob_thresh) + '\n')
    Y_pred = (model.predict_proba(X_test)[:, 1] > Prob_thresh).astype(int)
    file.write("Kappa-score on test data: " + str(cohen_kappa_score(Y_test, Y_pred)) + '\n')
    file.write("F1-score on test data: " + str(f1_score(Y_test, Y_pred)) + '\n')
    file.write("Modelling is successfully completed" + '\n')

def fit_data_mining_models(X_train, X_test, Y_train, Y_test, file="modelling_result.txt"):
    """
        This function use different data mining models (knn, random-forest, adaboost, and ensemble) for predicting gender.

        Parameters
        ----------
            X_train:  Train-dataset
            X_test:   Test-dataset
            Y_train:  Class-label for train-dataset
            Y_test:  Class-label for test-dataset
    """
    file.write('\n\n'+ "############ k-NEAREST NEIGHBOR CLASSIFIER ############" + '\n')
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model = knn_model.fit(X_train, Y_train)
    model_performance_evaluation(knn_model, X_train, X_test, Y_train, Y_test, "knn_", file)

    file.write('\n\n'+ "############ RANDOM-FOREST CLASSIFIER ###############" + '\n')
    # Initialize the model and set the important parameters
    rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=20, min_samples_split=10,
                                      min_samples_leaf=4, n_jobs=2, random_state=103)
    # Train the model
    rf_model = rf_model.fit(X_train, Y_train)
    ## Evaluate the model performance
    model_performance_evaluation(rf_model, X_train, X_test, Y_train, Y_test, "rf_", file)

    file.write('\n\n' + "############ AdaBoost CLASSIFIER #############" + '\n')
    # Initialize the model and set the important parameters
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=328)
    # Train the model
    adaboost = adaboost.fit(X_train, Y_train)
    ## Evaluate the model performance
    model_performance_evaluation(adaboost, X_train, X_test, Y_train, Y_test, "rf_", file)

    file.write('\n\n' + "########### ENSEMBLE OF ALL CLASSIFIERS ############" + '\n')
    # Initialize the model and set the important parameters
    ensemble_model = VotingClassifier(estimators=[('knn', knn_model), ('rf', rf_model), ('boost', adaboost)], voting='soft', weights=[2, 4, 1])
    # Train the model
    ensemble_model = ensemble_model.fit(X_train, Y_train)
    ## Evaluate the model performance
    model_performance_evaluation(ensemble_model, X_train, X_test, Y_train, Y_test, "rf_", file)


if __name__ == '__main__':

    file = open('gender_recognition_prediction_result.txt', 'w')

    ##If features have not been computed earlier (does not exist on the disk), process the data to extract important features for modelling
    if not exists('df_wav_features.csv'):
        df = preprocess_data()

    ### Load the preprocessed data with class label
    df = pd.read_csv('df_wav_features.csv')


    file.write('\n\n\n\n\n')
    file.write("############# Modeling ################" + '\n')
    features = list(df.columns)

    # extracting only features and removing the lebels
    features.remove("label")

    file.write("Printing feature list" + str(features) + '\n')

    # Prepare data for modeling
    file.write("Total data samples before applying any filtering: " + str(df.shape[0]) + '\n')

    # filtering out unknown data
    df = df[df.label != 'Unknown']
    file.write("Total data samples after filtering data-samples with Unknown label: " + str(df.shape[0]) + '\n')
    file.write("Unique labels in data samples: "+ str(df['label'].unique()) + '\n')

    # Filtering only male and female data
    df = df[(df['label'] == "Male") | (df['label'] == "Female")]

    # checking data reduction after filtering
    file.write("Total data samples after removing data-samples other than Male & Female: " + str(df.shape[0]) + '\n')

    # Label encoding male =0 and female =1
    df.loc[df.label == "Male", 'label'] = 0
    df.loc[df.label == "Female", 'label'] = 1

    X_train, X_test, Y_train, Y_test = prepare_train_test_data(df, features)
    file.write("Starting Modelling...." + '\n')
    fit_data_mining_models(X_train, X_test, Y_train, Y_test, file)

    file.write('\n\n\n\n\n')
    file.write("############# RE-SAMPLING ################" + '\n')

    # checking total number of male and female. Checking whether it is class imbalence problem or not
    file.write("Number of samples in each group: " + str(df.groupby('label').size()) + '\n')

    ## Oversampling the minority class (Female) to deal with class imbalance problem
    df_female_sampled = resample(df[df.label == 1],
                                   replace=True,  # sample with replacement
                                   n_samples=5632,  # to match majority class (Total male after filtering=5632)
                                   random_state=271)  # reproducible results by setting seed

    # Combine majority class with upsampled minority class
    df_sampled = pd.concat([df[df.label == 0], df_female_sampled])

    file.write("Number of samples in each group after re-sampling: " + str(df_sampled.groupby('label').size()) + '\n')

    X_train_sampled, X_test_sampled, Y_train_sampled, Y_test_sampled = prepare_train_test_data(df_sampled, features)

    fit_data_mining_models(X_train_sampled, X_test_sampled, Y_train_sampled, Y_test_sampled, file)

    ## close file after writing the results
    file.close()




