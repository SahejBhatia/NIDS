import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.lines as lines
import seaborn as sns
import pandas as pd
import math
import time
import random
import sys
## sklearn train test split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
## feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import SelectFromModel
# sklearn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Standardization
from sklearn.preprocessing import StandardScaler
# sklearn label encoder import
from sklearn.preprocessing import LabelEncoder
# plot confusion matrix import
from sklearn.metrics import plot_confusion_matrix
# data imbalance undersampling
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics # is used to create classification results
from sklearn.tree import export_graphviz # is used for plotting the decision tree
from six import StringIO # is used for plotting the decision tree
from IPython.display import Image # is used for plotting the decision tree
from IPython.core.display import HTML # is used for showing the confusion matrix
import pydotplus # i
import joblib

# suppress warnings
import warnings
warnings.filterwarnings('ignore')


## model evaluation
def eval_model(name,model, X_train, X_test, y_train, y_test,cols,type="binary"):
    # evaluate model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # joblib.dump(name, type(model).__name__ + '.pkl')
    joblib.dump(name, name+ '.pkl')
    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("**************************************************************************************** \n")
    print('Printing Accuracy report \n')
    print('Accuracy: %.2f' % (accuracy * 100.0))
    print("**************************************************************************************** \n ")
    # calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("**************************************************************************************** \n")
    print('Printing Confusion Matrix \n')
    print('Confusion Matrix:')
    print(cm)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  # annot=True to annotate cells, ftm='g' to disable scientific notation
    print("****************************************************************************************\n")

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');


    # calculate classification report
    if "set" not in type:
        print(f'Classification Report:{name}\n\n')
        print(classification_report(y_test, y_pred,target_names=cols))

        # print('trying to reverse encodeing \n\n')
        # label_encoder.inverse_transform(y_pred)
        # label_encoder.inverse_transform(y_test)
        # print(f'Classification Report on decoded :{name}\n\n')
        # print(classification_report(y_test, y_pred, target_names=cols))


    print("**************************************************************************************** \n")
    print('Printing ROC Curve details \n')
    # calculate roc curve
    if type=="binary":
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        print('ROC AUC: %.2f' % (roc_auc * 100.0))
    else:
        fpr, tpr, thresholds,roc_auc = "NaN", "NaN", "NaN", "NaN"
    

    # f1-score and precision-recall curve
    if type=="binary":
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        # print('ROC AUC: %.2f' % (roc_auc * 100.0))
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        average_precision = average_precision_score(y_test, y_pred)
        # print('F1-Score: %.2f' % (f1 * 100.0))
        # print('Precision: %.2f' % (precision * 100.0))
        # print('Recall: %.2f' % (recall * 100.0))
        # print('Average Precision: %.2f' % (average_precision * 100.0))

        plt.savefig(f'./images/{ax}_binary_confusionMatrix.png')
        print('Saving confusion matrix \n')
        # write the evaluation to a file
        print('saving deatils of this binary run to file \n')
        with open(f'./results/{name}_binary_evaluation.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'Confusion Matrix: \n{cm}\n')
            f.write(f'ROC AUC: {roc_auc}\n')
            f.write(f'F1-Score: {f1}\n')
            f.write(f'Precision: {precision}\n')
            f.write(f'Recall: {recall}\n')
            f.write(f'Average Precision: {average_precision}\n')
            f.write(f'Classification Report: \n{classification_report(y_test, y_pred, target_names=cols))}\n')
    else:
        fpr, tpr, thresholds,roc_auc = "NaN", "NaN", "NaN", "NaN"
        # calculate f1-score and precision-recall curve for multiclass from cm
        f1 = f1_score(y_test, y_pred, average=None)
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        average_precision = fscore
        ## multiclass f1-score
        # print('F1-Score:')
        # for i in range(len(f1)):
        #     print(f'Class {i}: {f1[i]*100.0}')
        # ## multiclass precision-recall curve
        # print('Precision-Recall Curve:')
        # for i in range(len(precision)):
        #     print(f'Class {i}: Precision: {precision[i]*100.0} Recall: {recall[i]*100.0} F1-Score: {fscore[i]*100.0}')
        # ## multiclass average precision
        # print('Average Precision:')
        # for i in range(len(average_precision)):
        #     print(f'Class {i}: {average_precision[i]*100.0}')

#printing desciions tree - for multiclass Descision tree
        if name == 'Multi_Class_Decision_Tree_with_best_feature_selection':
            #print feature importance
            print("\n\n\n****************************************************************************************\n")
            print("Printing Feature Importance for " + name+" : \n")
            importance = model.feature_importances_
            features_importance = list(zip(data.columns, model.feature_importances_))
            features_importance.sort(key=lambda x: x[1])
            print(features_importance)

            plt.style.use('default')
            ax1 = plt.bar([x for x in range(len(importance))], importance)
            # plt.show()
            plt.savefig('./images/' + name +'_FeatureImportance.png')

            dot_data = StringIO()
            export_graphviz(model, out_file=dot_data,
                            filled=True, rounded=True,
                            special_characters=True, feature_names=X.columns)
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            graph.write_png('./images/'+ name + 'CatAttack.png')
            Image(graph.create_png())

        elif name == 'Multi_Class_Random_Forest_with_best_feature_selection':
            print("\n\n****************************************************************************************\n")

            print('calculating feature scores\n')
            feature_scores = pd.Series(model.feature_importances_, index= best_features_selected_names).sort_values(ascending=False)

            # feature_scores

            ax2 = sns.barplot(x=feature_scores, y=feature_scores.index)

            # Add labels to the graph

            plt.xlabel('Feature Importance Score')

            plt.ylabel('Features')

            # Add title to the graph

            plt.title("Visualizing Important Features")

            # Visualize the graph

            # plt.show()
            # ax2.savefig('./images/' + name +'_FeatureImportance.png')
            plt.savefig('./images/' + name +'_FeatureImportance.png')


        print('\n\nsaving evaluation for multiclass to file \n\n')
        # write the evaluation to a file for multiclass
        with open(f'./results/{name}_multiclass_evaluation.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'Confusion Matrix: \n{cm}\n')
            f.write(f'ROC AUC: {roc_auc}\n')
            f.write(f'F1-Score: \n')
            for i in range(len(f1)):
                f.write(f'Class {i}: {f1[i]*100.0}\n')
            f.write(f'Precision-Recall Curve: \n')
            for i in range(len(precision)):
                f.write(f'Class {i}: Precision: {precision[i]*100.0} Recall: {recall[i]*100.0} F1-Score: {fscore[i]*100.0}\n')
            for i in range(len(average_precision)):
                f.write(f'Class {i}: {average_precision[i]*100.0}\n')
            f.write(f'Classification Report: \n{classification_report(y_test, y_pred, target_names=cols)}\n')

    print('saving confusion matrix\n\n')
     # plot confusion matrix,roc curve, precision-recall curve 1 coloumn subplot
    if type == "binary":
        # subplot, 3 row, 1 coloumn
        fig, ax = plt.subplots(3, 1, figsize=(30,20))
        # confusion matrix
        ax[0].set_title('Confusion Matrix', fontsize = 20)
        sns.set(font_scale=3.4) # for label size
        sns.heatmap(cm, annot=True, ax=ax[0])
        # plot_confusion_matrix(model,X_test,y_test, display_labels=['0', '1'] )
        # plot roc curve
        ax[1].set_title('ROC Curve', fontsize = 20)
        ax[1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plot precision-recall curve
        ax[2].set_title('Precision-Recall Curve', fontsize = 20)
        ax[2].plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
        ax[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax[2].set_xlabel('Recall', fontsize = 20)
        ax[2].set_ylabel('Precision', fontsize = 20)
        ax[2].set_ylim([0.0, 1.05])
        ax[2].set_xlim([0.0, 1.0])
        ax[2].legend(loc="lower left", fontsize = 20)

        plt.savefig(f'./results/{name}_binary_evaluation.png')

    else:
        plt.figure(figsize=(30,20))
        # confusion matrix
        plt.title('Confusion Matrix')
        sns.set(font_scale=3.4) # for label size
        sns.heatmap(cm, annot=True, xticklabels=cols, yticklabels=cols)
        plt.savefig(f'./results/{name}_multiclass_evaluation.png')
        
 

    return accuracy


if __name__ == "__main__":
    
    ## cleanup
    # remove old results
    import os
    import shutil
    print ('Welcome !! \n\n')
    print('checking for os and envionment settings \n\n')
    if os.path.exists('./results'):
        shutil.rmtree('./results')
    if os.path.exists('./images'):
        shutil.rmtree('./images')
    # create new results folder
    print("\n\n****************************************************************************************\n\n")
    print('Creating folders for images and results\n\n')


    os.makedirs('./results')
    os.makedirs('./images')

    ###############################################################################
    # Argument Parser
    ###############################################################################
    # first argument is the filename (csv)
    # second argument is the classification models (logistic, svm, tree, randomforest, extratrees, adaboost, gradientboost)

    if len(sys.argv) < 2:
        print("Please provide a filename")
        sys.exit(1)
    if len(sys.argv) < 3:
        print("Please provide a classification model")
        sys.exit(1)
    
    filename = sys.argv[1]
    classifier_method_name = sys.argv[2]

    if classifier_method_name not in ['lr_binary','rf_binary','dt_binary','lr_multi','rf_multi','dt_multi']:
        print("Please provide a valid classification model")
        print ("Valid models are: lr_binary, rf_binary, dt_binary, lr_multi, rf_multi, dt_multi")
        sys.exit(1)

    ###############################################################################
    # Data Loading and Preprocessing
    ###############################################################################
    # load data

    print ('reading file \n\n')
    data = pd.read_csv(filename)

  
    # drop columns "Stime" and "Ltime" (time) , "srcip" and "dstip" as they are not needed
    data = data.drop(['Stime', 'Ltime', 'srcip', 'dstip'], axis=1)
    # "attack_cat" to "non_attack" if empty
    data['attack_cat'].fillna('None', inplace=True)

    # drop na
    data = data.dropna()


    print("\n\n****************************************************************************************\n\n")
    print("Data loaded\n\n")
    # # red color font
    # print("\033[31m")
    # print("Data shape: ", data.shape)
    # print("\033[0m")
    # print("\n")
    # write the shape of the dataframe to a text file
    print('writing data shape to text file \n\n')
    with open(f'./results/{filename}_data_shape.txt', 'w') as f:
        f.write(f'Data shape: {data.shape}\n')
    # print("Data head: ")
    # print(data.head())
    # write the head of the dataframe to a text file
    print('writing dataframe head to file \n\n')
    with open(f'./results/{filename}_data_head.txt', 'w') as f:
        f.write(f'Data head: \n{data.head()}\n')
    # print("\n")

    # print("Data Dtypes: ")
    # print(data.dtypes)
    # write the data types to a csv file
    print('saving cleaned data to file \n\n')
    data.to_csv('./results/data_dtypes.csv', index=False)
    # print("\n")

    print("****************************************************************************************")
    # ###############################################################################
    # # Data Exploration
    # ###############################################################################

    print('  # # Data Exploration \n\n')
    y = data["Label"]
    y2 = data["attack_cat"]

    # plot histogram of the data
    print('ploting histogram of the data\n\n')
    data.hist(figsize=(20,20))
    plt.title("Histogram of the data")
    plt.tight_layout()
    plt.savefig("./images/histogram.png")



    # plot class distribution of y pie chart
    print('saving plot class distribution of Label pie chart to images folder \n\n\n')
    plt.figure(figsize=(20,20))
    # labels should of fontsize=25

    plt.pie(y.value_counts(), labels=y.value_counts().index, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 25})
    plt.title("Class Distribution : Binary Classification")
    plt.savefig("./images/binary_class_distribution.png")

    # plot class distribution of y2 pie chart
    print('saving plot class distribution of Cat attack pie chart to images folder\n\n')
    plt.figure(figsize=(20,20))
    # labels should of fontsize=25
    plt.pie(y2.value_counts(), labels=y2.value_counts().index, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 25})
    plt.title("Class Distribution : Multi Classification")
    plt.savefig("./images/multi_class_distribution.png")

    # plot class distribution of y2 pie chart without the "non_attack" class
    print('saving plot class distribution of Cat attack pie chart without the "non_attack" class to images folder  \n\n')
    plt.figure(figsize=(20,20))
    # labels should of fontsize=25
    plt.pie(y2.value_counts()[1:], labels=y2.value_counts()[1:].index, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 25})
    plt.title("Class Distribution : Multi Classification without non_attack")
    plt.savefig("./images/multi_class_distribution_without_non_attack.png")

    # plot correlation matrix
    print ('plot correlation matrix to images folder \n\n')
    plt.figure(figsize=(20,10))
    sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.1)
    plt.title("Correlation Matrix")
    plt.savefig("./images/correlation_matrix.png")

    
    # ###############################################################################
   
    
    # ###############################################################################
    # Label Encoding
    # ###############################################################################
    # # label encoding
    # # label encoding is used to convert the categorical data into numerical data
    # get all categorical coloumns
    print("****************************************************************************************\n\n")

    print('Performing Label Encoding using sklearn \n\n')
    cat_cols = data.select_dtypes(include=['object']).columns
    # print("Categorical Coloumns: ", cat_cols)
    # convert all categorical coloumns to dtype "str"
    data[cat_cols] = data[cat_cols].apply(lambda x: x.astype('str'))
    # get all numerical coloumns
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    # print("Numerical Coloumns: ", num_cols)

    # write the categorical coloumns and numerical coloumns to a txt file
    print('Writing the categorical coloumns and numerical coloumns to a txt file \n\n')
    with open(f'./results/{filename}_categorical_numerical_coloumns.txt', 'w') as f:
        f.write(f'Categorical Coloumns: {cat_cols}\n')
        f.write(f'Numerical Coloumns: {num_cols}\n')

    # list of "attack_cat" values
    attack_cat_values = data['attack_cat'].unique()

    # label encoding using sklearn
    label_encoder = LabelEncoder()
    for col in cat_cols:
        data[col] = label_encoder.fit_transform(data[col])

    print("\n\n****************************************************************************************\n\n")
    print("Computing\n\n")
    # print("data after label encoding: ")
    # print(data.head())
    #write the data after label encoding to a csv file
    print ('writing the data after label encoding to a csv file \n\n')
    data.to_csv('./results/data_after_label_encoding.csv', index=False)

    # ###############################################################################
    # # Data Seggregation
    # ###############################################################################
    # # seggregate data into training and testing
    # X is the data without "attack_cat" and "Label" column
    # y is the data only "Label" column


    X = data.drop(["attack_cat", "Label"], axis=1)
    y = data["Label"]
    y2 = data["attack_cat"]
    # # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # shape of X_train, X_test, y_train, y_test
    print("\n\n****************************************************************************************\n\n")
    # print("X_train shape: ", X_train.shape)
    # print("X_test shape: ", X_test.shape)
    # print("y_train shape: ", y_train.shape)
    # print("y_test shape: ", y_test.shape)



    # ###############################################################################
    # # Feature Analysis/Selection
    # ###############################################################################
     # # select top 25 features

    print('Performing Feature Analysis/ selection \n\n')
    # # feature selection using chi2
    print('feature selection using chi2\n\n')
    chi2_selector = SelectKBest(chi2, k=25)
    X_train_chi2 = chi2_selector.fit_transform(X_train, y_train)
    X_test_chi2 = chi2_selector.transform(X_test)
    # # feature selection using mutual_info_classif
    print('feature selection using mutual_info_classif \n\n')
    mutual_info_selector = SelectKBest(mutual_info_classif, k=25)
    X_train_mutual_info = mutual_info_selector.fit_transform(X_train, y_train)
    X_test_mutual_info = mutual_info_selector.transform(X_test)
    # # feature selection using f_classif
    print('feature selection using f_classif\n\n')
    f_selector = SelectKBest(f_classif, k=25)
    X_train_f = f_selector.fit_transform(X_train, y_train)
    X_test_f = f_selector.transform(X_test)

    # X_train_fs, X_test_fs, fs = select_features(X_train_f, y_train, X_test_f)
    # # what are scores for the features
    # for i in range(len(fs.scores_)):
    #     print('Feature %d: %f' % (i, fs.scores_[i]))
    # # plot the scores
    # plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    # plt.show()

    # ###############################################################################
    # # Model Training
    # ###############################################################################
    print("****************************************************************************************\n\n")
    print ('# # Model Training \n\n')
    classes=['0', '1']
    feature_selectors = ['chi2', 'mutual_info', 'f_classif']
    # Evaluate Logistic Regression on Binary Classification without feature selection
    print('# Evaluate Logistic Regression on Binary Classification without feature selection\n\n')
    lr = LogisticRegression(random_state=42)
    # eval_model(name,model, X_train, X_test, y_train, y_test,cols,type="binary"):
    fs1_acc = eval_model("Logistic_Regression_without_feature_selection", lr, X_train, X_test, y_train, y_test, classes,"binary.set")

    # Evaluate Logistic Regression on Binary Classification with chi2 feature selection
    print('# Evaluate Logistic Regression on Binary Classification with chi2 feature selection\n\n')
    lr = LogisticRegression(random_state=42)
    fs2_acc = eval_model("Logistic_Regression_with_chi2_feature_selection", lr, X_train_chi2, X_test_chi2, y_train, y_test, classes,"binary.set")

    print('# Evaluate Logistic Regression on Binary Classification with mutual_info_classif feature selection\n\n')
    # Evaluate Logistic Regression on Binary Classification with mutual_info_classif feature selection
    lr = LogisticRegression(random_state=42)
    fs3_acc=eval_model("Logistic_Regression_with_mutual_info_classif_feature_selection", lr, X_train_mutual_info, X_test_mutual_info, y_train, y_test, classes,"binary.set")

    print('# Evaluate Logistic Regression on Binary Classification with f_classif feature selection')
    # Evaluate Logistic Regression on Binary Classification with f_classif feature selection
    lr = LogisticRegression(random_state=42)
    fs4_acc=eval_model("Logistic_Regression_with_f_classif_feature_selection", lr, X_train_f, X_test_f, y_train, y_test, classes,"binary.set")

    print('getting the best feature\n\n')
    # get the best feature selector
    best_fs = max(fs2_acc, fs3_acc, fs4_acc)
    if best_fs == fs2_acc:
        best_fs_name = "chi2"
    elif best_fs == fs3_acc:
        best_fs_name = "mutual_info"
    else:
        best_fs_name = "f_classif"

    print("\n\n****************************************************************************************\n\n")
    print("Best Feature Selector: ", best_fs_name)
    # get the best feature set coloumn names
    best_features_selected = []
    if best_fs_name == "chi2":
        best_features_selected = list(chi2_selector.get_support(indices=True))
    elif best_fs_name == "mutual_info":
        best_features_selected = list(mutual_info_selector.get_support(indices=True))
    else:
        best_features_selected = list(f_selector.get_support(indices=True))
    
    # get the coloumn names of the best feature set
    best_features_selected_names = []
    for i in best_features_selected:
        best_features_selected_names.append(X.columns[i])

    print (best_features_selected_names)
  

    # print("Best Feature Set: ", best_features_selected_names)
    
    # write the best feature selector to a txt file
    with open(f'./results/{filename}_best_feature_selector.txt', 'w') as f:
        f.write(f'Best Feature Selector: {best_fs_name}\n')
        f.write(f'Best Feature Set:\n')
        for i in best_features_selected_names:
            f.write(f'{i}\n')

    # ###############################################################################
    # # Model Training with Chosen Feature Set on Binary Classification
    # ###############################################################################

    # print('\n\n# # Model Training with Chosen Feature Set on Binary Classification\n\n')
    # Evaluate Logistic Regression on Binary Classification with chi2 feature selection
    if classifier_method_name == "lr_binary":
        print('# Evaluate Logistic Regression on Binary Classification with chi2 feature selection\n\n')
        lr = LogisticRegression(random_state=42)
        fs2_acc = eval_model("Binary_Logistic_Regression_with_best_feature_selection", lr, X_train_chi2, X_test_chi2, y_train, y_test, classes,"binary")

    # Evaluate Random Forest on Binary Classification with chi2 feature selection
    if classifier_method_name == "rf_binary":
        print('# Evaluate Random Forest on Binary Classification with chi2 feature selection\n\n')
        rf = RandomForestClassifier(random_state=42)
        fs2_acc = eval_model("Binary_Random_Forest_with_best_feature_selection", rf, X_train_chi2, X_test_chi2, y_train, y_test, classes,"binary")

    # Evaluate Decision Tree on Binary Classification with chi2 feature selection
    if classifier_method_name == "dt_binary":
        print('# Evaluate Decision Tree on Binary Classification with chi2 feature selection\n\n')
        dt = DecisionTreeClassifier(random_state=42)
        fs2_acc = eval_model("Binary_Decision_Tree_with_best_feature_selection", dt, X_train_chi2, X_test_chi2, y_train, y_test, classes,"binary")


    # ###############################################################################
    # # Model Training with Chosen Feature Set on Multi-Class Classification
    # ###############################################################################
    if "multi" in classifier_method_name:
        print('# # Model Training with Chosen Feature Set on Multi-Class Classification\n\n')
        classes = attack_cat_values
        # data for multi-class classification for coloumns matching best_features_selected_names
        data_multi = data[best_features_selected_names]
        # add the attack category column to the data
        data_multi['attack_cat'] = data['attack_cat']
        # split data into train and test
        X = data_multi.drop(columns=['attack_cat'])
        y = data_multi['attack_cat']
        # Standardize the data
        X_std = StandardScaler().fit_transform(X)
        # split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

        # imbalance data undersampling
        undersample = RandomUnderSampler(sampling_strategy='majority')
        X_train, y_train = undersample.fit_resample(X_train, y_train)

        # Evaluate Logistic Regression on Multi-Class Classification with chi2 feature selection
        if classifier_method_name == "lr_multi":
            print('# Evaluate Logistic Regression on Multi-Class Classification with chi2 feature selection\n\n')
            lr = LogisticRegression(random_state=42)
            fs2_acc = eval_model("Multi_Class_Logistic_Regression_with_best_feature_selection", lr, X_train, X_test, y_train, y_test, classes,"multi")
            # fs2_acc = eval_model("Binary_Logistic_Regression_with_best_feature_selection", lr, X_train_chi2,
            #                      X_test_chi2, y_train, y_test, classes, "binary")

        # Evaluate Random Forest on Multi-Class Classification with chi2 feature selection
        if classifier_method_name == "rf_multi":
            print('# Evaluate Random Forest on Multi-Class Classification with chi2 feature selection\n\n')
            rf = RandomForestClassifier(random_state=42)
            fs2_acc = eval_model("Multi_Class_Random_Forest_with_best_feature_selection", rf, X_train, X_test, y_train, y_test, classes,"multi")
            # fs2_acc = eval_model("Binary_Random_Forest_with_best_feature_selection", rf, X_train_chi2, X_test_chi2,
            #                      y_train, y_test, classes, "binary")

        # Evaluate Decision Tree on Multi-Class Classification with chi2 feature selection
        if classifier_method_name == "dt_multi":
            print('# Evaluate Decision Tree on Multi-Class Classification with chi2 feature selection\n\n')
            dt = DecisionTreeClassifier(random_state=42)
            fs2_acc = eval_model("Multi_Class_Decision_Tree_with_best_feature_selection", dt, X_train, X_test, y_train, y_test, classes,"multi")
            # fs2_acc = eval_model("Binary_Decision_Tree_with_best_feature_selection", dt, X_train_chi2, X_test_chi2,
            #                      y_train, y_test, classes, "binary")

#
#test code for ANOVA classification feature selection
# def select_features(X_train, y_train, X_test):
# 	# configure to select all features
# 	fs = SelectKBest(score_func=f_classif, k='all')
# 	# learn relationship from training data
# 	fs.fit(X_train, y_train)
# 	# transform train input data
# 	X_train_fs = fs.transform(X_train)
# 	# transform test input data
# 	X_test_fs = fs.transform(X_test)
# 	return X_train_fs, X_test_fs, fs
#


    
    




    





