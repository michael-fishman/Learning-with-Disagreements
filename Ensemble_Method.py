import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Tokenizer, DebertaV2Model
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import warnings
import shutil
from itertools import product

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Define the distribution prediction model
class DistributionPredictor(nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_dim=256, dropout_prob=0.3):
        super(DistributionPredictor, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.relu1 = nn.ReLU()

        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.relu2 = nn.ReLU()

        # Output layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.log_softmax(x)

        return x


# Function to get embeddings for a single batch
def get_embeddings_for_batch(texts, tokenizer, model, device='cpu'):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Get CLS token embeddings
    return embeddings


def calculate_metrics(outputs, ground_truth, threshold=0.4):
    # Apply softmax to get probabilities
    outputs = torch.exp(outputs)

    # Move tensors to CPU and convert to numpy arrays
    ground_truth = ground_truth.cpu().numpy()
    outputs = outputs.cpu().detach().numpy()

    # Calculate predicted and true class indices
    predicted_classes = np.argmax(outputs, axis=1)
    true_classes = np.argmax(ground_truth, axis=1)

    # Calculate accuracy and macro F1 score
    accuracy = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='macro')

    # Threshold the outputs and ground truth to create binary labels for each class
    predicted_thresholded = (outputs > threshold).astype(int)
    true_thresholded = (ground_truth > threshold).astype(int)
    # Compute Class Threshold F1 score (CT F1) per class
    ct_f1_per_class = []
    for class_idx in range(ground_truth.shape[1]):
        ct_f1_class = f1_score(true_thresholded[:, class_idx], predicted_thresholded[:, class_idx])
        ct_f1_per_class.append(ct_f1_class)

    # Compute average CT F1 across all classes
    ct_f1 = np.mean(ct_f1_per_class)

    # Convert numpy arrays back to tensors for CE loss calculation
    outputs_tensor = torch.tensor(outputs).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ground_truth_tensor = torch.tensor(ground_truth).to(outputs_tensor.device)

    # Compute Cross-Entropy Loss (CE)
    ce_loss = F.cross_entropy(outputs_tensor, ground_truth_tensor.argmax(dim=1)).item()

    # Compute Jensen-Shannon Divergence (JSD) using scipy's jensenshannon function
    jsd_sum = 0.0
    for i in range(outputs.shape[0]):
        jsd_sum += jensenshannon(outputs[i], ground_truth[i]) ** 2
    jsd = jsd_sum / outputs.shape[0]

    # Compute KL Divergence (assuming the outputs and ground_truth are in log and prob space)
    log_outputs = torch.log(outputs_tensor + 1e-10)  # Add small epsilon to avoid log(0)
    kl_div = F.kl_div(log_outputs, ground_truth_tensor, reduction='batchmean').item()

    return accuracy, f1, ct_f1, ce_loss, jsd, kl_div


def list_of_strings_to_tensor(lst, device='cpu'):
    lst = [list(map(float, s.strip('[], ').split(', '))) for s in lst]
    return torch.tensor(lst).to(device)


def create_ensemble_labels(df):
    # Determining majority vote based on original distribution
    # majority_labels = df['original_distribution'].apply(lambda x: np.argmax(eval(x)))

    # Convert single labels to numeric values
    majority_labels = pd.to_numeric(df['majority_vote'])
    crowdtruth_labels = pd.to_numeric(df['crowdtruth_single_label'])
    bayesian_labels = pd.to_numeric(df['bayesian_single_label'])

    # Randomly assign each entry to one of the three labeling methods based on the given percentages
    choices = np.random.choice(['majority_vote', 'crowdtruth', 'bayesian'],
                               size=len(df), p=[0.50, 0.25, 0.25])

    # Apply chosen method to each row
    ensemble_labels = []
    for choice, index in zip(choices, df.index):
        if choice == 'majority_vote':
            label = majority_labels.loc[index]
        elif choice == 'crowdtruth':
            label = crowdtruth_labels.loc[index]
        elif choice == 'bayesian':
            label = bayesian_labels.loc[index]
        ensemble_labels.append(label)

    return np.array(ensemble_labels)


def main(dataset_name):
    results_path = f'./results'
    df = pd.read_csv(f'./datasets/final_datasets/{dataset_name}.csv')

    # Split the df to train and test using train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df['ensemble_labels'] = create_ensemble_labels(train_df)
    num_epochs = 15
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Initialize dictionary to store metrics for each combination
    heatmap_data = {
        "Accuracy": pd.DataFrame(index=['majority_vote', 'crowdtruth_single_label', 'bayesian_single_label'],
                                 columns=['ensemble_labels']),
        "F1 Score": pd.DataFrame(index=['majority_vote', 'crowdtruth_single_label', 'bayesian_single_label'],
                                 columns=['ensemble_labels']),
        "CT F1 Score": pd.DataFrame(index=['majority_vote', 'crowdtruth_single_label', 'bayesian_single_label'],
                                    columns=['ensemble_labels']),
        "Cross-Entropy Loss": pd.DataFrame(index=['majority_vote', 'crowdtruth_single_label', 'bayesian_single_label'],
                                           columns=['ensemble_labels']),
        "Jensen-Shannon Divergence": pd.DataFrame(
            index=['majority_vote', 'crowdtruth_single_label', 'bayesian_single_label'], columns=['ensemble_labels']),
        "KL Divergence": pd.DataFrame(index=['majority_vote', 'crowdtruth_single_label', 'bayesian_single_label'],
                                      columns=['ensemble_labels'])
    }

    # Store loss per epoch for plotting
    epoch_losses = {f'ensemble_{test_col}': {'train': [], 'test': []} for test_col in
                    ['majority_vote', 'crowdtruth_single_label', 'bayesian_single_label']}

    # Iterate over all combinations of columns for training and testing
    for train_col, test_col in product(['ensemble_labels'],
                                       ['majority_vote', 'crowdtruth_single_label', 'bayesian_single_label']):
        print(f'Training on {train_col}, Testing on {test_col}')

        # Use specific columns for train and test
        train_df_col = train_df[[train_col]].copy()
        test_df_col = test_df[[test_col]].copy()

        # Convert labels to appropriate format
        train_df_col['label'] = train_df_col[train_col].apply(
            lambda x: [1, 0, 0] if x == 0 else [0, 1, 0] if x == 0.5 else [0, 0, 1])
        test_df_col['label'] = test_df_col[test_col].apply(
            lambda x: [1, 0, 0] if x == 0 else [0, 1, 0] if x == 0.5 else [0, 0, 1])

        tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
        model = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base').to(device)

        train_annotation_distributions = torch.tensor(train_df_col['label'].tolist()).float().to(device)
        test_annotation_distributions = torch.tensor(test_df_col['label'].tolist()).float().to(device)

        output_dim = train_annotation_distributions[0].size()[0]
        input_dim = model.config.hidden_size

        predictor = DistributionPredictor(embedding_dim=input_dim, output_dim=output_dim).to(device)

        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                               verbose=True)

        for epoch in range(num_epochs):
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            predictor.train()
            epoch_train_loss = 0

            for i in range(0, len(train_df_col), batch_size):
                batch_texts = train_df['text'].iloc[i:i + batch_size].tolist()
                batch_labels = train_annotation_distributions[i:i + batch_size]

                train_embeddings = get_embeddings_for_batch(batch_texts, tokenizer, model, device=device)

                optimizer.zero_grad()
                outputs = predictor(train_embeddings)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

            epoch_train_loss /= len(train_df_col)
            epoch_losses[f'ensemble_{test_col}']['train'].append(epoch_train_loss)
            scheduler.step(epoch_train_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}')

            # Evaluate on the test set after each epoch
            predictor.eval()
            epoch_test_loss = 0
            all_test_outputs = []
            all_test_labels = []

            with torch.no_grad():
                for i in range(0, len(test_df_col), batch_size):
                    batch_texts = test_df['text'].iloc[i:i + batch_size].tolist()
                    batch_labels = test_annotation_distributions[i:i + batch_size]

                    test_embeddings = get_embeddings_for_batch(batch_texts, tokenizer, model, device=device)

                    outputs = predictor(test_embeddings)
                    loss = criterion(outputs, batch_labels)
                    epoch_test_loss += loss.item()

                    all_test_outputs.append(outputs)
                    all_test_labels.append(batch_labels)

            epoch_test_loss /= len(test_df_col)
            epoch_losses[f'ensemble_{test_col}']['test'].append(epoch_test_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {epoch_test_loss:.4f}')

        # Aggregate all test outputs and labels for metrics calculation
        all_test_outputs = torch.cat(all_test_outputs)
        all_test_labels = torch.cat(all_test_labels)
        test_accuracy, test_f1, test_ct_f1, test_ce_loss, test_jsd, test_kl_div = calculate_metrics(all_test_outputs,
                                                                                                    all_test_labels)

        # Store metrics in the heatmap data dictionary
        heatmap_data["Accuracy"].loc[test_col, 'ensemble_labels'] = test_accuracy
        heatmap_data["F1 Score"].loc[test_col, 'ensemble_labels'] = test_f1
        heatmap_data["CT F1 Score"].loc[test_col, 'ensemble_labels'] = test_ct_f1
        heatmap_data["Cross-Entropy Loss"].loc[test_col, 'ensemble_labels'] = test_ce_loss
        heatmap_data["Jensen-Shannon Divergence"].loc[test_col, 'ensemble_labels'] = test_jsd
        heatmap_data["KL Divergence"].loc[test_col, 'ensemble_labels'] = test_kl_div

    # Plot heatmaps for each metric in a single figure
    os.makedirs(results_path, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))

    # Set the head of the figure to the name of the dataset
    fig.suptitle(f'{dataset_name} Metrics Heatmaps', fontsize=20)
    axes = axes.flatten()

    for ax, (metric, data) in zip(axes[:-1], heatmap_data.items()):
        sns.heatmap(data.astype(float), annot=True, cmap="viridis", cbar=True, ax=ax)
        ax.set_title(f'{metric} Heatmap')
        ax.set_xlabel('Training Column')
        ax.set_ylabel('Testing Column')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_path, f'{dataset_name}_results.png'))
    plt.show()

    # Plot loss graphs for each combination of train and test columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'{dataset_name} Training and Testing Losses for Each Combination', fontsize=20)

    for ax, (combination, losses) in zip(axes, epoch_losses.items()):
        ax.plot(range(1, len(losses['train']) + 1), losses['train'], marker='o', linestyle='-', color='b',
                label='Train Loss')
        ax.plot(range(1, len(losses['test']) + 1), losses['test'], marker='o', linestyle='-', color='r',
                label='Test Loss')
        ax.set_title(f'Loss: {combination}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_path, f'{dataset_name}_losses.png'))
    plt.show()


if __name__ == '__main__':
    dataset_names = ['measuring-hate-speech', 'goemotions', 'social-bias']
    for i in dataset_names:
        main(i)