import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import dirichlet
from sklearn.model_selection import train_test_split
import os
class datasets_names:
    hate_speech = "hate-speech"
    goemotions = "goemotions"
    social_bias = "social-bias"




def general_bayesian_baseline(file_path_with_gold, data_set_name, sample_proportion=0.5):
    if (data_set_name == "hate-speech"):
        df = pd.read_csv(file_path_with_gold)
        original_mapping = {
            "example_id": "id",
            "annotator_id": "rater_id",
            "label": "annotator_label"
        }

        df = df.rename(columns=original_mapping)
        print(df['id'])
        data_with_gold = df

    elif (data_set_name == "goemotions"):
        df = pd.read_csv(file_path_with_gold)
        data_with_gold = df

    if (data_set_name == "goemotions"):
        unique_gold_data = data_with_gold.drop_duplicates(subset='id', keep='first')
        prior_probabilities_sampled = data_with_gold['annotator_label'].value_counts(normalize=True).sort_index()
        prior_probabilities_sampled = prior_probabilities_sampled.reindex([0, 1, 2], fill_value=0)
        print(prior_probabilities_sampled)
        sampled_data = data_with_gold

    elif (data_set_name == "hate-speech"):

        prior_probabilities_sampled = data_with_gold['annotator_label'].value_counts(normalize=True).sort_index()
        prior_probabilities_sampled = prior_probabilities_sampled.reindex([0, 0.5, 1], fill_value=0)
        print(prior_probabilities_sampled)
        sampled_data = data_with_gold

    # Prepare the rater labels and example IDs in sampled data
    raters_labels_sampled = sampled_data['annotator_label'].values
    print(len(raters_labels_sampled))

    example_ids_sampled = sampled_data['id'].values
    rater_ids_sampled = sampled_data['rater_id'].values

    # Create a mapping of (id, rater_id) to indices
    example_rater_map_sampled = {(example_ids_sampled[i], rater_ids_sampled[i]): i for i in range(len(example_ids_sampled))}

    # Example data setup
    n_items = len(np.unique(example_ids_sampled))
    n_raters = len(np.unique(rater_ids_sampled))
    n_labels = 3

    # Convert prior probabilities to Dirichlet parameters
    alpha_prior_sampled = prior_probabilities_sampled * 5  # Scaling factor for Dirichlet parameters

    # Function to estimate annotator reliability based on true labels
    def estimate_reliability_constant(raters_labels, true_labels, rater_ids):
        unique_raters = np.unique(rater_ids)

        reliability = np.zeros(len(unique_raters))

        for idx, rater in enumerate(unique_raters):

            rater_indices = [i for i in range(len(rater_ids)) if rater_ids[i] == rater]

            # Filter out out-of-bound indices
            #rater_indices = [i for i in rater_indices if i < len(raters_labels) and i < len(true_labels)]
            correct_labels = (raters_labels[rater_indices] == true_labels[rater_indices])
            reliability[idx] = correct_labels.mean()

        return reliability, unique_raters


    def estimate_reliability_std(raters_labels, example_ids, rater_ids):
        unique_raters = np.unique(rater_ids)
        unique_examples = np.unique(example_ids)
        print(f"raters_labels shape: {raters_labels.shape}")
        print(raters_labels)
        print(f"example_ids shape: {example_ids.shape}")
        print(f"rater_ids shape: {rater_ids.shape}")
        # Create a DataFrame for easier manipulation
        data = pd.DataFrame({'example_id': example_ids, 'rater_id': rater_ids, 'label': raters_labels})
        # Add debug prints

        # Calculate mean label for each example
        example_mean_labels = data.groupby('example_id')['label'].mean()

        reliability = np.zeros(len(unique_raters))

        for idx, rater in enumerate(unique_raters):
            # Get the indices for the current rater
            rater_indices = data[data['rater_id'] == rater].index

            # Get the example IDs and labels for the current rater
            rater_example_ids = data.loc[rater_indices, 'example_id']
            rater_labels = data.loc[rater_indices, 'label']

            # Calculate the mean labels for these examples
            mean_labels = example_mean_labels[rater_example_ids].values

            # Calculate the STD of the rater's labels compared to the mean labels
            reliability[idx] = np.std(rater_labels - mean_labels)

        # Lower STD indicates higher reliability, so we invert the values
        reliability = 1 / (1 + reliability)  # Adding 1 to avoid division by zero

        return reliability, unique_raters

    def update_beliefs_constant(raters_labels, rater_reliability, example_ids, rater_ids, unique_raters, example_rater_map, alpha_prior):
        unique_example_ids = np.unique(example_ids)
        n_items = len(unique_example_ids)
        n_labels = len(alpha_prior)

        posterior = np.zeros((n_items, n_labels))

        example_id_to_idx = {example_id: idx for idx, example_id in enumerate(unique_example_ids)}
        rater_id_to_idx = {rater: idx for idx, rater in enumerate(unique_raters)}

        # Define a weight factor to amplify the impact of observed ratings (e.g., set it to 2 or experiment with values)
        weight_factor = 1  # Increase this to give more weight to observations

        for i, example_id in enumerate(unique_example_ids):
            alpha_post = alpha_prior.copy()

            example_indices = [example_rater_map[(example_id, rater_ids[j])] for j in range(len(rater_ids)) if example_ids[j] == example_id]

            example_indices = [idx for idx in example_indices if idx < len(raters_labels)]

            for idx in example_indices:
                label = raters_labels[idx]
                rater_id = rater_ids[idx]
                rater_idx = rater_id_to_idx[rater_id]  # Use the mapping to get the correct index
                # Amplify the impact of rater reliability
                alpha_post[label] += rater_reliability[rater_idx] * weight_factor

            posterior[i, :] = dirichlet.mean(alpha_post)  # Calculate the mean of the Dirichlet distribution

        true_labels = np.zeros(len(example_ids), dtype=float)

        for j, example_id in enumerate(example_ids):
            idx = example_id_to_idx[example_id]
            true_labels[j] = np.argmax(posterior[idx, :])  # Update true labels to the most probable label
            if data_set_name == "hate-speech":
                true_labels[j] = true_labels[j] * 0.5
                print(f"True label: {true_labels[j]}")

        return posterior, true_labels, unique_example_ids
    for iteration in range(4):
        previous_rater_reliability = np.copy(rater_reliability_sampled) if iteration > 0 else None


        if data_set_name == "goemotions":
            rater_reliability_sampled, unique_raters_sampled = estimate_reliability_std(raters_labels_sampled, example_ids_sampled, rater_ids_sampled)
        elif data_set_name == "hate-speech":
            rater_reliability_sampled, unique_raters_sampled = estimate_reliability_std(raters_labels_sampled, example_ids_sampled, rater_ids_sampled)
        posterior_sampled, new_true_labels_sampled, unique_example_ids_sampled = update_beliefs_constant(
            raters_labels_sampled, rater_reliability_sampled, example_ids_sampled, rater_ids_sampled, unique_raters_sampled, example_rater_map_sampled, alpha_prior_sampled)
        print("new true labels len:" + str(len(new_true_labels_sampled)))

        reliability_changed = not np.array_equal(previous_rater_reliability, rater_reliability_sampled) if previous_rater_reliability is not None else False

        # Update true_labels_sampled for the next iteration
        true_labels_sampled = new_true_labels_sampled.copy()

        # Update the DataFrame with new columns
        sampled_data['updated_gold_label'] = np.nan
        sampled_data['updated_rater_reliability'] = np.nan
        sampled_data['updated_gold_label'] = true_labels_sampled
        for idx, rater in enumerate(unique_raters_sampled):
            sampled_data.loc[sampled_data['rater_id'] == rater, 'updated_rater_reliability'] = rater_reliability_sampled[idx]

        for idx, example_id in enumerate(unique_example_ids_sampled):
            posterior_for_example = posterior_sampled[idx].tolist()  # Convert numpy array to list
            sampled_data.loc[sampled_data['id'] == example_id, 'posterior_distribution'] = sampled_data.loc[sampled_data['id'] == example_id].apply(lambda x: posterior_for_example, axis=1)
        print(f"Iteration {iteration+1}")
        print("Posterior distribution for each item:")
        print(posterior_sampled)

        if reliability_changed:
            print("Rater reliability has changed.")
        else:
            print("Rater reliability has not changed.")

        print("True labels:", true_labels_sampled)
        print("Rater reliability:", rater_reliability_sampled)
        print(sampled_data)
    sampled_data.to_csv("bayesian_method_tagging.csv")
    print("Final estimated true labels:")
    print(true_labels_sampled)
    print(sampled_data['updated_gold_label'].value_counts())
    print("Posterior distribution over true labels for each item:")
    print(posterior_sampled)
    print("Final rater reliability:")
    print(rater_reliability_sampled)


file_path_with_gold = "Your file path"
data_set_name = "the data set name - hate-speech/goemotions/social-bias "

general_bayesian_baseline(file_path_with_gold,data_set_name, 0.5)

