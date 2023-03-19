"""
Interpreting transformers using Discrete Cosine Transformation of hidden states
"""
import sys

import numpy as np
from scipy.fftpack import dct
import tensorflow as tf
# import torch_dct as dct
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForMaskedLM
import matplotlib.pyplot as plt
import matplotlib

def extract_hidden_states(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding='max_length', max_length=512, truncation=True)
    output = model(**inputs, output_hidden_states=True)
    return output.hidden_states

def dct_transform(hidden_states):
    dct_transforms = {}
    for i, hidden_state in enumerate(hidden_states):
        num_neurons = hidden_state.size()[-1]
        dct_trans = np.abs(dct(hidden_state.view(-1, num_neurons).cpu().detach().numpy(), axis=0))
        dct_transforms[i] = dct_trans
    return dct_transforms

def plot_dct_vals(dct_transforms, plot_folder="./plots"):
    to_plot = {}
    for i in dct_transforms:
        transform_i = np.sum(dct_transforms[i], axis=1)
        to_plot[i] = transform_i / transform_i.sum()

    for i, transform in to_plot.items():
        plt.figure(figsize=(7,7))
        bars = plt.bar(x=range(len(transform)), height=transform, color='slateblue')
        plt.ylabel('Percentage weight')
        plt.title('Frequency composition of neuron activations: Layer {}'.format(i))
        plt.ylim(top=0.015)
        plt.savefig(plot_folder + "/plot_" + str(i) + ".png")
        # plt.show()

def plot_dct_bins(dct_transforms, plot_folder="./plots"):
    num_bins = 5
    bins_to_plot = []

    for i in dct_transforms:
        dct_bins = np.array([arr.sum() for arr in np.array_split(dct_transforms[i].sum(axis=1), num_bins)])
        dct_bins = dct_bins / dct_bins.sum()
        bins_to_plot.append(dct_bins.copy())

    freq_bins = ['Very low', 'Low', 'Medium', 'High', 'Very high']
    bar_labels = freq_bins.copy()
    bar_colors = ['red', 'lightblue', 'green', 'orange', 'purple']

    for i, bins in enumerate(bins_to_plot):
        plt.figure(figsize=(7,7))
        bars = plt.bar(freq_bins, bins, label=bar_labels, color='skyblue')
        plt.ylabel('Percentage weight')
        plt.title('Frequency composition of neuron activations: Layer {}'.format(i))
        plt.ylim(top=0.65)
        for j, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .005, "{:.2f}".format(bins[j]))
        plt.savefig(plot_folder + "/dct_bin_" + str(i) + ".png")

def plot_bins(dct_transforms, plot_folder="./plots"):
    num_bins = 5
    bins_to_plot = []
    for i in dct_transforms:
        dcts_sum_across_neurons = np.sum(dct_transforms[i], axis=1)
        dct_bins = np.split(dcts_sum_across_neurons, [2, 9, 34, 130])
        dct_bins = np.array([arr.sum() for arr in dct_bins])
        dct_bins = dct_bins / dct_bins.sum()
        bins_to_plot.append(dct_bins)
        print(dct_bins)
        print(">>>>>>>>")

    font = {'weight' : 'bold',
        'size'   : 19}

    matplotlib.rc('font', **font)

    freq_bins = ['L', 'ML', 'M', 'MH', 'H']
    bar_labels = freq_bins.copy()

    for i, bins in enumerate(bins_to_plot):
        plt.figure(figsize=(5,5))
        bars = plt.bar(freq_bins, bins, label=bar_labels, color='skyblue')
        plt.ylabel('Percentage weight', font=font)
        plt.title('Frequency composition: Layer {}'.format(i + 1), font=font)
        plt.ylim(top=0.85)
        #plt.bar_label(bar_container, fmt=bins)
        for j, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() - 0.001, yval + .005, "{:.2f}".format(bins[j]))
        plt.savefig(plot_folder + "/dct_new_bin_" + str(i) + ".png")
    
    # freq_bins = ['Very low', 'Low', 'Medium', 'High', 'Very high']
    # bar_labels = freq_bins.copy()
    # bar_colors = ['red', 'lightblue', 'green', 'orange', 'purple']

    # for i, bins in enumerate(bins_to_plot):
    #     plt.figure(figsize=(7,7))
    #     bars = plt.bar(freq_bins, bins, label=bar_labels, color='skyblue')
    #     plt.ylabel('Percentage weight')
    #     plt.title('Frequency composition of neuron activations: Layer {}'.format(i))
    #     # plt.ylim(top=0.65)
    #     for j, bar in enumerate(bars):
    #         yval = bar.get_height()
    #         plt.text(bar.get_x(), yval + .005, "{:.2f}".format(bins[j]))
    #     plt.savefig(plot_folder + "/dct_new_bin_" + str(i) + ".png")


def main():
    assert len(sys.argv) > 2, "Need to pass the path to the model as\npython interpreter.py '<path>' '<prompt>'"
    model_path = sys.argv[1]
    prompt = sys.argv[2]

    # Initialize the fine-tuned model
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Extract the hidden state activation values
    hidden_states = extract_hidden_states(model, tokenizer, prompt)

    # Perform dct transform on the hidden state values
    dct_transforms = dct_transform(hidden_states)

    # Plot the dct values
    # plot_dct_vals(dct_transforms)

    # Plot bins
    # plot_dct_bins(dct_transforms)
    plot_bins(dct_transforms)

if __name__ == "__main__":
    main()
