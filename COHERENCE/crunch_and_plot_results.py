import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from pathlib import Path
import json
import time
import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
from openai import OpenAI
from test_consistency import metadata2string
#from scipy.stats import pearsonr
import numpy as np

# A list of indeterminate or incorrect phrases in the data set.
improper_statements = ["plant is a tall plant",\
            "plant is capable of flower in spring", "plant is a angiosperm", "tree is a angiosperm",\
            "fruit is capable of grow in garden", "tree is a important food source", "fruit is a good snack",\
            "fruit is a sedative and pain killer", "plant is a reproductive structure",\
            "plant is capable of shade car", "forrest", "wort", "tree has a stump", "anemone", "bloomer",\
            "reed is capable of grow in garden", "composite is part of angiosperm", "bird desires fly",\
            "orchid is capable of flower in spring", "plant is a woody plant", "plant is a good snack", "snag",\
            "plant is a important food group", "sang", "contrivance", "pod", "may has a", "may is a", "opposite",\
            "plant is a depressant", "a plant is capable of grow in garden", "section", "is a actor", "quack-quack",\
            "plant is capable of grow in garden", "slug", "tree is part of forest", "stock is a flower",\
            "escalator is a music", "alcohol is capable of slow thinking", "ballet is a music", "score is a music",\
            "fruit is capable of destroy friendships", "plant is a part of plant", "composite is capable of"]
incorrect_validities = ["seed is a fruit", "plant is a building complex", "bird is a organism", "bird has a clapper",\
            "bird has a snout", "dodo desires fly", "bird has a lip", "plant has a organelle",\
            "fruit is not a animal", "bird is capable of grow", "fruit has a nucleus", "plant has a stump",\
            "flower is a organism", "plant has a nucleus", "flower is not a animal", "bird has a cell",\
            "bird has a nucleus", "flower has a vacuole", "fruit has a vacuole", "fruit has a organelle",\
            "fruit is a organism", "plant is a plant", "bird is capable of sound good", "ling is a plant",\
            "fruit is capable of suffering dehydration", "music is not a animal", "realisation is a music",\
            "plant is capable of bloom", "arbor is a tree", "bird is a important food group",\
            "flower is capable of suffering dehydration", "tree is a important food group", "tree is a good source of vitamins",
            "plant is a source of vitamin c", "bird is a good source of vitamins", "plant is a important food source",\
            "plant is a better snack than candy", "plant is capable of shade from sun", "fruit is capable of flower in spring",\
            "bird is capable of express feelings", "plant is part of angiosperm", "fruit is capable of branch out",\
            "flower is capable of grow in garden", "plant is capable of cloud judgement", "chapter is a music",\
            "bird has a organelle", "bird is a important food source", "flower is a source of vitamin c",
            "fruit is part of angiosperm", "fruit is a part of plant", "flower is a good source of vitamins",\
            "bird is a better snack than candy", "plant is part of forest", "flower has a xylem",\
            "bird has a vacuole", "plant is capable of shade lawn", "plant is capable of branch out",\
            "flower is capable of bloom", "plant has a ear", "fruit is capable of grow leaves", "fruit is capable of grow",\
            "fruit has a ear", "bird is a good snack", "tree has a vacuole", "fruit has a head","plant has a head",\
            "plant is capable of grow leaves", "plant has a cell", "fruit has a cell", "bird is capable of suffering dehydration",\
            "plant has a vacuole", "tree is a sedative and pain killer", "fruit is capable of drop leaves", "bird is part of forest",\
            "plant has a trunk", "tree is a source of vitamin c", "flower is capable of flower in spring",\
            "plant is capable of drop leaves", "fruit is a angiosperm", "bird is capable of cast shadow", "fruit is capable of cast shadow",\
            "flower is capable of branch into leaves", "plant is a numbing agent", "tree is part of angiosperm",\
            "flower is a plant", "plant is capable of cast shadow"]
contradictory_validities = ["mahogany has a nucleus", "lemon is a reproductive structure", "poppy has a vacuole",
            "daisy is capable of grow in garden", "carnation is a reproductive structure", "dodo has a cell",
            "falcon has a cell", "orchid is the opposite of animal", "chestnut is a important food source"]


def test_contains(testcase: dict, phrases: list[str]) -> bool:
    metadata = testcase["metadata"]
    property_phrase, _ = metadata2string(metadata["property"])
    entailed_phrase, _ = metadata2string(metadata["statement"])
    if "implicit_rule" in metadata.keys():
        entailing_phrase,_ = metadata2string(metadata["implicit_rule"])
    else:
        entailing_phrase = ""
    for phrase in phrases:
        if phrase in (property_phrase + entailing_phrase + entailed_phrase):
            return True
    return False


###########################################################################
def read_results(test_filename: str, model: str)-> list[list[str]]:
    # Read the pre-run test results file, throw out poorly worded questions
    # and return the remaineder as a list
    #
    # Each entry will be of the form ["x", "x", "x", "x", "x", "x"]
    # representing the truth and the model_answer
    # for base_property, entailing_phrase, and entailed_property

    # Read the test queries file
    tests = []
    with open(test_filename, 'r') as jsonfile:
        for line in jsonfile:
            tests.append(json.loads(line))
    jsonfile.close()

    # Read the test results file
    results_filename = test_filename.strip(".jsonl") + "_" + model.split("/")[-1] + ".txt"
    results_filename = "./Results/" + results_filename
    if not Path(results_filename).is_file():
        raise Exception("File Does Not Exist:", results_filename)
    results = []
    with open(results_filename, "r") as results_file:
        for row in results_file:
            results.append(row)
    results_file.close()
    results = results[: num_tests]

    results_list = []
    purged_tests = 0
    for index, (test, result) in enumerate(zip(tests, results, strict=True)):
        base_property_truth, base_property_model = result[0], result[2]
        entailing_phrase_truth, entailing_phrase_model = result[4], result[6]
        entailed_property_truth, entailed_property_model = result[8], result[10]

        string_a, string_b, string_c, string_d = result.split(".")
        base_property_truth, base_property_model = string_a[-1], string_b[0]
        entailing_phrase_truth, entailing_phrase_model = string_b[-1], string_c[0]
        entailed_property_truth, entailed_property_model = string_c[-1], string_d[0]
        # Sanity check.  The entailed_property_truth should always be "1" for this data set.
        if entailed_property_truth not in ["1", "0"]:
            raise Exception("Entailed Property Error:",base_property_truth, base_property_model,\
                entailing_phrase_truth, entailing_phrase_model, entailed_property_truth, entailed_property_model)

        # Throw out poorly worded, indeterminate, or incorrect tests
        if test_contains(test, improper_statements + contradictory_validities + incorrect_validities):
            #print("THROWING OUT THIS TEST TUPLE:\n",test,"\n")
            purged_tests += 1
            continue

        results_list.append([base_property_truth, base_property_model,\
                            entailing_phrase_truth, entailing_phrase_model,\
                            entailed_property_truth, entailed_property_model])
        # Print out the properties to check for poorly worded or inaccurate properties
        #property = test["metadata"]["property"]
        #print(index+1, metadata2string(property, enable_negation=True)[0], property)
        #statement = test["metadata"]["statement"]
        #print(index+1, metadata2string(statement, enable_negation=True)[0], statement)

    print("\nMODEL:",model)
    print(f"Threw out {purged_tests} poorly worded or contradictory tests.")
    return results_list

###########################################################################
def crunch_results(results_list: list[list[str]]) -> [float, float, float]:
    # 
    # Compare the model's answers to the truth.
    # Return the accuracy, consistency, and contrapositive_consistency.
    #
    # INPUT:
    #   results_list (list) each element of the form ["x", "x", "x", "x", "x", "x"]
    #                       representing truth and model_answer
    #                       for base_property, entailing_phrase, and entailed_property
    #
    # OUTPUT:
    #   accuracy (float)
    #   consistency (float)
    #   contrapositive_consistency (float)
    #

    # Set a threshold for a minimum number of valid tests for computing consistency
    min_threshold = 6

    total_facts, correct_facts = 0,0
    total_consistency_tests, matching_consistency_tests = 0,0
    total_contrapositive_tests, matching_contrapositive_tests = 0,0
    total_tests, matching_tests = 0,0
    tests_without_entailment = 0

    for index, result in enumerate(results_list):
        base_property_truth, base_property_model, entailing_phrase_truth,\
            entailing_phrase_model, entailed_property_truth, entailed_property_model = result

        # Check the overall accuracy of base_property, entailed_property, and entailing_phrase (if specified)
        #
        # Check base property
        total_facts += 1
        if base_property_model == base_property_truth:
            correct_facts += 1
        
        # Check entailed property
        total_facts += 1
        if entailed_property_model == entailed_property_truth:
            correct_facts += 1
        
        # Check entailing property
        # Skip cases where the entailing phrase is not specified
        if entailing_phrase_truth in ["0", "1"]: # In this data set, the entailing phrase is always true
            total_facts += 1
            if entailing_phrase_model == entailing_phrase_truth:
                correct_facts += 1
        else:
            print(index+1, "INACCURATE ENTAILING:",metadata2string(test["metadata"]["implicit_rule"]),\
                "truth=",entailing_phrase_truth,"model answer=",entailing_phrase_model)
        #
        # Only test for consistency where the entailing property is known to be true
        #
        if entailing_phrase_model != "1": # Entailing phrase is not known by the model to be true
            tests_without_entailment += 1
            continue

        # Check consistency of entailment
        if base_property_model == "1":
            total_consistency_tests +=1
            if entailed_property_model == "1":
                matching_consistency_tests += 1

        # Check contrapositive consistency
        if entailed_property_model == "0":
            total_contrapositive_tests +=1
            if base_property_model == "0":
                matching_contrapositive_tests += 1

        # Check the overall consistency between the base_property and the entailed_property
        total_tests += 1
        if base_property_model == entailed_property_model:
            matching_tests += 1

    accuracy = correct_facts/ total_facts if (total_facts >= min_threshold) else None
    consistency = matching_consistency_tests / total_consistency_tests if (total_consistency_tests >= min_threshold) else None
    contrapositive_consistency = matching_contrapositive_tests / total_contrapositive_tests if (total_contrapositive_tests >= min_threshold) else None

    print(f"Threw out {tests_without_entailment} tests where the entailment was unknown to the model.\n")
    print(f"Accuracy = {correct_facts} out of {total_facts}: {100 * accuracy:.1f}%\n")
    if consistency is not None:
        print("Consistency =", matching_consistency_tests, "out of", total_consistency_tests, ":",\
                round(100 * consistency, 1), "%\n")
    if contrapositive_consistency is not None:
        print("Contrapositive Consistency =", matching_contrapositive_tests, "out of", total_contrapositive_tests, ":",\
                round(100 * contrapositive_consistency, 1), "%\n")
    print(f"Bilateral Consistency = {matching_tests} out of {total_tests}: {100 * matching_tests / max(total_tests,1):.1f}%\n")

    return accuracy, consistency, contrapositive_consistency


##########################################################################

def models_in_filename(models: list[str], filename:str) -> bool:
    for model in models:
        if model in filename:
            return True
    return False

def model2filename(model: str, filename_list:[str]) -> bool:
    for filename in filename_list:
        if model in filename:
            return filename
    return None

##########################################################################

def entailing_known(result: list[str]) -> bool:
    # 
    # It's a valid result of consistency if the model knows the entailing phrase
    #
    # INPUT
    #   list(str)   base_truth, base_model, entailing_truth, entailing_model, entailed_truth, entailed_model
    #
    entailing_model = result[3]
    return entailing_model =="1"

def model_base_true(result: list[str]) -> bool:
    # 
    # Test consistency only when the model thinks the base_property is true
    #
    # INPUT
    #   list(str)   base_truth, base_model, entailing_truth, entailing_model, entailed_truth, entailed_model
    #
    base_property_model = result[1]
    return base_property_model == "1"

def model_entailed_false(result: list[str]) -> bool:
    # 
    # Test contrpositive consistency only when the model thinks the entailed_property is false
    #
    # INPUT
    #   list(str)   base_truth, base_model, entailing_truth, entailing_model, entailed_truth, entailed_model
    #
    entailed_property_model = result[5]
    return entailed_property_model == "0"

def calc_accuracy(result: list[str]) -> float:
    # 
    # Calculate accuracy of base_property and entailed property
    # Don't include entailing property in accuracy calculation because
    #   inaccurate entailments are thrown out of consistency calculation
    #
    # INPUT
    #   list(str)   base_truth, base_model, entailing_truth, entailing_model, entailed_truth, entailed_model
    #
    base_truth, base_model = result[0], result[1]
    entailed_truth, entailed_model = result[4], result[5]
    return (int(base_truth == base_model) + int(entailed_truth == entailed_model))/2

def calc_consistency(result: list[str]) -> int:
    # 
    # Calculate accuracy of base_property and entailed property
    # Don't include entailing property in accuracy calculation because
    #   inaccurate entailments are thrown out of consistency calculation
    #
    # INPUT
    #   list(str)   base_truth, base_model, entailing_truth, entailing_model, entailed_truth, entailed_model
    #
    base_truth, base_model = result[0], result[1]
    entailed_truth, entailed_model = result[4], result[5]
    return int(base_model == entailed_model)

def calc_correlation_acc_consist(results_list: list[list[str]]) -> float:
    #
    # Calculate the correlation between accuracy and consistency of the sampled results.
    #
    # INPUT:
    #  results_list (list)  Each entry will be of the form ["x", "x", "x", "x", "x", "x"]
    #                       representing the truth and the model_answer
    #                       for base_property, entailing_phrase, and entailed_property
    #
    # Valid test of consistency if entailing property is known and base_property is thought to be true
    accuracies =    [calc_accuracy(result) for result in results_list if (entailing_known(result) and model_base_true(result))]
    consistencies = [calc_consistency(result) for result in results_list if (entailing_known(result) and model_base_true(result))]
    correlation = np.corrcoef(accuracies, consistencies)[0][1]
    print(f"Correlation between accuracy and consistency is {correlation:.2f}")
    #
    # Valid test of contrapositive consistency if entailing property is known and base_property is thought to be true
    accuracies =    [calc_accuracy(result) for result in results_list if (entailing_known(result) and model_entailed_false(result))]
    consistencies = [calc_consistency(result) for result in results_list if (entailing_known(result) and model_entailed_false(result))]
    correlation = np.corrcoef(accuracies, consistencies)[0][1]
    print(f"Correlation between accuracy and contrapositive consistency is {correlation:.2f}")
    #
    # Valid test of bilateral consistency anytime entailing property is known
    accuracies =    [calc_accuracy(result) for result in results_list if entailing_known(result)]
    consistencies = [calc_consistency(result) for result in results_list if entailing_known(result)]
    correlation = np.corrcoef(accuracies, consistencies)[0][1]
    print(f"Correlation between accuracy and bilateral consistency is {correlation:.2f}")
    #
    return correlation


##########################################################################
# Calculate Accuracy, Consistency, Contrapositive Consistency, and Overall Consistency
# for every model with an results file in the Results folder.
#
# Make a scatter plot of Consistency Vs Accuracy.
# Can calculate a single value for each model, or divide each model's results into batches of "sample_size"
#
def scatter_plot_results(sample_size:int = 9999) -> None:
    #
    # INPUT:
    #   batch_size (int)    size of the batches for calculating accuracy and consistency (aka coherence) for each model
    #
    results_files = os.listdir("Results")
    results_files = [model2filename(model, results_files) for model in models_to_display if (model2filename(model, results_files) is not None)]

    plt.figure(1)
    plt.figure(figsize=(12,5))
    plt.figure(figsize=(12,8)) # Taller plot when we have to list consistency and contra-positive in the legend

    all_results_list = []
    for results_file in results_files:
        print(results_file)
        test_filename, model = results_file.split("_test_")
        test_filename = test_filename + "_test.jsonl"
        model = model.strip(".txt")
        model_text = model.split("-2024")[0]
        label_c, label_cc = model_text + " coherence", model_text + " contrapositive"
        results_list = read_results(test_filename, model)
        all_results_list = all_results_list + results_list
        results_list = results_list[: num_tests]

        # Use this for random samples with replacement.  When sample_size>=1289, this becomes a single sample per model
        if sample_size >= len(results_list):
            single_sample = True
            sample_size = len(results_list)
            num_samples = 1
        else:
            single_sample = False
            num_samples = 100
        for i in range(num_samples):
            sample_list = random.sample(results_list, sample_size)
            accuracy, consistency, contrapositive_consistency = crunch_results(sample_list)
            if consistency is not None:
                plt.plot(accuracy, consistency, "o", color = color_dict[model], markeredgecolor = "black" if ("nstruct" not in model) else None, label = label_c)
                label_c = None
            if (contrapositive_consistency is not None) and (single_sample==True):
                plt.plot(accuracy, contrapositive_consistency, "^", color = color_dict[model],\
                            markeredgecolor = "black" if ("nstruct" not in model) else None, label = label_cc)
                label_cc = None
            if single_sample:
                _ = calc_correlation_acc_consist(sample_list)
            print("\n\n")

        """
        ## Use this for sequential samples with no replacement
        #while len(results_list) > 0:
        #    sample_list = results_list[: sample_size]
        #    results_list = results_list[sample_size :]
        #    accuracy, consistency, contrapositive_consistency = crunch_results(sample_list)
        #    if consistency is not None:
        #        plt.plot(accuracy, consistency, "o", color = color_dict[model], markeredgecolor = "black" if ("nstruct" not in model) else None, label = label_c)
        #        label_c = None
        #    if (contrapositive_consistency is not None) and (sample_size >= 1289):
        #        plt.plot(accuracy, contrapositive_consistency, "^", color = color_dict[model],\
                            markeredgecolor = "black" if ("nstruct" not in model) else None, label = label_cc)
        #        label_cc = None
        #    if (sample_size >= 1289):
        #        _ = calc_correlation_acc_consist(sample_list)
        #    print("\n\n")
        """

    print("OVERALL CORRELATIONS FOR ALL MODELS")
    _ = calc_correlation_acc_consist(all_results_list)
    #
    if single_sample:
        # Use this to plot a single dot of consistency and contrapositive for each model
        plt.xlabel("Accuracy")
        plt.ylabel("Coherence/Contrapositive Coherence")
        plt.title("Coherence and Contrapositive Coherence Vs Accuracy of GPT, Mistral, and Claude LLMs")
        plt.legend(loc="upper left")
        plt.xlim(0.5,1)
        plt.ylim(0.7,1)
        plt.savefig("MARS_CoherenceAndContrapositiveCoherenceVsAccuracy_20240521.png")
    else:
        # Use this for the scatter plots of the batches (coherence vs accuracy only, not contra-positive coherence)
        plt.xlabel("Accuracy")
        plt.ylabel("Coherence")
        plt.legend(loc="lower right")
        plt.xlim(0.6,1)
        plt.ylim(0.2,1)
        plt.title("Coherence Vs Accuracy of LLMs (Batches of " + str(sample_size) + " Samples)")
        plt.savefig("MARS_CoherenceVsAccuracy_"+ str(sample_size) +"samples_20240502.png")
    return

##########################################################################
#
# Make a distribution plot of Accuracy.
#
def distribution_plot_accuracy_results(sample_size:int = 100, num_samples:int = 100) -> None:
    #
    # INPUT:
    #   batch_size (int)    size of the batches for calculating accuracy and consistency for each model
    #
    results_files = os.listdir("Results")
    results_files = [model2filename(model, results_files) for model in models_to_display if (model2filename(model, results_files) is not None)]

    plt.figure(2)
    plt.figure(figsize=(10,8))

    # Accuracy Distribution plot of each model
    for results_file in results_files:
        test_filename, model = results_file.split("_test_")
        test_filename = test_filename + "_test.jsonl"
        model = model.strip(".txt")
        model_text = model.split("-2024")[0]
        results_list = read_results(test_filename, model)
        results_list = results_list[: num_tests]
        accuracy_list, consistency_list = [], []
        for i in range(num_samples):
            sample_list = random.sample(results_list, sample_size)
            accuracy, consistency, contrapositive_consistency = crunch_results(sample_list)
            accuracy_list.append(accuracy)
        # Best fit curve
        sns.kdeplot(accuracy_list, color = color_dict[model], label = model, common_norm = True,\
                linestyle="dashed" if ("Instruct" in model) else "solid", alpha=0.5, bw_adjust=3, fill=True)
    
    # Accuracy Distribution plot of random baseline
    model = "random baseline"
    random_baseline = [[random.choice(["0", "1"]) for i in range(6)] for typle in range(len(results_list))][: num_tests]
    accuracy_list, consistency_list = [], []
    for i in range(num_samples):
        sample_list = random.sample(random_baseline, sample_size)
        accuracy, consistency, contrapositive_consistency = crunch_results(sample_list)
        accuracy_list.append(accuracy)
    # Best fit curve
    sns.kdeplot(accuracy_list, color = color_dict[model], label = model, common_norm = True,\
            linestyle="dashed", alpha=0.5, bw_adjust=3, fill=True)

    # Override my plot settings to match collaborators
    plt.xlim(0,1)
    plt.title('Accuracy Score Distribution', fontsize=18, fontweight='bold', color='#333333')
    plt.xlabel('Model Score', fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel('Density', fontsize=14, fontweight='bold', labelpad=10)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    #plt.yticks([])
    plt.legend(loc= "upper left", fontsize=12, frameon=False)
    plt.grid(True)
    plt.savefig("MARS_Accuracy_Distribution_20240521.png")
    #
    return

##########################################################################
#
# Make a distribution plot of Consistency.
#
def distribution_plot_consistency_results(sample_size:int = 100, num_samples:int = 100) -> None:
    #
    # INPUT:
    #   batch_size (int)    size of the batches for calculating accuracy and consistency for each model
    #
    results_files = os.listdir("Results")
    results_files = [model2filename(model, results_files) for model in models_to_display if (model2filename(model, results_files) is not None)]

    plt.figure(3)
    plt.figure(figsize=(10,8))
    #
    # Consistency distribution plot of each model
    for results_file in results_files:
        test_filename, model = results_file.split("_test_")
        test_filename = test_filename + "_test.jsonl"
        model = model.strip(".txt")
        model_text = model.split("-2024")[0]
        label_c, label_cc = model_text + " consistency", model_text + " contrapositive"
        results_list = read_results(test_filename, model)
        results_list = results_list[: num_tests]
        
        accuracy_list, consistency_list = [], []
        for i in range(num_samples):
            sample_list = random.sample(results_list, sample_size)
            accuracy, consistency, contrapositive_consistency = crunch_results(sample_list)
            consistency_list.append(consistency)
        # Best fit curve
        sns.kdeplot(consistency_list, color = color_dict[model], label = model, common_norm = True,\
                linestyle="dashed" if ("Instruct" in model) else "solid", alpha=0.5, bw_adjust=3, fill=True)

    # Consistency Distribution plot of random baseline
    model = "random baseline"
    random_baseline = [[random.choice(["0", "1"]) for i in range(6)] for tuple in range(len(results_list))]
    accuracy_list, consistency_list = [], []
    for i in range(num_samples):
        sample_list = random.sample(random_baseline, sample_size)
        accuracy, consistency, contrapositive_consistency = crunch_results(sample_list)
        consistency_list.append(consistency)
    # Best fit curve
    sns.kdeplot(consistency_list, color = color_dict[model], label = model, common_norm = True,\
            linestyle="dashed", alpha=0.5, bw_adjust=3, fill=True)
    # Use this to make room for the legend if there are many kde plots without histogram bars
    plt.xlim(0.3,1) if (len(models_to_display) > 3) else plt.xlim(0.5,1)
    #
    # Override my plot settings to match collaborators
    plt.xlim(0,1) 
    plt.title('Logical Coherence of Beliefs', fontsize=18, fontweight='bold', color='#333333')
    plt.xlabel('Leap-of-Thought Score', fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel('Distribution', fontsize=14, fontweight='bold', labelpad=10)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    #plt.yticks([])
    plt.legend(fontsize=12, frameon=False, loc="upper left")
    plt.grid(True)
    plt.savefig("MARS_Consistency_Distribution_20240521.png")
    #
    return

##########################################################################
test_filename = "data_hypernyms_hypernyms_explicit_only_short_neg_hypernym_rule_test.jsonl"

anthropic_models = ["claude-instant-1.2", "claude-2.0", "claude-2.1", "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
openai_models = ["davinci-002", "gpt-3.5-turbo", "gpt-4"]
openai_models = ["gpt-3.5-turbo", "gpt-4"]
mistral_models = ["open-mistral-7b", "open-mixtral-8x7b", "open-mixtral-8x22b"]
together_mistral_models = ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mixtral-8x22B-Instruct-v0.1"]
# Filenames for the together_mistral modes don't have the preceding "mistralai/"
together_mistral_models_short = [model.split("/")[-1] for model in together_mistral_models]

# claude-instant-1.2 was "yellow" but it didn't show up in the KDE plots without histogram bars
color_dict = {"claude-instant-1.2": "palegoldenrod", "claude-2.0": "gold", "claude-2.1": "darkorange", "claude-3-haiku-20240307": "lightsalmon",\
            "claude-3-sonnet-20240229": "red", "claude-3-opus-20240229": "darkred", "gpt-3.5-turbo": "dodgerblue", "gpt-4": "blue",\
            "open-mistral-7b": "greenyellow",  "Mistral-7B-Instruct-v0.2": "lightgreen", "open-mixtral-8x7b": "limegreen",\
            "Mixtral-8x7B-Instruct-v0.1": "mediumseagreen", "open-mixtral-8x22b": "seagreen", "Mixtral-8x22B-Instruct-v0.1": "darkgreen",\
            "random baseline": "lightgrey"
            }

#
# Control the plots with these values
#
num_tests = 9999  # Normally keep this larger than test set size (1289), set it lower when debugging
sample_size = 9999  # Set this large (>=1289) to get a single overall calculation for each model
models_to_display =  anthropic_models + mistral_models + together_mistral_models_short + openai_models

if __name__ == "__main__":
    # Crunch the pre-run results
    scatter_plot_results(sample_size)
    distribution_plot_accuracy_results(sample_size=100, num_samples=100)
    distribution_plot_consistency_results(sample_size=100, num_samples=100)



