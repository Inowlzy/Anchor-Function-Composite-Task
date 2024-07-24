import numpy as np


def one_hot_encode(sequence, num_classes):
    one_hot_encoded = np.zeros((len(sequence), num_classes))
    for idx, val in enumerate(sequence):
        one_hot_encoded[idx, val - 1] = 1
    return one_hot_encoded


def f_a(x, a):
    if a == 1:
        y = x + 5
    elif a == 2:
        y = x + 1
    elif a == 3:
        y = x - 2
    elif a == 4:
        y = x - 5
    return y


def f_a12(x, a1, a2):
    y_temp = f_a(x,a1)
    y = f_a(y_temp,a2)
    return y


def seq_gen_train(num_samples):
    a1_list = [1, 2, 3, 4]
    a2_list = [1, 2, 3, 4]
    task_combinations = [(3, 4)]
    data_combinations = [(a1, a2) for a1 in a1_list for a2 in a2_list if (a1, a2) not in task_combinations]

    all_train_sequences = []
    seeds = list(range(num_samples))

    for n in range(num_samples):
        seed = seeds[n]
        np.random.seed(seed)
        seq_input = np.random.choice([0] + list(range(5, 100)), size=9)
        an = np.random.choice(len(data_combinations))
        a1, a2 = data_combinations[an]
        i = np.random.choice(range(1, 7))
        while np.mod(seq_input[i-1], 8) == i:
            seq_input[i-1] = np.random.choice([0] + list(range(5, 100)))
        seq_input[i] = a1
        seq_input[i + 1] = a2

        input_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        input_mask[i - 1] = 1
        input_mask[i] = 1
        input_mask[i + 1] = 1

        seq_output = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        seq_output[i] = f_a(seq_input[i - 1], seq_input[i])
        seq_output[i + 1] = f_a12(seq_input[i - 1], seq_input[i], seq_input[i + 1])
        temp = np.random.choice([0] + list(range(5, 100)))
        for ii in range(i):
            seq_output[ii] = temp
        for ii in range(i + 2, 9):
            seq_output[ii] = seq_output[i + 1]
        num_classes = 117
        one_hot_encoded_output_sequence = one_hot_encode(seq_output, num_classes)

        output_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        output_mask[i+1] = 1

        all_train_sequences.append([seq_input, one_hot_encoded_output_sequence, output_mask, input_mask])

    return all_train_sequences


def seq_gen_test_data(num_samples):
    a1_list = [1, 2, 3, 4]
    a2_list = [1, 2, 3, 4]
    task_combinations = np.array([(3, 4)])
    data_combinations = [(a1, a2) for a1 in a1_list for a2 in a2_list if (a1, a2) not in task_combinations]

    all_test_data_sequences = []
    seeds = list(range(num_samples))

    for n in range(num_samples):
        seed = seeds[n]
        np.random.seed(seed)
        seq_input = np.random.choice([0] + list(range(5, 100)), size=9)
        an = np.random.choice(len(data_combinations))
        a1, a2 = data_combinations[an]
        i = np.random.choice(range(1, 7))
        while np.mod(seq_input[i-1], 8) != i:
            seq_input[i-1] = (seq_input[i-1] // 8) * 8 + i
        seq_input[i] = a1
        seq_input[i + 1] = a2

        input_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        input_mask[i - 1] = 1
        input_mask[i] = 1
        input_mask[i + 1] = 1

        seq_output = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        seq_output[i] = f_a(seq_input[i - 1], seq_input[i])
        seq_output[i + 1] = f_a12(seq_input[i - 1], seq_input[i], seq_input[i + 1])
        temp = np.random.choice([0] + list(range(5, 100)))
        for ii in range(i):
            seq_output[ii] = temp
        for ii in range(i + 2, 9):
            seq_output[ii] = seq_output[i + 1]
        num_classes = 117
        one_hot_encoded_output_sequence = one_hot_encode(seq_output, num_classes)

        output_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        output_mask[i+1] = 1

        all_test_data_sequences.append([seq_input, one_hot_encoded_output_sequence, output_mask, input_mask])

    return all_test_data_sequences


def seq_gen_test_task(num_samples):
    task_combinations = np.array([(3, 4)])

    all_test_task_sequences = []
    seeds = list(range(num_samples))

    for n in range(num_samples):
        seed = seeds[n]
        np.random.seed(seed)
        seq_input = np.random.choice([0] + list(range(5, 100)), size=9)
        an = np.random.choice(len(task_combinations))
        a1, a2 = task_combinations[an]
        i = np.random.choice(range(1, 7))
        while np.mod(seq_input[i-1], 8) != i:
            seq_input[i-1] = (seq_input[i-1] // 8) * 8 + i
        seq_input[i] = a1
        seq_input[i + 1] = a2

        input_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        input_mask[i - 1] = 1
        input_mask[i] = 1
        input_mask[i + 1] = 1

        seq_output = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        seq_output[i] = f_a(seq_input[i - 1], seq_input[i])
        seq_output[i + 1] = f_a12(seq_input[i - 1], seq_input[i], seq_input[i + 1])
        temp = np.random.choice([0] + list(range(5, 100)))
        for ii in range(i):
            seq_output[ii] = temp
        for ii in range(i + 2, 9):
            seq_output[ii] = seq_output[i + 1]
        num_classes = 117
        one_hot_encoded_output_sequence = one_hot_encode(seq_output, num_classes)

        output_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        output_mask[i+1] = 1

        all_test_task_sequences.append([seq_input, one_hot_encoded_output_sequence, output_mask, input_mask])

    return all_test_task_sequences
