import keras
import argparse
import json
import pandas as pd
from termcolor import colored
from tqdm import tqdm

from model import MODEL_TYPES
from keras.callbacks import History
from datetime import datetime
import pickle
import json
import os

def append_or_create_txt(filename, content_to_append):

    file_exists = os.path.exists(filename)

    with open(filename, 'a') as file:  # Open in 'append' mode
        if not file_exists:
            # File was just created, no need to add extra newline
            file.write(content_to_append)
        else:
            # File exists, add a newline before appending
            file.write("\n" + content_to_append)




def save_history(history: History, filename: str):
    try:
        with open(filename, 'w') as f:
            json.dump(history.history, f)
    except:
        with open(filename, 'w') as f:
            json.dump(history, f)


def save_dict(data, filename):

    if filename.endswith('.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    elif filename.endswith('.json'):
        with open(filename, 'w') as file:
            json.dump(data, file)
    else:
        raise ValueError("Filename must end in .pkl or .json")

is_image = lambda x: x in ['mnist', 'cifar10', 'cifar100']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=MODEL_TYPES.keys())
    parser.add_argument('--operator', type=str, choices=['depth', 'backbone_activation', 'recurrent_activation',
                                                         'bidirectional', 'kernel_initializer','recurrent_initializer',
                                                         'backwards', 'stateful',
                                                         'num_units', 'recurrent_dropout',
                                                         'kernel_regularizer',
                                                         'masking',
                                                         'recurrent_regularizer',
                                                         'bias_regularizer',
                                                         'activity_regularizer',
                                                         ])
    parser.add_argument('--depth_center', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--logs_dir', type=str, default='models_logs')
    args = parser.parse_args()
    model_histories = {}
    model_prefix = f'{args.model}_{args.operator}'
    # default model
    if args.operator == 'depth':
        for _d in tqdm(range(args.depth_center-3, args.depth_center+4)):
            model_name = f"{model_prefix}_{_d}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model, history = MODEL_TYPES[args.model](depth=_d)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history
    if args.operator == 'backbone_activation':
        from operators.commons import ACTIVATIONS
        for _act in tqdm(ACTIVATIONS):
            model_name = f"{model_prefix}_{_act}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model, history = MODEL_TYPES[args.model](backbone_activation=_act)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history_{datetime.now()}.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history
    if args.operator == 'recurrent_activation':
        if args.model in ['transformer', 'rnn']:
            raise NotImplementedError('Recurrent Activation is only supported for lstm and gru models.')
        from operators.commons import ACTIVATIONS
        for _act in tqdm(ACTIVATIONS):
            model_name = f"{model_prefix}_{_act}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model, history = MODEL_TYPES[args.model](recurrent_activation=_act)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history_{datetime.now()}.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history
    if args.operator == 'bidirectional':
        if args.model == 'transformer':
            raise NotImplementedError('Recurrent Activation is not supported for transformer model.')
        for _direction in [True, False]:
            model_name = f"{model_prefix}_{_direction}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = bidirectional_op(MODEL_TYPES[args.model], _direction)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history
    if args.operator == 'kernel_initializer':
        from operators.commons import INITIALIZERS
        for _initializer in tqdm(INITIALIZERS):
            model_name = f"{model_prefix}_{_initializer}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = initializer_op(MODEL_TYPES[args.model], _initializer)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history
    if args.operator == 'recurrent_initializer':
        if args.model == 'transformer':
            raise NotImplementedError('Recurrent Activation is not supported for transformer model.')
        from operators.commons import INITIALIZERS
        for _initializer in INITIALIZERS:
            model_name = f"{model_prefix}_{_initializer}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = recurrent_initializer_op(MODEL_TYPES[args.model], _initializer)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history
    if args.operator == 'backwards':
        if args.model == 'transformer':
            raise NotImplementedError('Backwards is not supported for transformer model.')
        for _b in [True, False]:
            model_name = f"{model_prefix}_{_b}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = backwards_op(MODEL_TYPES[args.model], _b)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history
    if args.operator == 'stateful':
        if args.model == 'transformer':
            raise NotImplementedError('Stateful Operation is not supported for transformer model.')
        for _b in [True, False]:
            model_name = f"{model_prefix}_{_b}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = stateful_op(MODEL_TYPES[args.model], _b)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history
    if args.operator == 'num_units':
        if args.model == 'transformer':
            raise NotImplementedError('`num_units` is not supported for transformer model.')
        for _b in [64, 128, 192, 256]:
            model_name = f"{model_prefix}_{_b}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = num_units_op(MODEL_TYPES[args.model], _b)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history
    if args.operator == 'recurrent_dropout':
        from operators.commons import DROPOUTS
        if args.model == 'transformer':
            raise NotImplementedError('`recurrent_dropout` is not supported for transformer model.')
        for _b in tqdm(DROPOUTS):
            model_name = f"{model_prefix}_{_b}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = recurrent_dropout_op(MODEL_TYPES[args.model], _b)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history

    if args.operator == 'recurrent_dropout':
        from operators.commons import DROPOUTS
        if args.model == 'transformer':
            raise NotImplementedError('`recurrent_dropout` is not supported for transformer model.')
        for _b in tqdm(DROPOUTS):
            model_name = f"{model_prefix}_{_b}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = recurrent_dropout_op(MODEL_TYPES[args.model], _b)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history

    if args.operator == 'masking':
        if args.model == 'transformer':
            raise NotImplementedError('Masking is not supported for transformer model.')
        for _mask in [True, False]:
            model_name = f"{model_prefix}_{_mask}"
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = masking_op(MODEL_TYPES[args.model], _mask)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history

    if args.operator == 'kernel_regularizer':
        from operators.commons import REGULARIZERS

        for reg_class, reg_params in tqdm(REGULARIZERS):
            if reg_class:
                reg_instance = reg_class(**reg_params) if isinstance(reg_params, dict) else reg_class(reg_params)
                reg_name = reg_class.__name__  # Extract class name for clarity (e.g., "L1", "L2", "L1L2")
                reg_str = f"{reg_name}_{reg_params}" if isinstance(reg_params, float) else f"{reg_name}_l1{reg_params['l1']}_l2{reg_params['l2']}"
            else:
                reg_instance = None
                reg_str = "none"  # For the no-regularization case

            model_name = f"{model_prefix}_{reg_str}"  # Create model name
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = kernel_regularizer_op(MODEL_TYPES[args.model], reg_instance)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history

    if args.operator == 'recurrent_regularizer':
        from operators.commons import REGULARIZERS
        for reg_class, reg_params in tqdm(REGULARIZERS):
            if reg_class:
                reg_instance = reg_class(**reg_params) if isinstance(reg_params, dict) else reg_class(reg_params)
                reg_name = reg_class.__name__  # Extract class name for clarity (e.g., "L1", "L2", "L1L2")
                reg_str = f"{reg_name}_{reg_params}" if isinstance(reg_params, float) else f"{reg_name}_l1{reg_params['l1']}_l2{reg_params['l2']}"
            else:
                reg_instance = None
                reg_str = "none"  # For the no-regularization case

            model_name = f"{model_prefix}_{reg_str}"  # Create model name
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = recurrent_regularizer_op(MODEL_TYPES[args.model], reg_instance)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history

    if args.operator == 'bias_regularizer':
        from operators.commons import REGULARIZERS
        for reg_class, reg_params in tqdm(REGULARIZERS):
            if reg_class:
                reg_instance = reg_class(**reg_params) if isinstance(reg_params, dict) else reg_class(reg_params)
                reg_name = reg_class.__name__  # Extract class name for clarity (e.g., "L1", "L2", "L1L2")
                reg_str = f"{reg_name}_{reg_params}" if isinstance(reg_params, float) else f"{reg_name}_l1{reg_params['l1']}_l2{reg_params['l2']}"
            else:
                reg_instance = None
                reg_str = "none"  # For the no-regularization case

            model_name = f"{model_prefix}_{reg_str}"  # Create model name
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = bias_regularizer_op(MODEL_TYPES[args.model], reg_instance)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history
    if args.operator == 'activity_regularizer':
        from operators.commons import REGULARIZERS
        for reg_class, reg_params in tqdm(REGULARIZERS):
            if reg_class:
                reg_instance = reg_class(**reg_params) if isinstance(reg_params, dict) else reg_class(reg_params)
                reg_name = reg_class.__name__  # Extract class name for clarity (e.g., "L1", "L2", "L1L2")
                reg_str = f"{reg_name}_{reg_params}" if isinstance(reg_params, float) else f"{reg_name}_l1{reg_params['l1']}_l2{reg_params['l2']}"
            else:
                reg_instance = None
                reg_str = "none"  # For the no-regularization case

            model_name = f"{model_prefix}_{reg_str}"  # Create model name
            print(f"CURRENT MODEL: {colored(model_name.upper(), 'red', attrs=['bold'])}")
            print(f"TRAINING MODEL {model_name}")
            model_fn = activity_regularizer_op(MODEL_TYPES[args.model], reg_instance)
            model = model_fn(num_classes)
            history, model = train_model(model, x_train, y_train, x_val, y_val, args.num_epochs, metrics=[], loss=loss)
            print(f"MODEL {model_name} TRAINED")
            save_history(history, f"{args.logs_dir}/{model_name}_history.json")
            print(f"SAVING MODEL {model_name}")
            model.save(f"{args.logs_dir}/{model_name}.keras")
            model_histories[model_name] = history.history
    now = datetime.now()
    print(f"SAVING MODEL HISTORIES")
    file_name = f'{args.logs_dir}/{model_prefix}_records_{now.hour}:{now.minute}.pkl'
    save_dict(model_histories, file_name)
    append_or_create_txt(f'{args.logs_dir}/all_records.txt', file_name)

if __name__ == '__main__':
    main()
