import typing

import numpy as np
import torch


def _generate_config(
    max_layers: int,
    max_width: int,
    min_width: int,
    random_state,
    selu_vs_relu_rate: float = 0.7,
    constant: int = 1,
):
    # How many layers and should their size drop
    layers = random_state.randint(2, max_layers)
    shrinking = _deeper_higher_chance(layers, max_layers, constant, random_state)

    # Width of initial layer and width of other layers
    initial_width = random_state.randint(min_width, max_width)
    widths = _get_widths(
        shrinking, layers, max_layers, initial_width, min_width, constant, random_state
    )

    # Architecture choices
    activation = _selu_vs_relu(selu_vs_relu_rate, random_state)[0]
    batchnorm = _deeper_higher_chance(layers, max_layers, constant, random_state)[0]
    dropout = _deeper_higher_chance(layers, max_layers, constant, random_state)[0]

    # Dropout found to harm generalization, turned off
    if dropout:
        where_to_dropout = _deeper_smaller_chance(
            layers, max_layers, constant, random_state, size=layers
        )
        dropout_rates = where_to_dropout * _how_much_dropout(layers, random_state)
    else:
        dropout_rates = [0 for _ in range(layers)]
    return {
        "name": f"Layers={widths},{_describe_dropout(dropout,dropout_rates)}{_describe_batchnorm(batchnorm)}Activation={_describe_activation(activation)}",
        "layers": widths,
        "activation": activation,
        "batch_norm": batchnorm,
        "dropouts": dropout_rates,
    }


#######################################################################################
#
#                                   GENERAL UTILS
#
#######################################################################################


def _deeper_smaller_chance(
    layers: int, max_layers: int, constant: int, random_state, size: int = 1
) -> bool:
    probability: float = (max_layers + constant - layers) / max_layers
    return random_state.choice([True, False], size, p=[probability, 1 - probability])


def _deeper_higher_chance(
    layers: int, max_layers: int, constant: int, random_state, size: int = 1
) -> bool:
    probability: float = (max_layers + constant - layers) / max_layers
    return random_state.choice([False, True], size, p=[probability, 1 - probability])


#######################################################################################
#
#                                      DROPOUT
#
#######################################################################################


def _how_much_dropout(layers, random_state):
    return random_state.choice([0.1, 0.2], size=layers)


#######################################################################################
#
#                                    ACTIVATION
#
#######################################################################################


# Only one decision about activations
def _selu_vs_relu(selu_probability, random_state):
    return random_state.choice(
        [torch.nn.SELU(), torch.nn.ReLU()],
        1,
        p=[selu_probability, 1 - selu_probability],
    )


#######################################################################################
#
#                               ARCHITECTURE DECISIONS
#
#######################################################################################


# divide by 2 each n layers, where n is randomized
def _get_widths(
    shrinking,
    layers: int,
    max_layers: int,
    initial_width: int,
    min_width: int,
    constant: int,
    random_state,
) -> typing.List:
    def division_rate():
        return random_state.choice([0.5, 0.6, 0.7, 0.8])

    def _get_sizes(
        initial_width: int, minimal_width: int, divider, division_rate: float
    ):
        current_width = initial_width
        for divide in divider:
            if divide:
                new_width = int(current_width * division_rate)
                if new_width > minimal_width:
                    current_width = new_width
            yield current_width

    if shrinking:
        where_to_divide = _deeper_higher_chance(
            layers, max_layers, constant, random_state, size=layers
        )
        return list(
            _get_sizes(initial_width, min_width, where_to_divide, division_rate())
        )
    return [initial_width for _ in range(layers)]


#######################################################################################
#
#                           DESCRIBE ARCHITECTURE ELEMENT
#
#######################################################################################


def _describe_batchnorm(exists: bool) -> str:
    return "Batchnorm," if exists else ""


def _describe_dropout(exists: bool, rates) -> str:
    return f"Dropout:{rates}," if exists else ""


def _describe_activation(activation) -> str:
    return "ReLU" if isinstance(activation, torch.nn.ReLU) else "SELU"
