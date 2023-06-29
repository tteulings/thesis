from argparse import ArgumentParser, Namespace


def flag_model_args(training: bool) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "data_directory", help="Name of the data directory to load."
    )

    if training:
        parser.add_argument(
            "-e",
            "--epochs",
            default=100,
            type=int,
            help="The number of epochs to train the model.",
        )
        parser.add_argument(
            "-p",
            "--print-every",
            default=1000,
            type=int,
            help="The number of epochs between every print of the loss.",
        )
        parser.add_argument(
            "--noise",
            default=None,
            type=float,
            help=(
                "The standard deviation of the normal distribution that is used"
                " to multiplicatively add relative noise to the input data."
            ),
        )
    else:
        parser.add_argument(
            "-n",
            "--num-rollouts",
            default=10,
            type=int,
            help=(
                "The number of rollouts to be performed before averaging"
                " results."
            ),
        )
        parser.add_argument(
            "output_directory",
            help="Name of the output directory to store stl files.",
        )

    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="The name of the checkpoint file to use.",
    )
    parser.add_argument(
        "-it",
        "--iterations",
        default=15,
        type=int,
        help="The number of message passing steps to perform.",
    )
    parser.add_argument(
        "-ls",
        "--latent-size",
        default=128,
        type=int,
        help="The latent size used throughout the model.",
    )
    parser.add_argument(
        "-nl",
        "--num-layers",
        default=2,
        type=int,
        help="The number of hidden layers to use in the internal mlps.",
    )

    return parser.parse_args()
