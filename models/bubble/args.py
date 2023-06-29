from argparse import ArgumentParser, Namespace


def bubble_model_args(training: bool) -> Namespace:
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
        parser.add_argument(
            "-lr",
            "--learning-rate",
            default=1e-5,
            type=float,
            help="The learning rate provided to the ADAM optimizer.",
        )
        parser.add_argument(
            "--pushforward",
            action="store_true",
            help=(
                "Whether to apply pushforward training with point-to-surface"
                " loss."
            )
            # default=0,
            # type=int,
            # help=(
            #     "The number of pushforward steps with point-to-surface loss to"
            #     " apply during training."
            # ),
        )
        parser.add_argument(
            "-k",
            "--tbptt",
            default=None,
            type=int,
            help=(
                "[Experimental] The number of iterations between each"
                " backpropagation step when using tbptt training (k - 1 steps"
                " are then trained on predicted data)."
            ),
        )
    else:
        parser.add_argument(
            "output_directory",
            help="Name of the output directory to store the STL files.",
        )
        parser.add_argument(
            "-n", "--steps", type=int, help="The number of steps to predict."
        )
        parser.add_argument(
            "-id",
            "--bubble-id",
            default=0,
            type=int,
            help="Index of the initial bubble object to load.",
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
        default=7,
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
    parser.add_argument(
        "--remesh-velocity",
        action="store_true",
        help="Whether to remesh velocity directly.",
    )
    parser.add_argument(
        "--target-acceleration",
        action="store_true",
        help="Whether to use acceleration (true) or velocity (false) targets.",
    )

    return parser.parse_args()
