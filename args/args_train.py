"""

JoinABLe training args

"""

from args import args_common


def get_parser():
    """Return the training args parser"""
    parser = args_common.get_parser()
    parser.add_argument(
        "--pre_net",
        type=str,
        default="mlp",
        choices=("mlp", "cnn"),
        help="Type of network to use in the pre-net."
    )
    parser.add_argument(
        "--max_nodes_per_batch",
        type=int,
        default=0,
        help="Max nodes in a 'dynamic' batch while training. Set to 0 to disable and use a fixed batch size."
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=384,
        help="Number of hidden units."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate."
    )
    parser.add_argument(
        "--mpn",
        type=str,
        choices=("gat", "gatv2"),
        default="gatv2",
        help="Message passing network to use."
    )
    parser.add_argument(
        "--mpn_layer_num",
        type=int,
        default="2",
        help="The number of the layers of the MPN."
    )
    parser.add_argument(
        "--post_net",
        type=str,
        choices=("mm", "mlp"),
        default="mlp",
        help="Post network method."
    )
    parser.add_argument(
        "--max_node_count",
        type=int,
        default=1024,
        help="Restrict training data to graph pairs with under this number of nodes.\
              Set to 0 to train on all data."
    )
    parser.add_argument(
        "--train_label_scheme",
        type=str,
        default="Joint",
        help="Labels to use for training as a string separated by commas.\
              Can include: Joint, Ambiguous, JointEquivalent, AmbiguousEquivalent, Hole, HoleEquivalent\
              Note: 'Ambiguous' are referred to as 'Sibling' labels in the paper."
    )
    parser.add_argument(
        "--test_label_scheme",
        type=str,
        default="Joint,JointEquivalent",
        help="Labels to use for testing as a string separated by commas.\
              Can include: Joint, Ambiguous, JointEquivalent, AmbiguousEquivalent, Hole, HoleEquivalent\
              Note: 'Ambiguous' are referred to as 'Sibling' labels in the paper."
    )
    parser.add_argument(
        "--hole_scheme",
        type=str,
        default="both",
        choices=("holes", "no_holes", "both"),
        help="Evaluate with or wthout joints whose geometry contains holes."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.6E-06,
        help="Threshold to use for accuracy and IoU calculation."
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=("bce", "mle", "focal", "symmetric"),
        default="mle",
        help="Loss to use."
    )
    parser.add_argument(
        "--reduction",
        type=str,
        choices=("sum", "mean"),
        default="mean",
        help="Loss reduction to use."
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=200.0,
        help="Positive class weight for BCE loss."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=5.0,
        help="Gamma parameter in focal loss to down-weight easy examples."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Alpha parameter in focal loss is the weight assigned to rare classes."
    )
    parser.add_argument(
        "--input_features",
        type=str,
        default="entity_types,length,area",
        help="Input features to use as a string separated by commas.\
                Can include: points, normals, tangents, trimming_mask,\
                axis_pos, axis_dir, bounding_box, entity_types\
                area, circumference, param_1, param_2\
                length, radius, start_point, middle_point, end_point"
    )
    parser.add_argument(
        "--without_synthetic",
        action="store_true",
        default=False,
        help="Skip synthetic joints."
    )
    parser.add_argument(
        "--feature_embedding",
        action="store_true",
        default=False,
        help="Using feature quantization and embedding."
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=9,
        help="Number of bit of quantization."
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor."
    )
    parser.add_argument(
        "--type_head",
        action="store_true",
        default=False,
        help="Using type head."
    )
    return parser


def get_args():
    """Get the args used for training"""
    parser = get_parser()
    return args_common.get_args(parser)
