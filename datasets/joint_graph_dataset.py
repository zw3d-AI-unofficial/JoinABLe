import json
import copy
import time
import math
import random
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import networkx as nx
from networkx.readwrite import json_graph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import plot_util
from joint import joint_axis
from datasets.joint_base_dataset import JointBaseDataset

class JointGraphDataset(JointBaseDataset):

    # The map of the entity types
    surface_type_map = {
        "PlaneSurfaceType": 0,
        "CylinderSurfaceType": 1,
        "ConeSurfaceType": 2,
        "SphereSurfaceType": 3,
        "TorusSurfaceType": 4,
        "NurbsSurfaceType": 5,
    }
    curve_type_map = {
        "Line3DCurveType": 0,
        "Arc3DCurveType": 1,
        "Circle3DCurveType": 2,
        "NurbsCurve3DCurveType": 3
    }
    entity_type_reverse_map = {
        0: "PlaneSurfaceType",
        1: "CylinderSurfaceType",
        2: "ConeSurfaceType",
        3: "SphereSurfaceType",
        4: "TorusSurfaceType",
        5: "NurbsSurfaceType",
        6: "Line3DCurveType",
        7: "Arc3DCurveType",
        8: "Circle3DCurveType",
        9: "NurbsCurve3DCurveType"
    }
    # The different types of labels for B-Rep entities
    label_map = {
        "Non-joint": 0,
        "Joint": 1,
        "Ambiguous": 2,  # aka Sibling
        "JointEquivalent": 3,
        "AmbiguousEquivalent": 4,
        "Hole": 5,
        "HoleEquivalent": 6,
    }
    # 10 x 10 grid
    grid_size = 10
    # Grid 1D length
    grid_len = grid_size * grid_size
    # pt.x, pt.y, pt.z, normal.x, normal.y, normal.z, mask
    grid_channels = 7
    # Total tensor size
    grid_total = grid_len * grid_channels

    # Face input features and their size
    face_grid_feature_map = {
        "points": grid_len * 3,
        "normals": grid_len * 3,
        "trimming_mask": grid_len * 1,
    }
    face_entity_feature_map = {
        "area": 1,
        "axis_dir": 3,
        "axis_pos": 3,
        "bounding_box": 6,
        "circumference": 1,
        "entity_types": len(surface_type_map),
        "param_1": 1,
        "param_2": 1
    }

    # Edge input features and their size
    edge_grid_feature_map = {
        "points": grid_size * 3,  # grid_len * 3,
        "tangents": grid_size * 3,  # grid_len * 3,
    }
    edge_entity_feature_map = {
        "axis_dir": 3,
        "axis_pos": 3,
        "bounding_box": 6,
        "entity_types": len(curve_type_map),
        "length": 1,
        "radius": 1,
        "start_point": 3,
        "middle_point": 3,
        "end_point": 3
    }

    #  The map of edge joint types
    joint_type_map = {
        "Other": 0,
        "Coincident": 1, 
        "Tangent": 2,
        "Concentric": 3, 
        "Parallel": 4,
        "Perpendicular": 5
    }

    def __init__(
        self,
        root_dir,
        split="train",
        random_rotate=False,
        delete_cache=False,
        limit=0,
        threads=1,
        shuffle_split=False,
        seed=42,
        center_and_scale=True,
        max_node_count=0,
        label_scheme=None,
        input_features="entity_types,length,face_reversed,edge_reversed",
        skip_far=False,
        skip_interference=False,
        skip_nurbs=False,
        joint_type="all",
        without_synthetic=False,
        feature_embedding=False,
        num_bits=9
    ):
        """
        Load the Fusion 360 Gallery joints dataset from graph data
        :param root_dir: Root path to the dataset
        :param split: string Either train, val, test, mix_test, or all set
        :param random_rotate: bool Randomly rotate the point features
        :param delete_cache: bool Delete the cached pickle files
        :param limit: int Limit the number of joints to load to this number
        :param threads: Number of threads to use for data loading
        :param shuffle_split: Shuffle the files within a split when loading from json data
        :param seed: Random seed to use
        :param center_and_scale: bool Center and scale the point features
        :param max_node_count: int Exclude joints with more than this number of combined graph nodes
        :param label_scheme: Label remapping scheme.
                Must be one of None, off, ambiguous_on, hole_on, ambiguous_hole_on
        :param input_features: Input features to use as a string separated by commas. Can include:
                points, normals, trimming_mask, entity_types, is_face, area, length,
                face_reversed, edge_reversed, reversed, convexity, dihedral_angle"
        """
        super().__init__(
            root_dir,
            split=split,
            random_rotate=random_rotate,
            delete_cache=delete_cache,
            limit=limit,
            threads=threads,
            shuffle_split=shuffle_split,
            seed=seed
        )
        self.center_and_scale = center_and_scale
        self.max_node_count = max_node_count
        self.labels_on, self.labels_off = self.parse_label_scheme_arg(label_scheme)
        self.skip_far = skip_far
        self.skip_interference = skip_interference
        self.skip_nurbs = skip_nurbs
        self.joint_type = joint_type
        self.without_synthetic = without_synthetic
        self.feature_embedding = feature_embedding
        self.num_bits = num_bits

        # Binary is / is not a joint entity
        self.num_classes = 2
        # The graphs as a (g1, g2, joint_graph) triple
        self.graphs = []
        # The graph file used to load g1 and g2
        self.graph_files = []
        # Flag indicating if the joint set geometry has holes
        self.has_holes = []
        # Transforms - a list (each joint set) of lists (each joint), containing a tuple
        # with the original 4x4 matrix transform for body1 and body2
        self.transforms = []
        # Tolerance to use for area/length comparisons
        self.rel_tol = 0.00015
        # Parse the input features requested
        feat_lists = self.parse_input_features_arg(input_features)
        self.input_features, self.grid_input_features, self.entity_input_features = feat_lists
        # Setup the cache, either deleting or loading cache data
        cache_loaded = self.setup_cache()
        if cache_loaded:
            to_remove = []
            # Filter joints based on max node count
            for i, (g1, g2, _) in enumerate(self.graphs):
                if self.max_node_count > 0:
                    if g1.num_nodes + g2.num_nodes > self.max_node_count:
                        to_remove.append(i)
            for i in reversed(to_remove):
                self.graphs.pop(i)
                self.files.pop(i)
                self.graph_files.pop(i)
                self.has_holes.pop(i)

            self.remove_all_unused_input_features()
            return

        # Get the joint files for our split
        joint_files = self.get_joint_files()

        start_time = time.time()
        if threads is None or threads < 2:
            # Serial Version
            for joint_file_name in tqdm(joint_files):
                gs = self.load_graph(joint_file_name)
                if gs is None:
                    continue
                graph1, graph2, joint_graph, joint_file, graph1_json_file, graph2_json_file, has_holes, transforms = gs
                self.files.append(joint_file.name)
                self.graphs.append([graph1, graph2, joint_graph])
                self.graph_files.append([
                    graph1_json_file.name,
                    graph2_json_file.name
                ])
                self.has_holes.append(has_holes)
                self.transforms.append(transforms)
        else:
            # Parallel Version
            graph_itr = Pool(self.threads).imap(self.load_graph, joint_files)
            for gs in tqdm(graph_itr, total=len(joint_files)):
                if gs is None:
                    continue
                graph1, graph2, joint_graph, joint_file, graph1_json_file, graph2_json_file, has_holes, transforms = gs
                self.files.append(joint_file.name)
                self.graphs.append([graph1, graph2, joint_graph])
                self.graph_files.append([
                    graph1_json_file.name,
                    graph2_json_file.name
                ])
                self.has_holes.append(has_holes)
                self.transforms.append(transforms)

        print(f"Total graph load time: {time.time() - start_time} sec")
        skipped_file_count = len(joint_files) - len(self.files)
        print(f"Skipped: {skipped_file_count} files")
        print(f"Done loading {len(self.graphs)} files")
        self.save_data_cache()

        # Remove unused input features after saving the cache
        # so we always store all features
        self.remove_all_unused_input_features()

    def quantile_quantization(
            self, 
            graph1_value, 
            graph2_value, 
            graph1_indices=None,
            graph2_indices=None,
            n_bits=9
        ):
        num_bins = int(2**n_bits)
        value_quantize1 = torch.zeros_like(graph1_value, dtype=torch.long)
        value_quantize2 = torch.zeros_like(graph2_value, dtype=torch.long)
        if graph1_indices is not None:
            quantiles = torch.quantile(
                torch.concat((graph1_value[graph1_indices], graph2_value[graph2_indices]), dim=0).float(), 
                torch.linspace(0, 1, num_bins), dim=0
            )
            value_quantize1[graph1_indices] = torch.bucketize(graph1_value[graph1_indices], quantiles)
            value_quantize2[graph2_indices] = torch.bucketize(graph2_value[graph2_indices], quantiles)
        else:
            quantiles = torch.quantile(
                torch.concat((graph1_value, graph2_value), dim=0).float(), 
                torch.linspace(0, 1, num_bins), dim=0
            )
            if len(graph1_value.shape) == 1:
                value_quantize1 = torch.bucketize(graph1_value, quantiles)
                value_quantize2 = torch.bucketize(graph2_value, quantiles)
            else:
                for i in range(graph1_value.shape[1]):
                    graph1_column = graph1_value[:, i].contiguous()
                    graph2_column = graph2_value[:, i].contiguous()
                    value_quantize1[:, i] = torch.bucketize(graph1_column, quantiles[:, i])
                    value_quantize2[:, i] = torch.bucketize(graph2_column, quantiles[:, i])
        return value_quantize1.long(), value_quantize2.long()

    def __getitem__(self, idx):
        graph1, graph2, joint_graph = self.graphs[idx]
        # Remap the augmented labels
        joint_graph = self.remap_labels(joint_graph, self.labels_on, self.labels_off)

        # Rotate if needed
        if self.random_rotate:
            rotation1 = self.get_random_rotation()
            rotation2 = self.get_random_rotation()
            if len(self.grid_input_features) > 0:
                graph1.x[:, :, :, :3] = self.rotate(graph1.x[:, :, :, :3], rotation1)
                graph1.x[:, :, :, 3:6] = self.rotate(graph1.x[:, :, :, 3:6], rotation1)
                graph2.x[:, :, :, :3] = self.rotate(graph2.x[:, :, :, :3], rotation2)
                graph2.x[:, :, :, 3:6] = self.rotate(graph2.x[:, :, :, 3:6], rotation2)
            for feature in ("axis_pos", "axis_dir", "start_point", "middle_point", "end_point"):
                if feature in self.entity_input_features:
                    graph1[feature] = self.rotate(graph1[feature], rotation1)
                    graph2[feature] = self.rotate(graph2[feature], rotation2)
            if "bounding_box" in self.entity_input_features:
                graph1.bounding_box[:, :3] = self.rotate(graph1.bounding_box[:, :3], rotation1)
                graph2.bounding_box[:, :3] = self.rotate(graph2.bounding_box[:, :3], rotation2)
                graph1.bounding_box[:, 3:6] = self.rotate(graph1.bounding_box[:, 3:6], rotation1)
                graph2.bounding_box[:, 3:6] = self.rotate(graph2.bounding_box[:, 3:6], rotation2)
        
        # Using feature quantization and embedding if needed
        if self.feature_embedding:
            for feature in ("axis_pos", "axis_dir", "bounding_box"):
                if feature in self.entity_input_features:
                    graph1[feature], graph2[feature] = self.quantile_quantization(
                        graph1[feature], 
                        graph2[feature], 
                        n_bits=self.num_bits
                    )
            
            face_indices1 = torch.where(graph1.is_face > 0.5)[0].long()
            face_indices2 = torch.where(graph2.is_face > 0.5)[0].long()
            for feature in ("area", "circumference", "param_1", "param_2"):
                if feature in self.entity_input_features:
                    graph1[feature], graph2[feature] = self.quantile_quantization(
                        graph1[feature], 
                        graph2[feature], 
                        graph1_indices=face_indices1,
                        graph2_indices=face_indices2,
                        n_bits=self.num_bits
                    )
            
            edge_indices1 = torch.where(graph1["is_face"] <= 0.5)[0].long()
            edge_indices2 = torch.where(graph2["is_face"] <= 0.5)[0].long()
            for feature in ("length", "radius", "start_point", "middle_point", "end_point"):
                if feature in self.entity_input_features:
                    graph1[feature], graph2[feature] = self.quantile_quantization(
                        graph1[feature], 
                        graph2[feature], 
                        graph1_indices=edge_indices1,
                        graph2_indices=edge_indices2,
                        n_bits=self.num_bits
                    )
        
        return [graph1, graph2, joint_graph]

    def collate_fn_fixed_batch_size(self, batch):
        bg1 = Batch.from_data_list([x[0] for x in batch])
        bg2 = Batch.from_data_list([x[1] for x in batch])
        jg = Batch.from_data_list([x[2] for x in batch])
        return bg1, bg2, jg

    def get_train_dataloader(self, max_nodes_per_batch=0, batch_size=1, shuffle=True, num_workers=0):
        if max_nodes_per_batch > 0:
            return JointGraphBatchDataLoader(self, max_nodes_per_batch=max_nodes_per_batch, shuffle=shuffle, drop_last=True)
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn_fixed_batch_size,
            num_workers=num_workers,
            drop_last=True,
        )

    def get_test_dataloader(self, batch_size=1, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_fixed_batch_size,
            num_workers=num_workers,
            drop_last=True,
        )

    @staticmethod
    def remap_labels(joint_graph, labels_on, labels_off):
        """Remap the labels for label augmentation"""
        # Return the labels as is,
        if labels_on is None or labels_off is None:
            return joint_graph
        label_matrix = joint_graph.edge_attr
        # Turn the labels we want on, on
        for label_on in labels_on:
            label_matrix[label_matrix == JointGraphDataset.label_map[label_on]] = 1
        # Turn the labels we want off, off
        for label_off in labels_off:
            label_matrix[label_matrix == JointGraphDataset.label_map[label_off]] = 0
        joint_graph.edge_attr = label_matrix
        joint_graph.joint_type_list = joint_graph.joint_type_matrix[label_matrix == 1]
        return joint_graph

    @staticmethod
    def parse_label_scheme_arg(label_scheme_string=None):
        """Parse the input feature lists from the arg string"""
        all_labels = list(JointGraphDataset.label_map.keys())
        all_labels_set = set(all_labels)
        if label_scheme_string is None:
            return None, None
        else:
            label_scheme_set = set(label_scheme_string.split(","))
        assert len(label_scheme_set) > 0
        # Assert if we have an invalid label
        for lss in label_scheme_set:
            assert lss in all_labels_set, f"Invalid label: {lss}"

        # Loop over to add them in the same order
        labels_on = []
        labels_off = []
        for label in all_labels:
            # Always set the gt joints on
            if label == "Joint":
                labels_on.append(label)
                continue
            # Always set the gt non-joints off
            if label == "Non-joint":
                labels_off.append(label)
                continue
            if label in label_scheme_set:
                # Labels we want to turn on
                labels_on.append(label)
            else:
                # Labels we want to turn off
                labels_off.append(label)
        return labels_on, labels_off

    def remove_all_unused_input_features(self):
        """Remove unused input features in all graphs"""
        for g1, g2, jg in self.graphs:
            # Remove the features we don't need to save GPU space
            g1 = self.remove_unused_input_features(g1)
            g2 = self.remove_unused_input_features(g2)

    def remove_unused_input_features(self, g):
        """Remove any unused input features"""
        # 'Remove' here is setting the features to None
        # as we can't delete the class properties
        edge_index = g.edge_index
        for g_key in g.keys():
            if g_key == "edge_index" or g_key == "is_face":
                continue
            if g_key == "x":
                # Remove all grid features as a group
                # only if we aren't using any
                if len(self.grid_input_features) == 0:
                    g.x = None
            else:
                if g_key not in self.entity_input_features:
                    g[g_key] = None
        return g

    @staticmethod
    def parse_input_features_arg(input_features_string=None, input_feature_type="both"):
        """Parse the input feature lists from the arg string"""
        all_face_grid = list(JointGraphDataset.face_grid_feature_map.keys())
        all_face_entity = list(JointGraphDataset.face_entity_feature_map.keys())
        all_edge_grid = list(JointGraphDataset.edge_grid_feature_map.keys())
        all_edge_entity = list(JointGraphDataset.edge_entity_feature_map.keys())
        if input_features_string is None:
            # Default to all features
            input_features = set(all_face_grid + all_face_entity + all_edge_grid + all_edge_entity)
        else:
            input_features = set(input_features_string.split(","))
        assert len(input_features) > 0

        # Filter the features to be either face, edge, or both
        if input_feature_type == "both":
            grid_feat = set(all_face_grid + all_edge_grid)
            entity_feat = set(all_face_entity + all_edge_entity)
        elif input_feature_type == "face":
            grid_feat = set(all_face_grid)
            entity_feat = set(all_face_entity)
        elif input_feature_type == "edge":
            grid_feat = set(all_edge_grid)
            entity_feat = set(all_edge_entity)
        both_feat = grid_feat.union(entity_feat)

        # Make sure we create a canonical order after performing non-deterministic set intersection
        ordered_feat = JointGraphDataset.get_default_input_features()
        return (
            JointGraphDataset.order_input_features(input_features.intersection(both_feat), ordered_feat),
            JointGraphDataset.order_input_features(input_features.intersection(grid_feat), ordered_feat),
            JointGraphDataset.order_input_features(input_features.intersection(entity_feat), ordered_feat)
        )

    @staticmethod
    def get_default_input_features():
        """Return a list of all input features in a canonical order without duplicates"""
        all_face_grid = list(JointGraphDataset.face_grid_feature_map.keys())
        all_face_entity = list(JointGraphDataset.face_entity_feature_map.keys())
        all_edge_grid = list(JointGraphDataset.edge_grid_feature_map.keys())
        all_edge_entity = list(JointGraphDataset.edge_entity_feature_map.keys())
        # List with duplicates
        all_feat_dup = all_face_grid + all_face_entity + all_edge_grid + all_edge_entity
        # List without duplicates, this works as dict guarantees order from python 3.7 on
        all_feat = list(dict.fromkeys(all_feat_dup))
        return all_feat

    @staticmethod
    def order_input_features(input_features_set, ordered_features=None):
        """Order an input feature set according to a canonical order"""
        if ordered_features is None:
            ordered_features = JointGraphDataset.get_default_input_features()
        ordered_output = []
        for feat in ordered_features:
            if feat in input_features_set:
                ordered_output.append(feat)
        return ordered_output

    @staticmethod
    def get_input_feature_size(input_features, input_feature_type):
        """Get the size of a given list of input features"""
        assert input_feature_type in {"face", "edge"}
        if input_feature_type == "face":
            feature_map = {}
            feature_map.update(JointGraphDataset.face_grid_feature_map)
            feature_map.update(JointGraphDataset.face_entity_feature_map)
        elif input_feature_type == "edge":
            feature_map = {}
            feature_map.update(JointGraphDataset.edge_grid_feature_map)
            feature_map.update(JointGraphDataset.edge_entity_feature_map)
        return sum([feature_map[f] for f in input_features])

    def load_graph_json_data(self, json_file):
        """Load and return the graph json data"""
        with open(json_file, encoding="utf8") as f:
            graph_json_data = json.load(f)

        # Check that we don't have any duplicate ids
        ids = [node["id"] for node in graph_json_data["nodes"]]
        assert len(ids) == len(set(ids)), "Duplicate ids in graph found"
        # assert np.all(np.diff(ids) > 0), "Ids not increasing"

        # Make the node id's sequential
        id_map = {}
        for index, node_id in enumerate(ids):
            id_map[node_id] = index

        # Reassign the node ids sequentially
        for node in graph_json_data["nodes"]:
            node["id"] = id_map[node["id"]]
        # Reassign the node ids sequentially and
        # manually duplicate the edges to keep track
        # rather than the automated conversion to
        # a bidirectional graph
        links = []
        edge_index = 0
        for link in graph_json_data["links"]:
            # Add the original link
            link["source"] = id_map[link["source"]]
            link["target"] = id_map[link["target"]]
            link["id"] = edge_index
            link["duplicates"] = False
            links.append(link)
            edge_index += 1
            # Now add the duplicate link with reverse direction
            # We do a shallow copy here
            # to avoid copying the arrays
            link_copy = copy.copy(link)
            # Manually reverse the link direction
            link_copy["source"] = link["target"]
            link_copy["target"] = link["source"]
            # Set the incremented id
            link_copy["id"] = edge_index
            # Add a flag to indicate a duplicated edge
            link_copy["duplicates"] = True
            links.append(link_copy)
            edge_index += 1
        graph_json_data["links"] = links
        # Set the type of graph to a directed one
        graph_json_data["directed"] = True
        return graph_json_data

    def rotate(self, inp, rotation):
        """Rotate the node features in the graph by a given rotation"""
        rotation = rotation.to(inp.device)
        orig_size = inp.size()
        inp = torch.mm(inp.view(-1, 3), rotation).view(orig_size)
        return inp

    @staticmethod
    def get_bounding_box(inp):
        pts = inp[:, :, :, :3].reshape((-1, 3))
        mask = inp[:, :, :, 6].reshape(-1)
        point_indices_inside_faces = mask == 1
        pts = pts[point_indices_inside_faces, :]
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
        return torch.tensor(box)

    @staticmethod
    def get_common_scale(grid1, grid2):
        bbox1 = JointGraphDataset.get_bounding_box(grid1)
        bbox2 = JointGraphDataset.get_bounding_box(grid2)
        # bbox_min = torch.minimum(bbox1[0], bbox2[0])
        # bbox_max = torch.maximum(bbox1[1], bbox2[1])
        # span = bbox_max - bbox_min
        # max_span = torch.max(span)
        # scale = 2.0 / max_span
        bboxes = torch.cat((bbox1, bbox2))
        scale = (1.0 / bboxes.abs().max()) * 0.999999
        return scale

    def scale_features(
        self,
        g1,
        g2,
        bbox1=None,
        bbox2=None
    ):
        """Scale the points for both graphs"""
        # Get the combined bounding box
        if g1.x is not None and g1.x.numel() != 0 and g2.x is not None and g2.x.numel() != 0:
            scale = JointGraphDataset.get_common_scale(g1.x, g2.x)
            g1.x[:, :, :, :3] *= scale
            # Check we aren't too far out of bounds due to the masked surface
            if torch.max(g1.x[:, :, :, :3]) > 2.0 or torch.max(g1.x[:, :, :, :3]) < -2.0:
                return False
            g2.x[:, :, :, :3] *= scale
            if torch.max(g2.x[:, :, :, :3]) > 2.0 or torch.max(g2.x[:, :, :, :3]) < -2.0:
                return False
        else:
            bbox_min = torch.tensor([
                bbox1["min_point"]["x"], 
                bbox1["min_point"]["y"], 
                bbox1["min_point"]["z"],
                bbox2["min_point"]["x"], 
                bbox2["min_point"]["y"], 
                bbox2["min_point"]["z"]
            ])
            bbox_max = torch.tensor([
                bbox1["max_point"]["x"], 
                bbox1["max_point"]["y"], 
                bbox1["max_point"]["z"],
                bbox2["max_point"]["x"], 
                bbox2["max_point"]["y"], 
                bbox2["max_point"]["z"]
            ])
            span = bbox_max - bbox_min
            max_span = torch.max(span)
            scale = 2.0 / max_span
        
        for feature in ("axis_pos", "bounding_box", "circumference", "param_1", "param_2", \
                        "length", "radius", "start_point", "middle_point", "end_point"):
            if g1[feature] is not None:
                g1[feature] *= scale
                g2[feature] *= scale
        if g1.area is not None:
            g1.area *= scale * scale
            g2.area *= scale * scale
        return True
    
    def is_overlapping(self, box1, box2):
        """
        Function to check if two 3D bounding boxes overlap.

        Parameters:
        box1 (tuple): Tuple representing the coordinates of box1 in the format (x1, y1, z1, x2, y2, z2),
                    where (x1, y1, z1) represents the coordinates of one corner,
                    and (x2, y2, z2) represents the coordinates of the opposite corner.
        box2 (tuple): Tuple representing the coordinates of box2 in the same format as box1.

        Returns:
        bool: True if the boxes overlap, False otherwise.
        """
        x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
        x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2

        # Check if there is no overlap in any dimension
        if x1_max < x2_min or x2_max < x1_min or \
        y1_max < y2_min or y2_max < y1_min or \
        z1_max < z2_min or z2_max < z1_min:
            return False
        else:
            return True

    def load_graph(self, joint_file_name):
        """Load a joint file and return a graph"""
        joint_file = self.root_dir / joint_file_name
        with open(joint_file, encoding="utf8") as f:
            joint_data = json.load(f)
        # Skip pairs that have interference
        if self.skip_interference:
            if "interference" in joint_data and joint_data["interference"]:
                return None
        g1, g1d, face_count1, edge_count1, g1_json_file = self.load_graph_body(
            joint_data["body_one"])
        if g1 is None:
            return None
        g2, g2d, face_count2, edge_count2, g2_json_file = self.load_graph_body(
            joint_data["body_two"])
        if g2 is None:
            return None
        # Limit the maximum number of combined nodes
        total_nodes = face_count1 + edge_count1 + face_count2 + edge_count2
        if self.max_node_count > 0:
            if total_nodes > self.max_node_count:
                return None
        # Skip parts pairs that are far away
        if self.skip_far:
            bbox1, bbox2 = g1d["properties"]["bounding_box"], g2d["properties"]["bounding_box"]
            bbox1 = (
                bbox1["min_point"]["x"],
                bbox1["min_point"]["y"],
                bbox1["min_point"]["z"],
                bbox1["max_point"]["x"],
                bbox1["max_point"]["y"],
                bbox1["max_point"]["z"],
            )
            bbox2 = (
                bbox2["min_point"]["x"],
                bbox2["min_point"]["y"],
                bbox2["min_point"]["z"],
                bbox2["max_point"]["x"],
                bbox2["max_point"]["y"],
                bbox2["max_point"]["z"],
            )
            if not self.is_overlapping(bbox1, bbox2):
                return None
        # Get the joint label matrix
        label_matrix, joint_type_list = self.get_label_matrix(
            joint_data,
            g1, g2,
            g1d, g2d,
            face_count1, face_count2,
            edge_count1, edge_count2
        )
        if label_matrix.sum() == 0:
            return None
        # Create the joint graph from the label matrix
        joint_graph = self.make_joint_graph(g1, g2, label_matrix, joint_type_list, joint_file_name)
        # Scale geometry features from both graphs with a common scale
        if self.center_and_scale:
            scale_good = self.scale_features(
                g1,
                g2,
                bbox1=g1d["properties"]["bounding_box"],
                bbox2=g2d["properties"]["bounding_box"],
            )
            # Throw out if we can't scale properly due to the masked surface area
            if not scale_good:
                print("Discarding graph with bad scale")
                return None
        # Flag to indicate if this design has holes
        holes = joint_data.get("holes", [])
        has_holes = len(holes) > 0
        # Transforms for this joint set
        transforms = self.get_joint_transforms(joint_data)
        return g1, g2, joint_graph, joint_file, g1_json_file, g2_json_file, has_holes, transforms

    def load_graph_body(self, body):
        """Load a graph created from a brep body"""
        body_json_file = self.root_dir / f"{body}.json"
        # Graph json that we feed in to make the graph
        g_json = self.load_graph_json_data(body_json_file)
        # Copy of the graph data
        gd = {"nodes": [], "properties": g_json["properties"]}
        # Center and scale if needed
        bbox = g_json["properties"]["bounding_box"]
        center = self.get_center_from_bounding_box(bbox)

        # Get the joint node indices that serve as labels
        face_count = g_json["properties"]["face_count"]
        edge_count = g_json["properties"]["edge_count"]
        node_count = len(g_json["nodes"])
        link_count = len(g_json["links"])
        # Throw out graphs without edges or with too many nodes
        if node_count < 2 or (self.max_node_count > 0 and node_count > self.max_node_count):
            return None, None, face_count, edge_count, body_json_file
        if link_count <= 0:
            return None, None, face_count, edge_count, body_json_file

        for _, node in enumerate(g_json["nodes"]):
            # Pull out the node features
            node["x"] = self.get_grid_features(node, center)
            if node["x"] is None:
                return None, None, face_count, edge_count, body_json_file
            # Get features for axis
            node["axis_pos"], node["axis_dir"] = self.get_node_axis(node, center)
            # Get features for bounding box
            node["bounding_box"] = self.get_node_bounding_box(node, center)

            # Pull out the surface or curve type
            node["entity_types"] = self.get_node_entity_type(node)
            # Feature indicating if this node is a B-Rep face
            node["is_face"] = torch.tensor(int("surface_type" in node), dtype=torch.long)

            # Features for faces or edges
            self.get_surface_params(node)
            self.get_curve_params(node, center)

            # Remove the grid node features to save extra copying
            # as we have already pulled out what we need
            self.delete_features(node)

        # Load the networkx graph file
        nxg = json_graph.node_link_graph(g_json)
        # Convert to a graph
        g = from_networkx(nxg)
        g = self.reshape_graph_features(g)
        # Check we have the same number of nodes and edges
        assert nxg.number_of_edges() == g.num_edges
        assert nxg.number_of_nodes() == g.num_nodes
        return g, gd, face_count, edge_count, body_json_file

    def delete_features(self, node):
        """Delete node feature provided in a list"""
        # Delete the grid node features
        features_to_keep = [
            "x",
            "axis_pos",
            "axis_dir",
            "bounding_box",
            "entity_types",
            "is_face",
            "area",
            "circumference",
            "param_1",
            "param_2",
            "length",
            "radius",
            "start_point",
            "middle_point",
            "end_point"
        ]
        keys = list(node.keys())
        for key in keys:
            if key not in features_to_keep:
                del node[key]

    def copy_features(self, node, gd):
        """Copy features and store"""
        # Ignore the grid features
        features_to_ignore = [
            "points",
            "normals",
            "trimming_mask",
            "tangents",
        ]
        node_copy = {}
        for key in node.keys():
            if key not in features_to_ignore:
                node_copy[key] = node[key]
        gd["nodes"].append(node_copy)

    def reshape_graph_features(self, g):
        """Reshape the multi-dimensional graph features from a list to tensors"""
        if g.x is not None and g.x.numel() != 0:
            g.x = g.x.reshape(
                (-1, self.grid_size, self.grid_size, self.grid_channels))
        return g

    def get_grid_features(self, node, center):
        """Return the grid features for a single node"""
        if len(node["points"]) == 0:
            return torch.tensor([])
        
        # B-Rep Face
        if "surface_type" in node:
            # 1D array with XYZ point values in a 2D grid
            # x1,y1,z1,x2,y2...
            points = node["points"]
            # 1D array with XYZ normal values in a 2D grid
            normals = node["normals"]
            # 1D array of binary numbers
            mask = node["trimming_mask"]
            # Reshape and concat
            points_2d = np.array(points).reshape((self.grid_len, 3))
            normals_2d = np.array(normals).reshape((self.grid_len, 3))
            # Normalize to vector length 1
            if np.any(np.linalg.norm(normals_2d, axis=1) == 0):
                return None
            normals_2d = normals_2d / np.linalg.norm(normals_2d, axis=1).reshape((-1, 1))
            # Center and we will scale later together with other features
            if self.center_and_scale:
                points_2d -= center
            mask_2d = np.array(mask).reshape((self.grid_len, 1))

        # B-Rep Edges
        elif "curve_type" in node:
            # 1D array with XYZ point values in a 1D grid
            points = node["points"]
            # Repeat the 10 points we have to fill a 10x10
            points_tile = np.tile(points, 10)
            points_2d = np.reshape(points_tile, (self.grid_len, 3))
            # Center and we will scale later together with other features
            if self.center_and_scale:
                points_2d -= center
            # No normals, just zeros
            # normals_2d = np.zeros((self.grid_len, 3))
            # Tangents - stored in the normals channel
            tangents = np.array(node["tangents"])
            # Repeat the 10 tangents we have to fill a 10x10
            tangents_tile = np.tile(tangents, 10)
            normals_2d = np.reshape(tangents_tile, (self.grid_len, 3))
            # Normalize to vector length 1
            normals_2d = normals_2d / np.linalg.norm(normals_2d, axis=1).reshape((-1, 1))
            # Mask is just 1 for everything
            mask_2d = np.ones((self.grid_len, 1))

        pts_normals_mask = np.concatenate((points_2d, normals_2d, mask_2d), axis=1)
        pts_normals_mask = pts_normals_mask.reshape(
            self.grid_size, self.grid_size, self.grid_channels
        )
        return torch.from_numpy(pts_normals_mask).float()

    def get_node_axis(self, node, center):
        if "axis_pos_x" in node:
            origin, direction = joint_axis.find_axis_line(node, return_numpy=True)
            origin -= center
            return torch.from_numpy(origin).float(), torch.from_numpy(direction).float()
        else:
            return None, None

    def get_node_bounding_box(self, node, center):
        if "bounding_box" in node:
            max_point = joint_axis.get_point(node["bounding_box"]["max_point"]) - center
            min_point = joint_axis.get_point(node["bounding_box"]["min_point"]) - center
            bounding_box = np.concatenate((max_point, min_point))
            return torch.from_numpy(bounding_box).float()
        else:
            return None

    def get_node_entity_type(self, node):
        """Get the entity type, either surface or curve type for the node"""
        if "surface_type" in node:
            surface_type = self.surface_type_map[node["surface_type"]]
            return torch.tensor(surface_type, dtype=torch.long)
        elif "curve_type" in node:
            curve_type = self.curve_type_map[node["curve_type"]]
            return torch.tensor(curve_type, dtype=torch.long)
        else:
            raise Exception("Unknown node entity type")
    
    def get_surface_params(self, node):
        if "surface_type" in node:
            node["area"] = torch.tensor(node["area"], dtype=torch.float)
            node["circumference"] = torch.tensor(node["circumference"], dtype=torch.float)
            node["param_1"] = torch.tensor(node["param_1"], dtype=torch.float)
            node["param_2"] = torch.tensor(node["param_2"], dtype=torch.float)
        else:
            node["area"] = torch.tensor(0, dtype=torch.float)
            node["circumference"] = torch.tensor(0, dtype=torch.float)
            node["param_1"] = torch.tensor(0, dtype=torch.float)
            node["param_2"] = torch.tensor(0, dtype=torch.float)

    def get_curve_params(self, node, center):
        if "curve_type" in node:
            node["length"] = torch.tensor(node["length"], dtype=torch.float)
            node["radius"] = torch.tensor(node["radius"], dtype=torch.float)
            node["start_point"] = torch.from_numpy(joint_axis.get_point(node, "start_point") - center).float()
            node["middle_point"] = torch.from_numpy(joint_axis.get_point(node, "middle_point") - center).float()
            node["end_point"] = torch.from_numpy(joint_axis.get_point(node, "end_point") - center).float()
        else:
            node["length"] = torch.tensor(0, dtype=torch.float)
            node["radius"] = torch.tensor(0, dtype=torch.float)
            node["start_point"] = torch.tensor([0, 0, 0], dtype=torch.float)
            node["middle_point"] = torch.tensor([0, 0, 0], dtype=torch.float)
            node["end_point"] = torch.tensor([0, 0, 0], dtype=torch.float)

    def get_node_area_length(self, node):
        """Get the area or length of a node"""
        if "surface_type" in node:
            return node["area"]
        elif "curve_type" in node:
            return node["length"]

    def add_joint_index(joint_indices, entity_index, entity_count):
        """Add a joint index to the list, checking its bounds"""
        # Check that the indices are within bounds
        assert entity_index >= 0
        assert entity_index < entity_count
        joint_indices.append(entity_index)

    def offset_joint_index(self, entity_index, entity_type, face_count, entity_count):
        """Offset the joint index for the label matrix"""
        joint_index = entity_index
        if entity_type == "BRepEdge":
            # If this is a brep edge we need to increment the index
            # to start past the number of faces as those are stored first
            joint_index += face_count
        # If we have a BRepFace life is simple...
        assert joint_index >= 0
        assert joint_index < entity_count
        return joint_index

    def get_label_matrix(self, joint_data, g1, g2, g1d, g2d, face_count1, face_count2, edge_count1, edge_count2):
        """Get the label matrix containing user selected entities and various label augmentations"""
        joints = joint_data["joints"]
        holes = joint_data.get("holes", [])
        entity_count1 = face_count1 + edge_count1
        entity_count2 = face_count2 + edge_count2
        # Labels are as follows:
        # 0 - Non joint
        # 1 - Joints (selected by user)
        # 2 - Ambiguous joint
        # 3 - Joint equivalents
        # 4 - Ambiguous joint equivalents
        # 5 - Hole
        # 6 - Hole equivalents
        label_matrix = torch.zeros((entity_count1, entity_count2), dtype=torch.long)
        joint_type_matrix = torch.zeros((entity_count1, entity_count2), dtype=torch.long)
        for i, joint in enumerate(joints):  
            # Check synthetic
            if self.without_synthetic:
                if "is_synthetic" in joint and joint["is_synthetic"]:
                    continue
            entity1 = joint["geometry_or_origin_one"]["entity_one"]
            entity1_index = entity1["index"]
            entity1_type = entity1["type"]
            entity2 = joint["geometry_or_origin_two"]["entity_one"]
            entity2_index = entity2["index"]
            entity2_type = entity2["type"]
            joint_type = self.joint_type_map.get(joint["joint_type"], 0)
            # Check joint type
            if self.joint_type != "all":
                using_joint_types = self.joint_type.split(',')
                flag = False
                for item in using_joint_types:
                    if joint_type == self.joint_type_map[item]:
                        flag = True
                        break
                if not flag:
                    continue
            # Check nurbs
            if self.skip_nurbs:
                if entity1_type == "BRepFace" and entity1["surface_type"] == "NurbsSurfaceType":
                    continue
                if entity2_type == "BRepFace" and entity2["surface_type"] == "NurbsSurfaceType":
                    continue
                if entity1_type == "BRepEdge" and entity1["curve_type"] == "NurbsCurve3DCurveType":
                    continue
                if entity2_type == "BRepEdge" and entity2["curve_type"] == "NurbsCurve3DCurveType":
                    continue
            # Offset the joint indices for use in the label matrix
            entity1_index = self.offset_joint_index(
                entity1_index, entity1_type, face_count1, entity_count1)
            entity2_index = self.offset_joint_index(
                entity2_index, entity2_type, face_count2, entity_count2)
            # Set the joint equivalent indices
            eq1_indices = self.get_joint_equivalents(
                joint["geometry_or_origin_one"], face_count1, entity_count1)
            eq2_indices = self.get_joint_equivalents(
                joint["geometry_or_origin_two"], face_count2, entity_count2)
            # Add the actual entities
            eq1_indices.append(entity1_index)
            eq2_indices.append(entity2_index)
            # For every pair we set a joint
            for eq1_index in eq1_indices:
                for eq2_index in eq2_indices:
                    # Only set non-joints, we don't want to replace other labels
                    if label_matrix[eq1_index][eq2_index] == self.label_map["Non-joint"]:
                        label_matrix[eq1_index][eq2_index] = self.label_map["JointEquivalent"]
                        joint_type_matrix[eq1_index][eq2_index] = joint_type
            # Set the user selected joint indices
            label_matrix[entity1_index][entity2_index] = self.label_map["Joint"]
            joint_type_matrix[entity1_index][entity2_index] = joint_type
        # # Include ambiguous and hole labels
        # # Adding separate labels to the label_matrix
        # # We need to do this after all joints are marked out as labels
        # g1_ambiguous, g2_ambiguous = self.set_ambiguous_labels(g1, g2, label_matrix)
        # g1_holes, g2_holes = self.set_hole_labels(g1d, g2d, label_matrix, joint_data)

        # # Only do further work if we have holes or ambiguous entities
        # eq_count = len(g1_ambiguous) + len(g2_ambiguous) + len(g1_holes) + len(g2_holes)
        # if eq_count > 0:
        #     # First calculate the axis lines and cache them
        #     g1_axis_lines = self.get_axis_lines_from_graph(g1d)
        #     g2_axis_lines = self.get_axis_lines_from_graph(g2d)

        #     # Now find and set the equivalents
        #     self.set_equivalents(
        #         g1_ambiguous, g2_ambiguous,
        #         g1_axis_lines, g2_axis_lines,
        #         label_matrix, self.label_map["AmbiguousEquivalent"]
        #     )
        #     self.set_equivalents(
        #         g1_holes, g2_holes,
        #         g1_axis_lines, g2_axis_lines,
        #         label_matrix, self.label_map["HoleEquivalent"]
        #     )
        return label_matrix, joint_type_matrix

    def make_joint_graph(self, graph1, graph2, label_matrix, joint_type_matrix, joint_file_name):
        """Create a joint graph connecting graph1 and graph2 densely"""
        nodes_indices_first_graph = torch.arange(graph1.num_nodes)
        # We want to treat both graphs as one, so order the indices of the second graph's nodes
        # sequentially after the first graph's node indices
        nodes_indices_second_graph = torch.arange(graph2.num_nodes) + graph1.num_nodes
        edges_between_graphs_1 = torch.cartesian_prod(nodes_indices_first_graph, nodes_indices_second_graph).transpose(1, 0)
        # Pradeep: turn these on of we want bidirectional edges among the bodies
        # edges_between_graphs_2 = torch.cartesian_prod(nodes_indices_second_graph, nodes_indices_first_graph).transpose(1, 0)
        # edges_between_graphs = torch.cat((edges_between_graphs_1, edges_between_graphs_2), dim=0)
        num_nodes = graph1.num_nodes + graph2.num_nodes
        empty = torch.zeros((num_nodes, 1))
        joint_graph = Data(x=empty, edge_index=edges_between_graphs_1)
        joint_graph.num_nodes = num_nodes
        joint_graph.edge_attr = label_matrix.view(-1)
        joint_graph.num_nodes_graph1 = graph1.num_nodes
        joint_graph.num_nodes_graph2 = graph2.num_nodes
        joint_graph.joint_type_matrix = joint_type_matrix.view(-1)
        joint_graph.joint_file_name = joint_file_name
        return joint_graph

    def get_joint_equivalents(self, geometry, face_count, entity_count):
        """Get the joint equivalent indices that are preprocessed in the graph json"""
        indices = []
        if "entity_one_equivalents" in geometry:
            for entity in geometry["entity_one_equivalents"]:
                index = self.offset_joint_index(
                    entity["index"],
                    entity["type"],
                    face_count,
                    entity_count
                )
                indices.append(index)
        return indices

    def set_ambiguous_labels(self, g1, g2, label_matrix):
        """
        Set the ambiguous labels by setting
        the ambiguous entities in label_matrix to a different value
        """
        joint_entity_indices = (label_matrix == self.label_map["Joint"]).nonzero(as_tuple=False)
        g1_entities = joint_entity_indices[:, 0].tolist()
        g2_entities = joint_entity_indices[:, 1].tolist()
        g1_ambiguous_labels = self.get_ambiguous_labels(g1, set(g1_entities))
        g2_ambiguous_labels = self.get_ambiguous_labels(g2, set(g2_entities))
        # Keep track of the ambiguous labels that we actually apply to the label matrix
        g1_ambiguous_labels_applied = []
        g2_ambiguous_labels_applied = []
        # We want to point from the joint entity on one body
        # to all of the ambiguous entities on the other body
        for index, g1_ambiguous_label_indices in g1_ambiguous_labels.items():
            g2_entity_index = g2_entities[index]
            for g1_ambiguous_label in g1_ambiguous_label_indices:
                # Only set non-joints, we don't want to replace other labels
                if label_matrix[g1_ambiguous_label][g2_entity_index] == self.label_map["Non-joint"]:
                    label_matrix[g1_ambiguous_label][g2_entity_index] = self.label_map["Ambiguous"]
                    g1_ambiguous_labels_applied.append({
                        "target": g1_ambiguous_label,
                        "pair": g2_entity_index
                    })
        for index, g2_ambiguous_label_indices in g2_ambiguous_labels.items():
            g1_entity_index = g1_entities[index]
            for g2_ambiguous_label in g2_ambiguous_label_indices:
                if label_matrix[g1_entity_index][g2_ambiguous_label] == self.label_map["Non-joint"]:
                    label_matrix[g1_entity_index][g2_ambiguous_label] = self.label_map["Ambiguous"]
                    g2_ambiguous_labels_applied.append({
                        "target": g2_ambiguous_label,
                        "pair": g1_entity_index
                    })
        return g1_ambiguous_labels_applied, g2_ambiguous_labels_applied

    def get_num_out_edges(self, g, node_index):
        """Get the number of outgoing edges from a graph node"""
        # Source edges
        edge_src = g.edge_index[0]
        # Mask for the number of source edges from the given node index
        mask = edge_src == node_index
        # Tally up the number
        return mask.sum()

    def get_ambiguous_labels(self, g, joint_entities):
        """
        Get the ambiguous labels for those found in graph g
        i.e. unlabeled entities that look similar to labeled ones
        """
        # Store with joint entity index as key and an array of ambiguous indices
        ambiguous_labels = {}
        for index, entity in enumerate(joint_entities):
            # This is the user selected entity
            entity_area = g.area[entity]
            entity_length = g.length[entity]
            entity_is_face = g.is_face[entity]
            entity_type = int(torch.argmax(g.entity_types[entity]))
            entity_convexity = int(torch.argmax(g.convexity[entity]))
            entity_num_links = self.get_num_out_edges(g, entity)

            for i in range(g.num_nodes):
                # Ignore matches to other entities already selected by the user
                if i in joint_entities:
                    continue
                # Ignore entities that are labels already
                label_area = g.area[i]
                label_length = g.length[i]
                label_is_face = g.is_face[i]
                label_type = int(torch.argmax(g.entity_types[i]))
                label_convexity = int(torch.argmax(g.convexity[i]))
                label_num_links = self.get_num_out_edges(g, i)
                # If the entity type matches
                if entity_type == label_type and entity_num_links == label_num_links:
                    # If this is a face compare area
                    if label_is_face:
                        if math.isclose(entity_area, label_area, rel_tol=self.rel_tol):
                            if index not in ambiguous_labels:
                                ambiguous_labels[index] = []
                            ambiguous_labels[index].append(i)

                    # If this is an edge compare length and convexity
                    else:
                        if entity_convexity == label_convexity:
                            if math.isclose(entity_length, label_length, rel_tol=self.rel_tol):
                                if index not in ambiguous_labels:
                                    ambiguous_labels[index] = []
                                ambiguous_labels[index].append(i)
        return ambiguous_labels

    def get_axis_lines_from_graph(self, gd):
        """Calculate the axis lines for all entities in a graph"""
        axis_lines = []
        for node in gd["nodes"]:
            # We also add None items here so the node indices match
            axis_line = joint_axis.find_axis_line(node, return_numpy=True)
            axis_lines.append(axis_line)
        return axis_lines

    def set_equivalents(self, g1_entities, g2_entities,
                        g1_axis_lines, g2_axis_lines, label_matrix, label):
        """Set labels for equivalents to entities we found in both graphs"""
        if len(g1_entities) > 0:
            self.set_graph_equivalents(g1_entities, g1_axis_lines, label_matrix, label, 1)
        if len(g2_entities) > 0:
            self.set_graph_equivalents(g2_entities, g2_axis_lines, label_matrix, label, 2)

    def set_graph_equivalents(self, entities, axis_lines, label_matrix, label, graph_order):
        """Set labels for equivalents to entities we found in a single graph"""
        # Next we check if the axis lines are colinear making an equivalent
        equivalents = []
        # Set of all the indices to check against
        target_indices = set([f["target"] for f in entities])
        # For every target entity we identified
        for entity_dict in entities:
            # The index of the target node in the graph
            target_index = entity_dict["target"]
            # The pair in the other body it is connected to
            pair_index = entity_dict["pair"]
            # The axis line of the target
            target_axis_line = axis_lines[target_index]
            if target_axis_line is None:
                continue
            target_origin, target_direction = target_axis_line
            if target_origin is None or target_direction is None:
                continue

            # Traverse all nodes and see if there are equivalents
            for node_index, axis_line in enumerate(axis_lines):
                if axis_line is None:
                    continue
                origin, direction = axis_line
                if origin is None or direction is None:
                    continue
                # Don't bother if this is another target entity
                if node_index in target_indices:
                    continue
                # Only test and set non joint entities
                if graph_order == 1:
                    if label_matrix[node_index][pair_index] != self.label_map["Non-joint"]:
                        continue
                elif graph_order == 2:
                    if label_matrix[pair_index][node_index] != self.label_map["Non-joint"]:
                        continue
                is_colinear = joint_axis.check_colinear_with_tolerance(
                    target_axis_line, axis_line
                )
                if not is_colinear:
                    continue
                if graph_order == 1:
                    label_matrix[node_index][pair_index] = label
                elif graph_order == 2:
                    label_matrix[pair_index][node_index] = label

    def set_hole_labels(self, g1d, g2d, label_matrix, joint_data):
        """Set hole labels between a hole and a mating entity"""
        holes = joint_data.get("holes", [])
        if holes is None or len(holes) == 0:
            return [], []
        # Split the hole entities into those belonging to graph 1 and 2
        g1_hole_entities, g2_hole_entities = self.get_graph_holes(
            holes,
            joint_data["body_one"],
            joint_data["body_two"]
        )
        g1_holes = self.set_graph_hole_labels(g1_hole_entities, g1d, g2d, label_matrix, 1)
        g2_holes = self.set_graph_hole_labels(g2_hole_entities, g2d, g1d, label_matrix, 2)
        # Extend with reverse labels so we find equivalents for both target and pair
        g1_holes_rev = self.reverse_graph_hole_labels(g1_holes)
        g2_holes_rev = self.reverse_graph_hole_labels(g2_holes)
        g1_holes.extend(g2_holes_rev)
        g2_holes.extend(g1_holes_rev)
        return g1_holes, g2_holes

    def get_graph_holes(self, holes, body_one, body_two):
        """Get the hole data according to each graph"""
        g1_holes = []
        g2_holes = []
        for hole in holes:
            hole_entity_lists = [hole["faces"], hole["edges"]]
            for hole_entity_list in hole_entity_lists:
                for hole_entity in hole_entity_list:
                    if hole["body"] == body_one:
                        g1_holes.append(hole_entity)
                    elif hole["body"] == body_two:
                        g2_holes.append(hole_entity)
                    else:
                        raise Exception("Hole from unknown body")
        return g1_holes, g2_holes

    def set_graph_hole_labels(self, holes, hole_graph, pair_graph, label_matrix, graph_order):
        """Set the hole labels for a single graph"""
        skip_curve_types = {
            "Line3DCurveType",
            "InfiniteLine3DCurveType",
            "Degenerate3DCurveType",
            "NurbsCurve3DCurveType"
        }
        skip_surface_types = {
            "NurbsSurfaceType"
        }
        hole_graph_node_count = len(hole_graph["nodes"])
        pair_graph_node_count = len(pair_graph["nodes"])
        g_holes = []
        for hole in holes:
            # Skip straight lines
            if "curve_type" in hole and hole["curve_type"] in skip_curve_types:
                continue
            if "surface_type" in hole and hole["surface_type"] in skip_surface_types:
                continue
            hole_entity_index = hole["index"]
            hole_entity_type = hole["type"]
            hole_graph_index = hole_entity_index
            if hole_entity_type == "BRepEdge":
                # Offset the index by the number of faces
                # as edges come after faces in the graph
                hole_graph_index += hole_graph["properties"]["face_count"]
            assert hole_graph_index < hole_graph_node_count
            hole_mates = self.find_hole_mates(hole, hole_graph_index, hole_graph, pair_graph)
            for hole_mate in hole_mates:
                assert hole_mate < pair_graph_node_count
                # Only set non-joint labels, don't overwrite others
                if graph_order == 1:
                    if label_matrix[hole_graph_index][hole_mate] == self.label_map["Non-joint"]:
                        label_matrix[hole_graph_index][hole_mate] = self.label_map["Hole"]
                        g_holes.append({
                            "target": hole_graph_index,
                            "pair": hole_mate,
                        })
                elif graph_order == 2:
                    if label_matrix[hole_mate][hole_graph_index] == self.label_map["Non-joint"]:
                        label_matrix[hole_mate][hole_graph_index] = self.label_map["Hole"]
                        g_holes.append({
                            "target": hole_graph_index,
                            "pair": hole_mate,
                        })
        return g_holes

    def reverse_graph_hole_labels(self, g_holes):
        """Return the reverse target/pair graph hole labels"""
        rev_holes = []
        for g_hole in g_holes:
            rev_holes.append({
                "target": g_hole["pair"],
                "pair": g_hole["target"]
            })
        return rev_holes

    def find_hole_mates(self, hole_entity, hole_graph_index, hole_graph, pair_graph):
        """Find the mates for a hole in the pair graph
            returning the node indices in the pair graph"""
        hole_entity_type = hole_entity["type"]
        node_matches = []
        hole_node = hole_graph["nodes"][hole_graph_index]
        # Loop over the nodes with the same entity type
        if hole_entity_type == "BRepFace":
            pair_start_index = 0
            pair_end_index = pair_graph["properties"]["face_count"]
        elif hole_entity_type == "BRepEdge":
            pair_start_index = pair_graph["properties"]["face_count"]
            pair_end_index = len(pair_graph["nodes"])
        for pair_node_index in range(pair_start_index, pair_end_index):
            pair_node = pair_graph["nodes"][pair_node_index]
            if self.is_node_match(hole_node, pair_node):
                # print(hole_graph_index, pair_node_index)
                node_matches.append(pair_node_index)
        return node_matches

    def is_node_match(self, node1, node2):
        """Check if the nodes match based on entity type and area/length"""
        node1_type = self.get_node_entity_type(node1)
        node2_type = self.get_node_entity_type(node2)
        if node1_type != node2_type:
            return False
        node1_area_length = self.get_node_area_length(node1)
        node2_area_length = self.get_node_area_length(node2)
        if math.isclose(node1_area_length, node2_area_length, rel_tol=self.rel_tol):
            return True
        else:
            return False

    def save_data_cache(self, data=None):
        """Save a pickle of the data"""
        data = {
            "graphs": self.graphs,
            "graph_files": self.graph_files,
            "has_holes": self.has_holes,
            "transforms": self.transforms
        }
        super().save_data_cache(data)

    def load_data_cache(self):
        """Load a pickle of the data"""
        # Call the base class to handle the basic loading
        data = super().load_data_cache()
        if not data:
            return False
        # Graph specific data
        self.graphs = data["graphs"][:self.cache_limit]
        if "graph_files" in data:
            self.graph_files = data["graph_files"][:self.cache_limit]
        if "has_holes" in data:
            self.has_holes = data["has_holes"][:self.cache_limit]
        if "transforms" in data:
            self.transforms = data["transforms"][:self.cache_limit]

        print(f"Data cache loaded from: {self.cache_file}")
        return True

    def plot(self, idx=0):
        """Debug functionality to plot a pair of joint graphs from a given index"""
        g1, g2, joint_graph = self.graphs[idx]
        label_matrix = joint_graph.edge_attr.view(joint_graph.num_nodes_graph1, joint_graph.num_nodes_graph2)
        joint_file = Path(self.files[idx])
        fig = plt.figure(figsize=plt.figaspect(0.5))
        fig.suptitle(joint_file.name, fontsize=16)
        # Get the labels as a dict where:
        # - key is the node index within the graph
        # - value is the label as an int
        joint_label_indices = torch.nonzero(label_matrix)
        g1_label_indices = joint_label_indices[:, 0].tolist()
        g2_label_indices = joint_label_indices[:, 1].tolist()
        g1_labels = {}
        g2_labels = {}
        for g1_label_index, g2_label_index in zip(g1_label_indices, g2_label_indices):
            label_value = int(label_matrix[g1_label_index][g2_label_index])
            if g1_label_index in g1_labels:
                # Make sure we don't overwrite the joint and ambiguous labels with equivalents
                if label_value == self.label_map["Joint"]:
                    g1_labels[g1_label_index] = label_value
                elif g1_labels[g1_label_index] != self.label_map["Joint"] and g1_labels[g1_label_index] != self.label_map["Ambiguous"]:
                    g1_labels[g1_label_index] = label_value
            else:
                g1_labels[g1_label_index] = label_value
            if g2_label_index in g2_labels:
                if label_value == self.label_map["Joint"]:
                    g2_labels[g2_label_index] = label_value
                elif g2_labels[g2_label_index] != self.label_map["Joint"] and g2_labels[g2_label_index] != self.label_map["Ambiguous"]:
                    g2_labels[g2_label_index] = label_value
            else:
                g2_labels[g2_label_index] = label_value
        # Graph 1
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.set_title("Body 1")
        self.plot_graph(g1, ax, labels=g1_labels)
        # Graph 2
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.set_title("Body 2")
        self.plot_graph(g2, ax, labels=g2_labels)
        plt.show()

    def plot_graph(self, graph, ax, labels=None, show_face_points=True, show_face_normals=False):
        """Debug functionality to plot a graph from a given index"""
        if labels is not None:
            color = np.array([[0.5, 0.5, 0.5, 0.2]])
        else:
            color = None
        plot_util.plot_uvsolid(
            graph.x,
            ax,
            labels=labels,
            points=show_face_points,
            normals=show_face_normals,
            color=color
        )


class JointGraphBatchDataLoader:
    """
    A custom dataloader to batch joint graphs into variable sized batches
    based on the maximum node count in a batch
    """
    def __init__(self, dataset, max_nodes_per_batch, shuffle=True, drop_last=True):
        self.dataset = dataset
        if shuffle:
            self.sampler = torch.utils.data.RandomSampler(dataset)
        else:
            self.sampler = torch.utils.data.SequentialSampler(dataset)
        self.max_nodes_per_batch = max_nodes_per_batch
        self.drop_last = drop_last

    def _make_batch_from_data_list(self, batch_list):
        """
        Make a PyG batch from the data list

        Args:
            batch_list (List[torch_geometric.data.Data]): List of PyG Data objects

        Returns:
            from torch_geometric.data.Batch: PyG Batch object
        """
        bg1 = Batch.from_data_list([x[0] for x in batch_list])
        bg2 = Batch.from_data_list([x[1] for x in batch_list])
        bjg = Batch.from_data_list([x[2] for x in batch_list])
        return [bg1, bg2, bjg]

    def __iter__(self):
        batch = []
        node_count = 0
        for idx in self.sampler:
            g1, g2, jg = self.dataset[idx]
            if g1.num_nodes + g2.num_nodes > self.max_nodes_per_batch:
                continue
            curr_node_count = g1.num_nodes + g2.num_nodes
            node_count += curr_node_count
            # TODO(pradeep): check if it's ok to not insert the data into the
            # batch if it exceeds the max. node count. This may lead to skipping some of the data.
            # But maybe randomization will take care of covering all data in the dataset.
            if node_count > self.max_nodes_per_batch and len(batch) > 0:
                batched_graphs = self._make_batch_from_data_list(batch)
                yield batched_graphs
                batch = []
                node_count = 0
            batch.append([g1, g2, jg])
        if node_count > 0 and not self.drop_last:
            batched_graphs = self._make_batch_from_data_list(batch)
            yield batched_graphs
