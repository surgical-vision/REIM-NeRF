from .blender import BlenderDataset
from .llff import LLFFDataset
from .reim_json import REIMNeRFDataset as REIMNeRFDataset_json
from .reim_json_render import REIMNeRFDataset as REIMNeRFDataset_json_render

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'reim_json': REIMNeRFDataset_json,
                'reim_json_render':REIMNeRFDataset_json_render}