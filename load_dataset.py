import collections
import os
from xml.etree.ElementTree import Element as ET_Element
import torchvision
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image




class RoadCracksDetection(torchvision.datasets.VisionDataset):

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform) 


        imgs_dir = os.path.join(root, image_set,"images")

        file_names_imgs = [os.path.join(imgs_dir, file) for file in sorted(os.listdir(imgs_dir))]
        
        targets_dir = os.path.join(root, image_set,"annotations","xmls")
        file_names_targets = [os.path.join(targets_dir, file) for file in sorted(os.listdir(targets_dir))]
        self.images = file_names_imgs
        self.targets = file_names_targets


        assert len(self.images) == len(self.targets)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        xml = self.parse_xml(ET_parse(self.targets[index]).getroot())
        target = self.parse_dict(xml)

        print("###")
        print(target)

        
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target   

    def __len__(self) -> int:
        return len(self.images)   

    def parse_dict(self, xml_out_dict: dict) -> dict[str, Any]:
        in_dict = xml_out_dict['annotation']
        out_dict = {'labels': [], 'boxes': [], 'image_id': [], 'area': [], 'iscrowd': []}
        for obj in in_dict['object']:
            out_dict['labels'].append(obj['name'])
            out_dict['boxes'].append([obj['bndbox']['xmin'], obj['bndbox']['ymin'], obj['bndbox']['xmax'], obj['bndbox']['ymax']])
            out_dict['image_id'].append(int(in_dict['filename'][:10].replace('.jpg', '')))
            out_dict['area'].append((obj['bndbox']['xmax']-obj['bndbox']['xmin'])*(obj['bndbox']['ymax']- obj['bndbox']['ymin']))
            out_dict['iscrowd'].append(False)
        print(in_dict)
        print("Out dict", out_dict)
        return out_dict
    @staticmethod
    def parse_xml(node: ET_Element) -> Dict[str, Any]:
        xml_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(RoadCracksDetection.parse_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            xml_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                xml_dict[node.tag] = text

        return xml_dict  

