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
import torch



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

        # Acquire all filenames to use for access in dataloader
        self.image_set = image_set
        imgs_dir = os.path.join(root, image_set,"images")
        file_names_imgs = [os.path.join(imgs_dir, file) for file in sorted(os.listdir(imgs_dir))]
        self.images = file_names_imgs
        self.images_filenames = [os.path.join(file) for file in sorted(os.listdir(imgs_dir))]
        if image_set=='train':
            # Only acquire annotations if in training
            targets_dir = os.path.join(root, image_set,"annotations","xmls")
            file_names_targets = [os.path.join(targets_dir, file) for file in sorted(os.listdir(targets_dir))]
            self.targets = file_names_targets

        if image_set=='train':
            assert len(self.images) == len(self.targets)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target=None
        if self.image_set == 'train':
            xml = self.parse_xml(ET_parse(self.targets[index]).getroot())
            target = self.parse_dict(xml)

        
        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.image_set=='test':
            return img, self.images_filenames[index]
        if not len(target)==0: 
            if self.image_set=='train':
                return img, target, self.images_filenames[index]
            else:
                return img, self.images_filenames[index]
        else:
            del self.images[index]
            del self.targets[index]
            return self.__getitem__(index)


    def __len__(self) -> int:
        return len(self.images) 

    def collate_fn(self, batch): 
        """ Collate_fn for manual batching of images that may be of
        varying sizes  """ 
        if self.image_set=='train':
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
            name = [item[2] for item in batch]
            return [data, target, name]
        else:
            data = [item[0] for item in batch]
            name = [item[1] for item in batch]
            return [data, name]

    def parse_dict(self, xml_out_dict: dict) -> dict[str, Any]:
        """
        Args:
            xml_out_dict: resulting dict from processing xml-file with parse_xml
        Returns:
            Dict representation of xml object with relevant data from xml
        """
        in_dict = xml_out_dict['annotation']
        out_dict = {'boxes': [], 'labels': [], 'image_id': []}
        
        boxes = []
        labels = []
        for i in range(len(in_dict['object'])):
            
            match in_dict['object'][i]['name']:
                case 'D00':
                    obj_class = 1 # longitudinal crack
                case 'D10':
                    obj_class = 2 # transverse crack
                case 'D20':
                    obj_class = 3 # alligator crack
                case 'D40':
                    obj_class = 4 # pothole

            labels.append(obj_class)
            boxes.append([float(in_dict['object'][i]['bndbox']['xmin']), float(in_dict['object'][i]['bndbox']['ymin']),
            float(in_dict['object'][i]['bndbox']['xmax']), float(in_dict['object'][i]['bndbox']['ymax'])])
        
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.as_tensor(int(in_dict['filename'].replace('.jpg', '')[-6:]), dtype=torch.int64)
        if len(in_dict['object']) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        out_dict["boxes"] = boxes
        out_dict["labels"] = labels
        out_dict["image_id"] = image_id


        return out_dict
    @staticmethod
    def parse_xml(node: ET_Element) -> Dict[str, Any]:
        """ Parse xml file and return dict-representation """
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

