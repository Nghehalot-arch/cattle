import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import register_coco_instances

RED = (0,255,0)

def make_infer_split(root, split_name, target_folder, sample_annotation_path):
    infer_annotation_path = os.path.join(root, 'annotations/{}_infer.json'.format(split_name))
    # if os.path.exists(infer_annotation_path):
    #     return
    import json
    infer_image_path = os.path.join(root, target_folder)
    with open(os.path.join(root, sample_annotation_path)) as f:
        sample_annotation = json.load(f)
    infer_annotation = sample_annotation.copy()
    infer_annotation['images'] = []
    infer_annotation['annotations'] = []
    all_images_target = sorted(os.listdir(infer_image_path))
    for image_id, image in enumerate(all_images_target):
        infer_annotation['images'].append({
            'license': None,
            'file_name': image,
            'coco_url': None,
            "height": 1440,
            "width": 2560,
            'id': image_id
        })
        infer_annotation['annotations'].append({
            'segmentation': None,
            'num_keypoints': 13,
            'area': 0,
            'iscrowd': 0,
            'keypoints': [0,0,0]*13,
            'image_id': image_id,
            'bbox': [0,0,0,0],
            'category_id': 1,
            'id': image_id,
            'inmodal_bbox': None,
            'inmodal_seg': None
        })
    with open(infer_annotation_path, 'w') as f:
        json.dump(infer_annotation, f)


def add_keypoint_metadata(dataset_name):
    meta = MetadataCatalog.get(dataset_name)
    meta.thing_classes = ['cattle']
    meta.keypoint_names =  ['p1','p2','p3','p4','p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13']
    meta.keypoint_flip_map = [['p1','p1'],['p2', 'p3'],['p4','p5'],['p6','p7'],['p8','p9'],['p10','p11'],['p12','p13']]


def register_keypoints_dataset(root, coco_folder='coco_format', prefix='keypoints'):
    root = os.path.join(root, 'keypoints', coco_folder)
    splits = ['train', 'val', 'test', 'demo']
    for split in splits:
        dataset_name = "{}_{}".format(prefix, split)
        register_coco_instances(dataset_name, {}, 
            os.path.join(root, "annotations/{}.json".format(split)),
            os.path.join(root, "{}_imgs".format(split))
        )
        add_keypoint_metadata(dataset_name)
        # meta.keypoint_connection_rules = [['p1','p2',RED], ['p1','p3',RED],['p2','p4',RED],['p4','p5',RED], ['p5','p6',RED],['p6','p7',RED],['p7','p8',RED],['p7','p11',RED],['p11','p13',RED],['p11','p10',RED],
                                        #   ['p10','p12',RED],['p12','p13',RED],['p3','p12',RED],['p3','p9',RED]]

    test_infer_folder = 'test_imgs'
    make_infer_split(root, 'test', test_infer_folder, 'annotations/test.json')
    register_coco_instances("{}_test_infer".format(prefix), {},
        os.path.join(root, "annotations/test_infer.json"),
        os.path.join(root, test_infer_folder)   
    )
    add_keypoint_metadata("{}_test_infer".format(prefix))

    demo_infer_folder = 'demo_imgs'
    make_infer_split(root, 'demo', demo_infer_folder, 'annotations/demo.json')
    register_coco_instances("{}_demo_infer".format(prefix), {},
        os.path.join(root, "annotations/demo_infer.json"),
        os.path.join(root, demo_infer_folder)
    )
    add_keypoint_metadata("{}_demo_infer".format(prefix))


_root = os.getenv("CATTLE_DATASETS", "datasets")
register_keypoints_dataset(_root)
register_keypoints_dataset(_root, coco_folder='thermal_coco_format', prefix='thermal_keypoints')
