import pickle
import numpy as np
import json
from mmdet.datasets.coco import COCO


coco_annos = COCO('data/totaltext/instances_test.json')
# coco_annos = COCO('/data/ww/mmocr/data/ctw1500_words/instances_test.json')
results = pickle.load(open('work_dirs/tps_tot_e2e.pkl', 'rb'))
img_ids = coco_annos.get_img_ids()
output = []
for i, img_id in enumerate(img_ids):
    img_name = coco_annos.load_imgs([img_id])[0]['file_name']
    # idx = str(img_id).zfill(7)
    res = results[i]
    ins = res['boundary_result']
    recs = res['strs']
    for j in range(len(ins)):
        pts = np.array(ins[j][:-1]).reshape(-1,2).tolist()
        score = ins[j][-1]
        rec = recs[j]
        out = {
            # "image_id": int(img_name[5:-4]),
            "image_id": i,
            "category_id": 1,
            "polys": pts,
            "rec": rec,
            "score": score
        }
        output.append(out)


with open('tools/evaluation_det_e2e_offline_v3/totaltext_res/my_text_results.json', 'w') as f:
    json.dump(output, f)
